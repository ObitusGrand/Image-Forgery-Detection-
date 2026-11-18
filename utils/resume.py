"""
DeepForensics — Crash-Safe Atomic Checkpoint & Auto-Resume Manager
==================================================================
Provides stage-level and epoch-level fault tolerance for long GPU training
runs in WSL2 (`train_model.py` and `train_multibranch.py`).

How it works:
    - Stores per-run resume artifacts under ``checkpoints/<model_stem>/``:
        * ``state.json``          — atomic JSON state (stage, epoch, best_val_auc, wait, error)
        * ``latest_stage1.keras`` — full model + optimizer state after every Stage 1 epoch
        * ``latest_stage2.keras`` — full model + optimizer state after every Stage 2 epoch
    - Uses atomic writes (write to ``*.tmp`` then ``os.replace``) so a sudden
      power loss, WSL2 disconnect, or CUDA OOM mid-save NEVER corrupts the
      last good checkpoint or best model file.
    - Preserves ``best_val_auc`` and EarlyStopping ``wait`` counter across
      restarts so resumed runs never overwrite a better previous model.
    - On re-running the exact same training command after an error/interrupt:
        * If Stage 1 stopped at epoch k, resumes Stage 1 from ``initial_epoch=k``.
        * If Stage 1 finished and Stage 2 stopped at epoch m, skips Stage 1
          and resumes Stage 2 from ``initial_epoch=m``.
        * Pass ``--fresh`` to clear resume state and train from scratch.

CLI self-test (simulates mid-Stage-1 & mid-Stage-2 crashes and verifies resume):
    python -m utils.resume
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tensorflow import keras


DEFAULT_CHECKPOINT_ROOT = Path("checkpoints")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _atomic_save_model(model: keras.Model, target_path: Path) -> None:
    """Save a Keras model atomically via a temporary ``.keras`` file."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # Keras 3 requires the filename to end with `.keras`
    tmp_path = target_path.with_name(f".{target_path.stem}.tmp.keras")
    model.save(tmp_path)
    os.replace(tmp_path, target_path)


class TrainingResumeManager:
    """Manages persistent stage/epoch state and atomic checkpoints for one model."""

    def __init__(
        self,
        model_path: Path,
        checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
        fresh: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.run_dir = Path(checkpoint_root) / self.model_path.stem
        self.state_path = self.run_dir / "state.json"
        self.tuner_dir = self.run_dir / "tuner"
        self.best_hp_path = self.model_path.with_name(f"best_hyperparams_{self.model_path.stem}.json")

        if fresh and self.run_dir.exists():
            shutil.rmtree(self.run_dir, ignore_errors=True)
            print(f"[RESUME] --fresh requested: cleared checkpoint & tuner state at {self.run_dir}")

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _default_state(self) -> dict:
        return {
            "model_path": str(self.model_path),
            "tuning_phase1_completed": False,
            "tuning_phase1_best_hp": None,
            "tuning_phase2_results": [],
            "tuning_phase2_completed": False,
            "best_hyperparams": None,
            "stage1_completed": False,
            "stage1_epoch": 0,
            "stage1_wait": 0,
            "stage2_completed": False,
            "stage2_epoch": 0,
            "stage2_wait": 0,
            "best_val_auc": -1.0,
            "completed": False,
            "last_error": None,
            "updated_at": _utc_now(),
        }

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                base = self._default_state()
                base.update(data)
                return base
            except Exception as exc:
                print(f"[RESUME] Warning: corrupt state.json ({exc}), resetting state.")
        return self._default_state()

    def save_state(self) -> None:
        self.state["updated_at"] = _utc_now()
        _atomic_write_json(self.state_path, self.state)

    def latest_checkpoint_path(self, stage: int) -> Path:
        return self.run_dir / f"latest_stage{stage}.keras"

    def has_resumable_checkpoint(self, stage: int) -> bool:
        epoch_key = f"stage{stage}_epoch"
        ckpt = self.latest_checkpoint_path(stage)
        return (
            not self.state.get(f"stage{stage}_completed", False)
            and int(self.state.get(epoch_key, 0)) > 0
            and ckpt.exists()
        )

    def get_initial_epoch(self, stage: int) -> int:
        if self.has_resumable_checkpoint(stage):
            return int(self.state.get(f"stage{stage}_epoch", 0))
        return 0

    def get_wait(self, stage: int) -> int:
        if self.has_resumable_checkpoint(stage):
            return int(self.state.get(f"stage{stage}_wait", 0))
        return 0

    def get_best_val_auc(self) -> float:
        return float(self.state.get("best_val_auc", -1.0))

    def save_best_hyperparams(self, params: dict) -> None:
        """Persist winning Bayesian hyperparameters both in state.json and standalone JSON."""
        self.state["best_hyperparams"] = params
        self.state["tuning_phase2_completed"] = True
        self.state["last_error"] = None
        self.save_state()
        _atomic_write_json(self.best_hp_path, params)

    def load_saved_hyperparams(self) -> Optional[dict]:
        """Return previously found best hyperparameters if available."""
        if self.state.get("best_hyperparams"):
            return dict(self.state["best_hyperparams"])
        if self.best_hp_path.exists():
            try:
                return json.loads(self.best_hp_path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def record_tuning_interruption(self, phase: str, exc: BaseException) -> None:
        """Log crash/interrupt during Bayesian hyperparameter search."""
        err_type = type(exc).__name__
        self.state["last_error"] = {
            "stage": f"tuning_{phase}",
            "error_type": err_type,
            "message": str(exc),
            "timestamp": _utc_now(),
        }
        self.save_state()
        print("\n" + "!" * 68)
        print(f"  [INTERRUPTED] {err_type} during Bayesian Tuning ({phase})")
        print(f"  Error  : {exc}")
        print(f"  Tuner  : {self.tuner_dir} (completed trials preserved via overwrite=False)")
        print("  Action : Re-run the exact same command to resume tuning automatically.")
        print("           (Or pass --fresh to wipe tuner + checkpoints and restart.)")
        print("!" * 68)
        if not isinstance(exc, KeyboardInterrupt):
            traceback.print_exc()

    def mark_stage_completed(self, stage: int) -> None:
        self.state[f"stage{stage}_completed"] = True
        self.state["last_error"] = None
        self.save_state()

    def mark_all_completed(self, cleanup_latest: bool = True) -> None:
        self.state["completed"] = True
        self.state["last_error"] = None
        self.save_state()
        if cleanup_latest:
            for st in (1, 2):
                p = self.latest_checkpoint_path(st)
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass

    def record_interruption(
        self,
        stage: int,
        exc: BaseException,
        model: Optional[keras.Model] = None,
    ) -> None:
        """Log crash/interrupt metadata so the next run resumes cleanly."""
        epoch_done = int(self.state.get(f"stage{stage}_epoch", 0))
        err_type = type(exc).__name__
        self.state["last_error"] = {
            "stage": stage,
            "completed_epoch": epoch_done,
            "error_type": err_type,
            "message": str(exc),
            "timestamp": _utc_now(),
        }
        self.save_state()

        print("\n" + "!" * 68)
        print(f"  [INTERRUPTED] {err_type} during Stage {stage} (last saved epoch: {epoch_done})")
        print(f"  Error      : {exc}")
        if self.latest_checkpoint_path(stage).exists():
            print(f"  Checkpoint : {self.latest_checkpoint_path(stage)}")
        elif self.model_path.exists():
            print(f"  Best model : {self.model_path}")
        print("  Action     : Re-run the exact same command to resume automatically.")
        print("               (Or pass --fresh to restart from Stage 1, Epoch 0.)")
        print("!" * 68)
        if not isinstance(exc, KeyboardInterrupt):
            traceback.print_exc()


class ResumableTrainingCallback(keras.callbacks.Callback):
    """Unified atomic best-model saver, per-epoch resume saver, and EarlyStopping.

    Replaces separate ModelCheckpoint + EarlyStopping callbacks so resumed
    runs inherit exact ``best_val_auc`` and ``wait`` counts with zero drift.
    """

    def __init__(
        self,
        manager: TrainingResumeManager,
        stage: int,
        monitor: str = "val_auc",
        patience: int = 7,
    ) -> None:
        super().__init__()
        self.manager = manager
        self.stage = int(stage)
        self.monitor = monitor
        self.patience = int(patience)
        self.best_val: float = manager.get_best_val_auc()
        self.wait: int = manager.get_wait(self.stage)
        self.stopped_early: bool = False

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        logs = logs or {}
        metric_val = logs.get(self.monitor)
        if metric_val is None:
            metric_val = logs.get("val_accuracy", 0.0)
        metric_val = float(metric_val)

        # 1. Track & atomically save global best model across stages/restarts
        if metric_val > self.best_val:
            prev = self.best_val
            self.best_val = metric_val
            self.wait = 0
            _atomic_save_model(self.model, self.manager.model_path)
            prev_str = f"{prev:.4f}" if prev >= 0 else "init"
            print(
                f"\n  [RESUME-CKPT] Stage {self.stage} Epoch {epoch + 1}: "
                f"{self.monitor} improved {prev_str} -> {metric_val:.4f} "
                f"(saved {self.manager.model_path.name})"
            )
        else:
            self.wait += 1
            print(
                f"\n  [RESUME-CKPT] Stage {self.stage} Epoch {epoch + 1}: "
                f"{self.monitor}={metric_val:.4f} did not beat {self.best_val:.4f} "
                f"(patience {self.wait}/{self.patience})"
            )

        # 2. Atomically save latest full checkpoint (weights + optimizer + LR step)
        latest_path = self.manager.latest_checkpoint_path(self.stage)
        _atomic_save_model(self.model, latest_path)

        # 3. Atomically persist resume state JSON
        self.manager.state[f"stage{self.stage}_epoch"] = epoch + 1
        self.manager.state[f"stage{self.stage}_wait"] = self.wait
        self.manager.state["best_val_auc"] = self.best_val
        self.manager.state["last_error"] = None
        self.manager.state[f"stage{self.stage}_last_metrics"] = {
            k: float(v) for k, v in logs.items() if isinstance(v, (int, float))
        }
        self.manager.save_state()

        # 4. Early stopping check
        if self.patience > 0 and self.wait >= self.patience:
            self.stopped_early = True
            self.model.stop_training = True
            print(
                f"  [EARLY STOP] Stage {self.stage} triggered at epoch {epoch + 1} "
                f"(best {self.monitor}={self.best_val:.4f})."
            )


def load_resumable_model(
    ckpt_path: Path,
    custom_objects: Optional[dict] = None,
) -> keras.Model:
    """Load a saved checkpoint model with full optimizer state for exact resume."""
    return keras.models.load_model(ckpt_path, custom_objects=custom_objects or {})


def clear_gpu_memory() -> None:
    """Release Keras graph & Python references between tuning trials (12GB VRAM safety)."""
    keras.backend.clear_session()
    gc.collect()


def create_vram_safe_bayesian_tuner(
    hypermodel,
    directory: Path,
    project_name: str,
    max_trials: int = 6,
    seed: int = 42,
    objective_name: str = "val_auc",
):
    """Create a resumable Keras Tuner BayesianOptimization instance with per-trial VRAM cleanup.

    Uses ``overwrite=False`` so if a crash/interrupt happens mid-search,
    re-running the exact same script resumes from the next unfinished trial.
    """
    try:
        import keras_tuner as kt
    except ImportError as exc:
        raise ImportError(
            "keras-tuner is required for Bayesian hyperparameter optimization. "
            "Install it in your WSL2 environment with: pip install 'keras-tuner>=1.4.6' "
            "(or pass --no-tune to train directly with default hyperparameters)."
        ) from exc

    class _VRAMSafeBayesianOptimization(kt.BayesianOptimization):
        def run_trial(self, trial, *args, **kwargs):
            try:
                return super().run_trial(trial, *args, **kwargs)
            finally:
                clear_gpu_memory()

    directory.mkdir(parents=True, exist_ok=True)
    return _VRAMSafeBayesianOptimization(
        hypermodel=hypermodel,
        objective=kt.Objective(objective_name, direction="max"),
        max_trials=max_trials,
        seed=seed,
        directory=str(directory),
        project_name=project_name,
        overwrite=False,
    )


# ── CLI Self-Test (Simulated Crash & Auto-Resume) ────────────────────────────

def _self_test() -> None:
    import tempfile
    import numpy as np

    print("=" * 64)
    print("  DeepForensics Crash-Safe Auto-Resume Self-Test")
    print("=" * 64)

    class CrashAfterFirstEpoch(keras.callbacks.Callback):
        def __init__(self, target_epoch: int):
            super().__init__()
            self.target_epoch = target_epoch

        def on_epoch_end(self, epoch, logs=None):
            if epoch == self.target_epoch:
                raise RuntimeError(f"Simulated crash right after epoch {epoch + 1}")

    rng = np.random.default_rng(42)
    x = rng.normal(size=(16, 8)).astype(np.float32)
    y = rng.integers(0, 2, size=(16, 1)).astype(np.float32)

    def make_tiny():
        inp = keras.Input(shape=(8,))
        h = keras.layers.Dense(16, activation="relu")(inp)
        out = keras.layers.Dense(1, activation="sigmoid")(h)
        m = keras.Model(inp, out)
        m.compile(optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=[keras.metrics.AUC(name="auc")])
        return m

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        model_path = tmp / "test_resume_model.keras"
        ckpt_root = tmp / "checkpoints"

        # ── Attempt 1: Stage 1 crashes after epoch 1 of 3 ───────────────
        mgr1 = TrainingResumeManager(model_path, checkpoint_root=ckpt_root)
        cb1 = ResumableTrainingCallback(mgr1, stage=1, monitor="val_auc", patience=5)
        m1 = make_tiny()
        crashed = False
        try:
            m1.fit(x, y, validation_data=(x, y), epochs=3, initial_epoch=0, callbacks=[cb1, CrashAfterFirstEpoch(0)], verbose=0)
        except RuntimeError as exc:
            crashed = True
            mgr1.record_interruption(1, exc, m1)
        assert crashed and mgr1.has_resumable_checkpoint(1)
        assert mgr1.get_initial_epoch(1) == 1
        print("  [PASS] Stage 1 simulated crash saved state at epoch 1/3")

        # ── Attempt 2: Resume Stage 1 from epoch 1 -> 3 ─────────────────
        mgr2 = TrainingResumeManager(model_path, checkpoint_root=ckpt_root)
        assert mgr2.has_resumable_checkpoint(1) and mgr2.get_initial_epoch(1) == 1
        m2 = load_resumable_model(mgr2.latest_checkpoint_path(1))
        cb2 = ResumableTrainingCallback(mgr2, stage=1, monitor="val_auc", patience=5)
        m2.fit(x, y, validation_data=(x, y), epochs=3, initial_epoch=mgr2.get_initial_epoch(1), callbacks=[cb2], verbose=0)
        mgr2.mark_stage_completed(1)
        assert mgr2.state["stage1_completed"] is True
        assert mgr2.state["stage1_epoch"] == 3
        print("  [PASS] Stage 1 resumed from epoch 1 -> 3 and marked complete")

        # ── Attempt 3: Stage 2 crashes after epoch 1 of 2, then resumes ─
        mgr3 = TrainingResumeManager(model_path, checkpoint_root=ckpt_root)
        assert mgr3.state["stage1_completed"] is True  # Skips Stage 1!
        m3 = load_resumable_model(model_path)
        cb3 = ResumableTrainingCallback(mgr3, stage=2, monitor="val_auc", patience=5)
        try:
            m3.fit(x, y, validation_data=(x, y), epochs=2, initial_epoch=0, callbacks=[cb3, CrashAfterFirstEpoch(0)], verbose=0)
        except RuntimeError as exc:
            mgr3.record_interruption(2, exc, m3)
        assert mgr3.has_resumable_checkpoint(2) and mgr3.get_initial_epoch(2) == 1

        mgr4 = TrainingResumeManager(model_path, checkpoint_root=ckpt_root)
        m4 = load_resumable_model(mgr4.latest_checkpoint_path(2))
        cb4 = ResumableTrainingCallback(mgr4, stage=2, monitor="val_auc", patience=5)
        m4.fit(x, y, validation_data=(x, y), epochs=2, initial_epoch=mgr4.get_initial_epoch(2), callbacks=[cb4], verbose=0)
        mgr4.mark_stage_completed(2)
        mgr4.mark_all_completed(cleanup_latest=True)
        assert mgr4.state["completed"] is True
        assert model_path.exists()
        assert not mgr4.latest_checkpoint_path(1).exists()
        assert not mgr4.latest_checkpoint_path(2).exists()
        print("  [PASS] Stage 2 skipped Stage 1, resumed epoch 1 -> 2, cleaned up")

    print("-" * 64)
    print("  All crash-safe auto-resume checks passed!")
    print("=" * 64)


if __name__ == "__main__":
    _self_test()
