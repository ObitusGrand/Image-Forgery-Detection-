"""
DeepForensics — Probability Calibration & Weighted Ensemble Utilities
======================================================================
Implements deep learning post-hoc calibration and calibrated multi-model
weighted ensembling:
    - Temperature Scaling (Guo et al., 2017): optimizes scalar T > 0 on
      validation logits to minimize Negative Log-Likelihood (NLL).
    - Expected Calibration Error (ECE) & reliability bin statistics.
    - Weighted ensemble probability fusion:
          p_ens = w * p_new_cal + (1 - w) * p_old_cal
    - JSON config persistence (`ensemble_config.json`) shared between
      `calibrate_ensemble.py`, `evaluate_model.py`, and `app.py`.

CLI self-test:
    python -m utils.calibration
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


EPS = 1e-6
DEFAULT_ENSEMBLE_CONFIG_PATH = Path("ensemble_config.json")


# ── Logit / Probability Transforms ───────────────────────────────────────────

def prob_to_logit(probs: np.ndarray) -> np.ndarray:
    """Convert probabilities in [0, 1] to numerically safe log-odds (logits)."""
    p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def logit_to_prob(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid transform from logits to [0, 1]."""
    z = np.clip(np.asarray(logits, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling T > 0 to probabilities via logit rescaling."""
    t = max(float(temperature), 1e-3)
    if abs(t - 1.0) < 1e-6:
        return np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    logits = prob_to_logit(probs)
    return logit_to_prob(logits / t)


# ── Temperature Scaling Optimization (Pure NumPy — No SciPy Required) ────────

def binary_nll(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Compute binary negative log-likelihood (cross-entropy loss)."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temperature(
    y_true: np.ndarray,
    probs: np.ndarray,
    t_min: float = 0.25,
    t_max: float = 5.0,
) -> float:
    """Find scalar temperature T in [t_min, t_max] minimizing validation NLL.

    Uses coarse log-grid search followed by golden-section refinement so it
    has zero external optimization dependencies and guarantees global convergence
    for the convex 1D temperature scaling objective.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    logits = prob_to_logit(np.asarray(probs, dtype=np.float64).ravel())

    def objective(t: float) -> float:
        return binary_nll(y, logit_to_prob(logits / t))

    # Coarse grid search over log-spaced temperatures
    grid = np.geomspace(t_min, t_max, num=120)
    losses = [objective(float(t)) for t in grid]
    best_idx = int(np.argmin(losses))

    lo = float(grid[max(0, best_idx - 1)])
    hi = float(grid[min(len(grid) - 1, best_idx + 1)])

    # Golden-section refinement
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    for _ in range(40):
        c = hi - (hi - lo) * inv_phi
        d = lo + (hi - lo) * inv_phi
        if objective(c) < objective(d):
            hi = d
        else:
            lo = c

    return float(0.5 * (lo + hi))


# ── Expected Calibration Error (ECE) ─────────────────────────────────────────

def compute_ece(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Compute Expected Calibration Error (ECE) and reliability bin statistics.

    Returns dict with:
        - ``ece``: float in [0, 1] (lower is better calibrated)
        - ``bin_edges``: list of bin edges
        - ``bin_confidences``: mean predicted probability per bin
        - ``bin_accuracies``: empirical positive fraction per bin
        - ``bin_counts``: sample count per bin
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), 0.0, 1.0)
    n = len(y)
    if n == 0:
        return {
            "ece": 0.0,
            "bin_edges": [],
            "bin_confidences": [],
            "bin_accuracies": [],
            "bin_counts": [],
        }

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(np.sum(mask))
        bin_counts.append(count)
        if count > 0:
            avg_conf = float(np.mean(p[mask]))
            avg_acc = float(np.mean(y[mask]))
            ece += (count / n) * abs(avg_acc - avg_conf)
            bin_confidences.append(avg_conf)
            bin_accuracies.append(avg_acc)
        else:
            bin_confidences.append(float(0.5 * (lo + hi)))
            bin_accuracies.append(0.0)

    return {
        "ece": float(ece),
        "bin_edges": edges.tolist(),
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }


# ── Ensemble Config Dataclass ────────────────────────────────────────────────

@dataclass
class EnsembleConfig:
    """Calibrated ensemble parameters persisted to ``ensemble_config.json``."""

    weight_new: float = 0.75
    weight_legacy: float = 0.25
    threshold: float = 0.50
    temp_new: float = 1.0
    temp_legacy: float = 1.0
    default_mode: str = "ensemble"
    use_tta_default: bool = False
    calibrated: bool = False
    metrics: Optional[dict] = None

    def combine(self, p_new: float, p_legacy: float) -> tuple[float, float, float]:
        """Calibrate both probabilities and return (p_new_cal, p_legacy_cal, p_ensemble)."""
        p_new_cal = float(apply_temperature(np.array([p_new]), self.temp_new)[0])
        p_legacy_cal = float(apply_temperature(np.array([p_legacy]), self.temp_legacy)[0])
        p_ens = float(self.weight_new * p_new_cal + self.weight_legacy * p_legacy_cal)
        return p_new_cal, p_legacy_cal, float(np.clip(p_ens, 0.0, 1.0))

    def save(self, path: Path = DEFAULT_ENSEMBLE_CONFIG_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_ENSEMBLE_CONFIG_PATH) -> "EnsembleConfig":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ── CLI Self-Test ────────────────────────────────────────────────────────────

def _self_test() -> None:
    print("=" * 60)
    print("  DeepForensics Calibration & Ensemble Self-Test")
    print("=" * 60)
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=500)
    # Overconfident logits (scale 3x)
    raw_logits = (y * 2 - 1) * 3.0 + rng.normal(0, 1.5, size=500)
    probs = logit_to_prob(raw_logits)

    ece_before = compute_ece(y, probs)["ece"]
    nll_before = binary_nll(y, probs)
    temp = fit_temperature(y, probs)
    cal_probs = apply_temperature(probs, temp)
    ece_after = compute_ece(y, cal_probs)["ece"]
    nll_after = binary_nll(y, cal_probs)

    print(f"  Fitted Temperature : {temp:.3f}")
    print(f"  NLL : {nll_before:.4f} -> {nll_after:.4f}")
    print(f"  ECE : {ece_before:.4f} -> {ece_after:.4f}")
    assert nll_after <= nll_before + 1e-6

    cfg = EnsembleConfig(weight_new=0.8, weight_legacy=0.2, temp_new=temp, temp_legacy=1.0)
    pn, pl, pe = cfg.combine(0.9, 0.3)
    assert 0.0 <= pe <= 1.0
    print(f"  Ensemble combine(0.9, 0.3) -> p_new={pn:.3f}, p_leg={pl:.3f}, p_ens={pe:.3f}")
    print("-" * 60)
    print("  All calibration checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
