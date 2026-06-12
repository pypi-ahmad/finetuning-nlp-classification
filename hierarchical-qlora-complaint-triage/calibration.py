"""
calibration.py
─────────────────────────────────────────────────────────────────────────────
Turn raw candidate-scores into *trustworthy* probabilities.

Two options (both fit on validation, applied to test):
  * Temperature scaling — one scalar T; softmax(scores / T). Preserves the
    argmax (so accuracy is unchanged) and only rescales confidence. The standard
    first choice for neural-net calibration.
  * Isotonic regression — a non-parametric monotonic map confidence→accuracy.
    More flexible, can fix non-uniform miscalibration, but can overfit small
    validation sets.

`expected_calibration_error` quantifies the gap between confidence and accuracy.
"""

from __future__ import annotations

import numpy as np
import torch


def softmax_T(scores: np.ndarray, T: float) -> np.ndarray:
    s = scores / T
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    return p / p.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(1); pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(val_scores: np.ndarray, val_labels: np.ndarray,
                    max_iter: int = 200) -> float:
    """Optimise scalar T to minimise NLL on validation (LBFGS via torch)."""
    s = torch.tensor(val_scores, dtype=torch.float32)
    y = torch.tensor(val_labels, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True)  # optimise log T (keeps T > 0)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=max_iter)
    nll = torch.nn.functional.cross_entropy

    def closure():
        opt.zero_grad()
        loss = nll(s / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().item())


def fit_isotonic(val_conf: np.ndarray, val_correct: np.ndarray):
    """Return an isotonic map from confidence → empirical accuracy."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val_conf, val_correct)
    return iso


def reliability_points(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    return np.array(xs), np.array(ys)


if __name__ == "__main__":
    # synthetic over-confident classifier → temperature should be > 1 and cut ECE
    rng = np.random.default_rng(0)
    N, K = 2000, 9
    true = rng.integers(0, K, N)
    logits = rng.normal(0, 1, (N, K))
    logits[np.arange(N), true] += 2.0           # correct class favoured
    scores = logits * 3.0                        # inflate → over-confident
    p0 = softmax_T(scores, 1.0)
    T = fit_temperature(scores[:1000], true[:1000])
    p1 = softmax_T(scores[1000:], T)
    print(f"fitted T = {T:.2f}")
    print(f"ECE before {expected_calibration_error(p0[1000:], true[1000:]):.3f}"
          f"  after {expected_calibration_error(p1, true[1000:]):.3f}")
