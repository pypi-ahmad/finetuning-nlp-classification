"""
active_learning.py
─────────────────────────────────────────────────────────────────────────────
Uncertainty-sampling utilities for the relabel/retrain loop.

The premise: labelling complaints is expensive, so spend the budget where the
model is *least sure*. We score an unlabeled pool, rank by uncertainty, take the
top-`budget`, "relabel" them (here: reveal the gold label — an oracle stand-in),
add them to the training set, and retrain. We compare **uncertainty sampling**
against a **random** control to show the selection actually helps.

Uncertainty measures (on the L1 stage probabilities):
  * least_confidence : 1 - max_p           (simple, intuitive)
  * margin           : 1 - (p1 - p2)       (top-2 closeness)
  * entropy          : Shannon entropy      (spread across all classes)
"""

from __future__ import annotations

import numpy as np


def least_confidence(probs: np.ndarray) -> np.ndarray:
    return 1.0 - probs.max(axis=1)


def margin_uncertainty(probs: np.ndarray) -> np.ndarray:
    part = np.sort(probs, axis=1)[:, -2:]
    return 1.0 - (part[:, 1] - part[:, 0])


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


_STRATEGIES = {
    "least_confidence": least_confidence,
    "margin": margin_uncertainty,
    "entropy": entropy,
}


def select(probs: np.ndarray, budget: int, strategy: str = "entropy",
           rng: np.random.Generator | None = None) -> np.ndarray:
    """Return indices of the `budget` most-uncertain rows (or random if strategy='random')."""
    n = len(probs)
    budget = min(budget, n)
    if strategy == "random":
        rng = rng or np.random.default_rng(0)
        return rng.choice(n, size=budget, replace=False)
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    u = _STRATEGIES[strategy](probs)
    return np.argsort(-u)[:budget]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(9), size=100)
    idx = select(probs, 10, "entropy")
    print("most-uncertain idx:", idx[:10])
    print("their mean max-prob:", round(probs[idx].max(1).mean(), 3),
          "vs overall:", round(probs.max(1).mean(), 3))
