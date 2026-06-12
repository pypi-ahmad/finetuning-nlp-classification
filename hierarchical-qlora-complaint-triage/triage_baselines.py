"""
triage_baselines.py
─────────────────────────────────────────────────────────────────────────────
A classical, CPU-only sanity baseline: TF-IDF + multinomial Logistic Regression.

This is the bar the LLM must beat to justify QLoRA. It is genuinely strong on
bag-of-words signals (CFPB narratives are keyword-rich), so it keeps the project
honest — if a linear model on n-grams already nails L1, the LLM's value is in the
harder, context-dependent L2 step and in calibration.

Reports L1 macro-F1 and (flat) L2 macro-F1 on CFPB rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline


def _pipe() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=3, ngram_range=(1, 2),
                                  max_features=50000, stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
    ])


@dataclass
class BaselineResult:
    l1_macro_f1: float
    l1_accuracy: float
    l2_macro_f1: float
    l2_accuracy: float


def run_tfidf_baseline(train_df, test_df) -> tuple[BaselineResult, dict]:
    # ── L1 (all rows) ──────────────────────────────────────────────────────
    l1 = _pipe().fit(train_df["text"], train_df["l1"])
    l1_pred = l1.predict(test_df["text"])
    l1_f1 = f1_score(test_df["l1"], l1_pred, average="macro")
    l1_acc = accuracy_score(test_df["l1"], l1_pred)

    # ── L2 (CFPB rows only; flat over all issues) ───────────────────────────
    tr2 = train_df[(train_df["source"] == "cfpb") & train_df["l2"].notna()]
    te2 = test_df[(test_df["source"] == "cfpb") & test_df["l2"].notna()]
    l2 = _pipe().fit(tr2["text"], tr2["l2"])
    l2_pred = l2.predict(te2["text"])
    l2_f1 = f1_score(te2["l2"], l2_pred, average="macro")
    l2_acc = accuracy_score(te2["l2"], l2_pred)

    res = BaselineResult(l1_f1, l1_acc, l2_f1, l2_acc)
    models = {"l1": l1, "l2": l2}
    return res, models


if __name__ == "__main__":
    from triage_data import build_dataframe, make_splits
    df, hier = build_dataframe()
    sp = make_splits(df)
    res, _ = run_tfidf_baseline(sp["train"], sp["test"])
    print("TF-IDF + LogReg baseline")
    print(f"  L1  macro-F1 {res.l1_macro_f1:.3f} | acc {res.l1_accuracy:.3f}")
    print(f"  L2  macro-F1 {res.l2_macro_f1:.3f} | acc {res.l2_accuracy:.3f}  (CFPB, flat)")
