"""
triage_data.py
─────────────────────────────────────────────────────────────────────────────
Curate Banking77 + CFPB into a single, two-level (L1 product → L2 issue)
complaint-triage dataset, and build the label hierarchy.

Pipeline
    1. CFPB  — load the balanced API subset (download_cfpb.py), clean redacted
       narratives, keep L2 issues with enough samples (anti-sparsity), cap
       over-represented issues, truncate long text.
    2. Banking77 — fold in as **L1 augmentation only** (it is a clean card/bank
       benchmark with no CFPB-style issue labels). Intents are keyword-mapped to
       the three consumer-banking L1 products.
    3. Build HIERARCHY {L1: [L2, …]} and id maps.
    4. Stratified split into train / val / test / pool (pool = unlabeled-style
       reserve for the active-learning loop).

Output columns: text, l1, l2 (None for Banking77), source ("cfpb"|"banking77").
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from triage_config import CONFIG, TriageConfig

# ─────────────────────────────────────────────────────────────────────────────
# Banking77 intent → L1 product (keyword router; B77 is card/bank/transfer only)
# ─────────────────────────────────────────────────────────────────────────────
_B77_CARD = ("card", "atm", "contactless", "pin", "disposable", "visa",
             "mastercard", "apple_pay", "google_pay", "cash")
_B77_TRANSFER = ("transfer", "beneficiary", "receiving", "top_up", "topping",
                 "direct_debit", "exchange", "fiat")


def banking77_to_l1(intent: str) -> str:
    s = intent.lower()
    # transfer/funding terms win first (e.g. "top_up_by_card" → money movement)
    if any(k in s for k in _B77_TRANSFER):
        return "Money transfer & virtual currency"
    if any(k in s for k in _B77_CARD):
        return "Credit card / prepaid"
    return "Bank account / savings"   # identity / account / pin / passcode / etc.


# ─────────────────────────────────────────────────────────────────────────────
# CFPB narrative cleaning
# ─────────────────────────────────────────────────────────────────────────────
_REDACT = re.compile(r"\b[X]{2,}\b")          # CFPB redacts PII as XXXX
_WS = re.compile(r"\s+")


def clean_narrative(text: str | None, max_chars: int) -> str | None:
    if not isinstance(text, str):
        return None
    t = _REDACT.sub("[redacted]", text)
    t = _WS.sub(" ", t).strip()
    # If the complaint is almost entirely redaction, it carries no routing signal.
    if t.count("[redacted]") > 0 and len(t.replace("[redacted]", "").strip()) < 40:
        return None
    return t[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────
# Main curation
# ─────────────────────────────────────────────────────────────────────────────
def build_dataframe(cfg: TriageConfig = CONFIG) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return (curated_df, hierarchy). Raises if CFPB subset is missing."""
    if not cfg.cfpb_parquet.exists():
        raise FileNotFoundError(
            f"{cfg.cfpb_parquet} not found. Run:  uv run python download_cfpb.py"
        )

    # ---- CFPB ----------------------------------------------------------------
    cfpb = pd.read_parquet(cfg.cfpb_parquet)
    cfpb = cfpb.rename(columns={"product_l1": "l1", "issue": "l2",
                                "complaint_what_happened": "text"})
    cfpb["text"] = cfpb["text"].map(lambda t: clean_narrative(t, cfg.max_chars))
    cfpb = cfpb.dropna(subset=["text", "l1", "l2"])
    cfpb = cfpb[cfpb["text"].str.len() >= cfg.min_chars]

    # anti-sparsity: drop (l1,l2) issue classes with too few samples
    counts = cfpb.groupby(["l1", "l2"]).size()
    keep = counts[counts >= cfg.min_l2_count].index
    cfpb = cfpb.set_index(["l1", "l2"]).loc[keep].reset_index()

    # balance: cap each issue. Shuffle within group, then take the first N
    # (groupby.head). Avoids groupby.apply, which drops grouping cols in pandas 3.
    cfpb = (cfpb.sample(frac=1, random_state=cfg.seed)
                .groupby(["l1", "l2"], group_keys=False).head(cfg.max_per_l2)
                .reset_index(drop=True))
    cfpb["source"] = "cfpb"
    cfpb = cfpb[["text", "l1", "l2", "source"]]

    frames = [cfpb]

    # ---- Banking77 (L1 augmentation) -----------------------------------------
    if cfg.use_banking77:
        from datasets import load_dataset
        b77 = load_dataset("PolyAI/banking77")
        names = b77["train"].features["label"].names
        b = pd.DataFrame({"text": b77["train"]["text"],
                          "intent": [names[i] for i in b77["train"]["label"]]})
        b["l1"] = b["intent"].map(banking77_to_l1)
        b["l2"] = None                      # no CFPB-style issue label
        b["source"] = "banking77"
        if cfg.banking77_cap and len(b) > cfg.banking77_cap:
            b = b.sample(cfg.banking77_cap, random_state=cfg.seed)
        frames.append(b[["text", "l1", "l2", "source"]])

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # ---- hierarchy -----------------------------------------------------------
    hierarchy: dict[str, list[str]] = {}
    for l1, sub in (df[df["source"] == "cfpb"].groupby("l1")["l2"]):
        hierarchy[l1] = sorted(sub.dropna().unique().tolist())
    # ensure B77-only L1s exist in the hierarchy even if no CFPB L2 (rare)
    for l1 in sorted(df["l1"].unique()):
        hierarchy.setdefault(l1, [])

    return df, hierarchy


# ─────────────────────────────────────────────────────────────────────────────
# Splits
# ─────────────────────────────────────────────────────────────────────────────
def make_splits(df: pd.DataFrame, cfg: TriageConfig = CONFIG) -> dict[str, pd.DataFrame]:
    """Stratified (by L1) train/val/test/pool split. train is capped."""
    rng = np.random.default_rng(cfg.seed)
    parts = {"train": [], "val": [], "test": [], "pool": []}
    for l1, g in df.groupby("l1"):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(n * cfg.test_frac)
        n_val = int(n * cfg.val_frac)
        n_pool = int(n * cfg.pool_frac)
        parts["test"].append(idx[:n_test])
        parts["val"].append(idx[n_test:n_test + n_val])
        parts["pool"].append(idx[n_test + n_val:n_test + n_val + n_pool])
        parts["train"].append(idx[n_test + n_val + n_pool:])
    out = {}
    for k, chunks in parts.items():
        sel = np.concatenate(chunks)
        rng.shuffle(sel)
        out[k] = df.loc[sel].reset_index(drop=True)
    # cap train for tractable local training
    if len(out["train"]) > cfg.train_cap:
        out["train"] = out["train"].sample(cfg.train_cap, random_state=cfg.seed).reset_index(drop=True)
    return out


def label_maps(hierarchy: dict[str, list[str]]):
    """Stable id maps for L1 and for the flat L2 space."""
    l1_labels = sorted(hierarchy.keys())
    l2_labels = sorted({l2 for subs in hierarchy.values() for l2 in subs})
    l1_to_id = {l: i for i, l in enumerate(l1_labels)}
    l2_to_id = {l: i for i, l in enumerate(l2_labels)}
    return l1_labels, l2_labels, l1_to_id, l2_to_id


if __name__ == "__main__":
    df, hier = build_dataframe()
    print(f"Curated rows: {len(df)}  (cfpb={sum(df.source=='cfpb')}, "
          f"banking77={sum(df.source=='banking77')})")
    print(f"L1 classes: {len(hier)} | total L2 issues: "
          f"{sum(len(v) for v in hier.values())}")
    print("\nHierarchy (L1 → #L2):")
    for l1 in sorted(hier):
        print(f"  {l1:38s} {len(hier[l1]):2d} issues")
    splits = make_splits(df)
    print("\nSplits:", {k: len(v) for k, v in splits.items()})
