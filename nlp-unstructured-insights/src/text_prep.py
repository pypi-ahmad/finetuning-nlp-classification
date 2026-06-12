from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9']+")
ALLOWED_CHARS_RE = re.compile(r"[^a-z0-9\s\.,!?'\-]")

ACTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "quality_issue": (
        "stale",
        "spoiled",
        "mold",
        "expired",
        "broken",
        "defective",
        "leaking",
        "contaminated",
        "rancid",
        "bad taste",
        "awful taste",
    ),
    "shipping_service_issue": (
        "late delivery",
        "never arrived",
        "damaged package",
        "shipping",
        "delivery",
        "courier",
        "customer service",
        "support",
        "refund",
        "return",
    ),
    "pricing_value_issue": (
        "too expensive",
        "overpriced",
        "price",
        "cost",
        "value",
        "discount",
        "coupon",
    ),
    "usability_packaging_issue": (
        "hard to open",
        "packaging",
        "label",
        "instructions",
        "messy",
        "spills",
        "seal",
        "lid",
    ),
    "taste_preference": (
        "too sweet",
        "too salty",
        "bland",
        "bitter",
        "texture",
        "flavor",
    ),
}

POSITIVE_CUES = (
    "love",
    "great",
    "excellent",
    "perfect",
    "favorite",
    "recommend",
    "best",
)


@dataclass(slots=True)
class PreprocessingConfig:
    text_col: str = "Text"
    score_col: str = "Score"
    drop_empty: bool = True
    sample_size: int | None = 30000
    random_state: int = 42


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""
    value = html.unescape(str(text))
    value = unicodedata.normalize("NFKC", value).lower().strip()
    value = URL_RE.sub(" ", value)
    value = EMAIL_RE.sub(" ", value)
    value = HTML_TAG_RE.sub(" ", value)
    value = value.replace("\n", " ").replace("\r", " ")
    value = ALLOWED_CHARS_RE.sub(" ", value)
    value = MULTISPACE_RE.sub(" ", value)
    return value.strip()


def strip_stopwords(text: str) -> str:
    tokens = [tok for tok in TOKEN_RE.findall(text) if tok not in ENGLISH_STOP_WORDS and len(tok) > 2]
    return " ".join(tokens)


def infer_actionable_category(text: str, rating: float | int | None = None) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return "other_actionable"

    scores: dict[str, int] = {}
    for category, keywords in ACTION_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in normalized:
                score += 1
        scores[category] = score

    best_category = max(scores, key=scores.get)
    if scores[best_category] > 0:
        return best_category

    has_positive_cue = any(cue in normalized for cue in POSITIVE_CUES)
    if rating is not None:
        if float(rating) >= 4 and has_positive_cue:
            return "praise_loyalty"
        if float(rating) <= 2:
            return "general_negative_experience"
    if has_positive_cue:
        return "praise_loyalty"
    return "other_actionable"


def severity_bucket(rating: float | int | None, category: str) -> str:
    if rating is None or pd.isna(rating):
        return "unknown"
    numeric = float(rating)
    if numeric <= 2:
        return "high"
    if numeric <= 3:
        return "medium"
    if category in {"shipping_service_issue", "quality_issue"}:
        return "medium"
    return "low"


def prepare_reviews_frame(df: pd.DataFrame, config: PreprocessingConfig | None = None) -> pd.DataFrame:
    cfg = config or PreprocessingConfig()
    missing = [col for col in (cfg.text_col, cfg.score_col) if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    work = work.drop_duplicates(subset=[cfg.text_col]).reset_index(drop=True)
    work["raw_text"] = work[cfg.text_col].astype(str)
    work["clean_text"] = work["raw_text"].map(normalize_text)
    work["clean_text_nostop"] = work["clean_text"].map(strip_stopwords)
    work["text_len_chars"] = work["clean_text"].str.len()
    work["text_len_tokens"] = work["clean_text"].str.count(r"\S+")
    work["action_category"] = [
        infer_actionable_category(text=t, rating=r) for t, r in zip(work["clean_text"], work[cfg.score_col], strict=False)
    ]
    work["severity"] = [
        severity_bucket(rating=r, category=c) for r, c in zip(work[cfg.score_col], work["action_category"], strict=False)
    ]
    work["needs_action"] = (~work["action_category"].isin({"praise_loyalty"})).astype(int)

    if cfg.drop_empty:
        work = work[work["clean_text"].str.len() > 0]

    if cfg.sample_size is not None and len(work) > cfg.sample_size:
        work = work.sample(cfg.sample_size, random_state=cfg.random_state)

    return work.reset_index(drop=True)


def top_terms_by_category(df: pd.DataFrame, text_col: str = "clean_text_nostop", category_col: str = "action_category", top_k: int = 12) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for category, block in df.groupby(category_col):
        token_counts: dict[str, int] = {}
        for text in block[text_col].fillna(""):
            for token in TOKEN_RE.findall(text):
                if token in ENGLISH_STOP_WORDS or len(token) < 3:
                    continue
                token_counts[token] = token_counts.get(token, 0) + 1
        top_terms = sorted(token_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        records.append(
            {
                "action_category": category,
                "n_examples": len(block),
                "top_terms": ", ".join(term for term, _ in top_terms),
            }
        )
    return pd.DataFrame(records).sort_values("n_examples", ascending=False)


def build_relevance_mask(categories: Iterable[str], expected_category: str) -> list[int]:
    return [1 if cat == expected_category else 0 for cat in categories]

