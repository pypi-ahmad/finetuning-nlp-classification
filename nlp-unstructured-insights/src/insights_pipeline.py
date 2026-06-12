from __future__ import annotations

import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.metrics import f1_score, fbeta_score, recall_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    import faiss
except Exception:  # pragma: no cover - optional dependency
    faiss = None

try:
    from lazypredict.Supervised import LazyClassifier
except Exception:  # pragma: no cover - optional dependency
    LazyClassifier = None

try:
    from flaml import AutoML
except Exception:  # pragma: no cover - optional dependency
    AutoML = None

LEADERBOARD_COLUMNS = [
    "project_name",
    "task_type",
    "library_source",
    "model_name",
    "cv_metric_mean",
    "cv_metric_std",
    "holdout_primary_metric",
    "holdout_secondary_metric",
    "holdout_tertiary_metric",
    "calibration_metric",
    "train_time_sec",
    "infer_latency_ms",
    "model_size_mb",
    "interpretability_note",
    "rank_score",
    "final_rank",
]

LAZY_TO_MANUAL_FAMILY = {
    "logisticregression": "logistic_regression",
    "ridgeclassifier": "ridge_classifier",
    "ridgeclassifiercv": "ridge_classifier",
    "linearsvc": "linear_svc",
    "passiveaggressiveclassifier": "passive_aggressive",
    "sgdclassifier": "sgd_classifier",
    "randomforestclassifier": "random_forest",
    "extratreesclassifier": "extra_trees",
    "decisiontreeclassifier": "decision_tree",
    "multinomialnb": "multinomial_nb",
    "bernoullinb": "multinomial_nb",
}

MANUAL_FAMILY_DISPLAY = {
    "logistic_regression": "Manual::LogisticRegression",
    "ridge_classifier": "Manual::RidgeClassifier",
    "linear_svc": "Manual::LinearSVC",
    "passive_aggressive": "Manual::PassiveAggressiveClassifier",
    "sgd_classifier": "Manual::SGDClassifier",
    "random_forest": "Manual::RandomForestClassifier",
    "extra_trees": "Manual::ExtraTreesClassifier",
    "decision_tree": "Manual::DecisionTreeClassifier",
    "multinomial_nb": "Manual::MultinomialNB",
}


@dataclass(slots=True)
class RetrievalResult:
    query: str
    rank: int
    corpus_id: int
    score: float
    text: str
    action_category: str


def embed_texts(
    texts: Sequence[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> tuple[np.ndarray, SentenceTransformer]:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return np.asarray(embeddings, dtype=np.float32), model


def build_faiss_index(embeddings: np.ndarray):
    if faiss is None:
        raise ImportError("faiss-cpu is not available. Install it with `uv add faiss-cpu`.")
    matrix = np.ascontiguousarray(embeddings.astype(np.float32))
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def semantic_search(
    query: str,
    corpus_texts: Sequence[str],
    corpus_categories: Sequence[str],
    model: SentenceTransformer,
    index,
    top_k: int = 10,
) -> pd.DataFrame:
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = index.search(np.ascontiguousarray(query_embedding.astype(np.float32)), top_k)
    rows: list[dict[str, object]] = []
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        rows.append(
            {
                "rank": rank,
                "corpus_id": int(idx),
                "score": float(score),
                "text": corpus_texts[idx],
                "action_category": corpus_categories[idx],
            }
        )
    return pd.DataFrame(rows)


def precision_at_k(binary_relevance: Sequence[int | float], k: int = 10) -> float:
    rel = np.asarray(list(binary_relevance)[:k], dtype=float)
    if rel.size == 0:
        return float("nan")
    return float(np.mean(rel))


def ndcg_at_k(binary_relevance: Sequence[int | float], k: int = 10) -> float:
    rel = np.asarray(list(binary_relevance)[:k], dtype=float)
    if rel.size == 0:
        return float("nan")
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    gains = (2**rel - 1) * discounts
    dcg = float(np.sum(gains))
    ideal_rel = np.sort(rel)[::-1]
    ideal_dcg = float(np.sum((2**ideal_rel - 1) * discounts))
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def cluster_coherence(embeddings: np.ndarray, labels: Sequence[int]) -> float:
    labels_arr = np.asarray(labels)
    if embeddings.shape[0] != labels_arr.shape[0]:
        raise ValueError("Embeddings and labels must have the same number of rows.")

    mask = labels_arr != -1
    unique_labels = np.unique(labels_arr[mask])
    if mask.sum() < 10 or unique_labels.size < 2:
        return float("nan")
    return float(silhouette_score(embeddings[mask], labels_arr[mask], metric="cosine"))


def run_bertopic(
    texts: Sequence[str],
    embeddings: np.ndarray | None = None,
    min_topic_size: int = 30,
    random_state: int = 42,
):
    from bertopic import BERTopic

    topic_model = BERTopic(
        min_topic_size=min_topic_size,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=False,
    )
    topics, probabilities = topic_model.fit_transform(list(texts), embeddings=embeddings)
    info = topic_model.get_topic_info()
    return topic_model, np.asarray(topics), probabilities, info


def summarize_topics(
    topic_model,
    topic_info: pd.DataFrame,
    texts: Sequence[str],
    topics: Sequence[int],
    top_n_topics: int = 12,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    topics_arr = np.asarray(topics)
    for _, row in topic_info.head(top_n_topics + 1).iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue
        topic_words = topic_model.get_topic(topic_id) or []
        top_terms = ", ".join(word for word, _ in topic_words[:8])
        representative = topic_model.get_representative_docs(topic_id) or []
        representative_text = representative[0] if representative else ""
        summary = f"Theme around {top_terms}" if top_terms else "General feedback cluster"
        records.append(
            {
                "topic_id": topic_id,
                "size": int(row["Count"]),
                "top_terms": top_terms,
                "representative_text": representative_text[:300],
                "summary": summary,
                "avg_text_len": float(np.mean([len(texts[i]) for i in np.where(topics_arr == topic_id)[0]])),
            }
        )
    return pd.DataFrame(records).sort_values("size", ascending=False).reset_index(drop=True)


def _to_dense_if_needed(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _predict_proba_if_available(model: Any, x_test: Any) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(x_test)
        except Exception:
            return None
    return None


def _confidence_quality(
    y_true: Sequence[Any],
    y_prob: np.ndarray | None,
    class_order: Sequence[Any] | None = None,
) -> float:
    if y_prob is None:
        return float("nan")
    if y_prob.ndim != 2 or y_prob.size == 0:
        return float("nan")

    classes = list(class_order) if class_order is not None else sorted(pd.unique(pd.Series(y_true)))
    class_to_idx = {label: i for i, label in enumerate(classes)}
    y_indices = np.array([class_to_idx.get(label, -1) for label in y_true], dtype=int)
    valid_mask = y_indices >= 0
    if not np.any(valid_mask):
        return float("nan")

    y_indices = y_indices[valid_mask]
    probs = y_prob[valid_mask]
    one_hot = np.eye(probs.shape[1])[y_indices]
    brier = np.mean(np.sum((one_hot - probs) ** 2, axis=1))
    return float(max(0.0, 1.0 - brier))


def _critical_issue_recall(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    high_priority_labels: Sequence[str] = ("quality_issue", "shipping_service_issue"),
) -> float:
    mask = np.array([label in set(high_priority_labels) for label in y_true], dtype=bool)
    if mask.sum() == 0:
        return float("nan")
    y_true_binary = np.array([1 if label in set(high_priority_labels) else 0 for label in y_true], dtype=int)
    y_pred_binary = np.array([1 if label in set(high_priority_labels) else 0 for label in y_pred], dtype=int)
    return float(recall_score(y_true_binary[mask], y_pred_binary[mask], zero_division=0))


def _measure_infer_latency_ms(predict_fn: Callable[[Any], Any], x: Any, n_calls: int = 4) -> float:
    n_rows = x.shape[0] if hasattr(x, "shape") else len(x)
    sample = x[: min(n_rows, 256)]
    sample_rows = sample.shape[0] if hasattr(sample, "shape") else len(sample)
    if sample_rows == 0:
        return float("nan")
    start = time.perf_counter()
    for _ in range(n_calls):
        _ = predict_fn(sample)
    elapsed = time.perf_counter() - start
    return float((elapsed / n_calls) * 1000.0)


def _estimate_model_size_mb(model: Any) -> float:
    try:
        data = pickle.dumps(model)
        return float(len(data) / (1024**2))
    except Exception:
        return float("nan")


def classification_metrics_bundle(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_prob: np.ndarray | None = None,
    class_order: Sequence[Any] | None = None,
    high_priority_labels: Sequence[str] = ("quality_issue", "shipping_service_issue"),
) -> dict[str, float]:
    return {
        "holdout_primary_metric": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "holdout_secondary_metric": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "holdout_tertiary_metric": _critical_issue_recall(y_true, y_pred, high_priority_labels=high_priority_labels),
        "calibration_metric": _confidence_quality(y_true, y_prob, class_order=class_order),
    }


def make_leaderboard_row(
    *,
    project_name: str,
    task_type: str,
    library_source: str,
    model_name: str,
    cv_metric_mean: float | int | None,
    cv_metric_std: float | int | None,
    holdout_primary_metric: float | int | None,
    holdout_secondary_metric: float | int | None,
    holdout_tertiary_metric: float | int | None,
    calibration_metric: float | int | None,
    train_time_sec: float | int | None,
    infer_latency_ms: float | int | None,
    model_size_mb: float | int | None,
    interpretability_note: str,
) -> dict[str, Any]:
    return {
        "project_name": project_name,
        "task_type": task_type,
        "library_source": library_source,
        "model_name": model_name,
        "cv_metric_mean": cv_metric_mean,
        "cv_metric_std": cv_metric_std,
        "holdout_primary_metric": holdout_primary_metric,
        "holdout_secondary_metric": holdout_secondary_metric,
        "holdout_tertiary_metric": holdout_tertiary_metric,
        "calibration_metric": calibration_metric,
        "train_time_sec": train_time_sec,
        "infer_latency_ms": infer_latency_ms,
        "model_size_mb": model_size_mb,
        "interpretability_note": interpretability_note,
    }


def run_lazypredict_discovery(
    x_train: Any,
    y_train: Sequence[Any],
    x_valid: Any,
    y_valid: Sequence[Any],
    project_name: str,
    max_models: int = 24,
    timeout_sec: int = 420,
    top_n_rows: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if LazyClassifier is None:
        return pd.DataFrame(), pd.DataFrame()

    x_train_dense = _to_dense_if_needed(x_train)
    x_valid_dense = _to_dense_if_needed(x_valid)

    model = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        predictions=False,
        random_state=42,
        n_jobs=-1,
        max_models=max_models,
        timeout=timeout_sec,
    )
    models_df, _ = model.fit(x_train_dense, x_valid_dense, y_train, y_valid)
    models_df = models_df.copy()
    if "lazy_model_name" not in models_df.columns:
        if "Model" in models_df.columns:
            models_df = models_df.rename(columns={"Model": "lazy_model_name"})
        else:
            models_df = models_df.reset_index()
            first_col = models_df.columns[0]
            models_df = models_df.rename(columns={first_col: "lazy_model_name"})

    f1_col = "F1 Score" if "F1 Score" in models_df.columns else "F1"
    secondary_col = "Balanced Accuracy" if "Balanced Accuracy" in models_df.columns else "Accuracy"

    rows: list[dict[str, Any]] = []
    for _, row in models_df.head(top_n_rows).iterrows():
        rows.append(
            make_leaderboard_row(
                project_name=project_name,
                task_type="classification",
                library_source="lazypredict",
                model_name=str(row["lazy_model_name"]),
                cv_metric_mean=np.nan,
                cv_metric_std=np.nan,
                holdout_primary_metric=float(row.get(f1_col, np.nan)),
                holdout_secondary_metric=float(row.get(secondary_col, np.nan)),
                holdout_tertiary_metric=np.nan,
                calibration_metric=np.nan,
                train_time_sec=float(row.get("Time Taken", np.nan)),
                infer_latency_ms=np.nan,
                model_size_mb=np.nan,
                interpretability_note="Discovery-only benchmark across diverse model families.",
            )
        )
    return models_df, pd.DataFrame(rows)


def map_lazy_model_to_family(model_name: str) -> str | None:
    key = "".join(ch for ch in str(model_name).lower() if ch.isalnum())
    for alias, family in LAZY_TO_MANUAL_FAMILY.items():
        if alias in key:
            return family
    return None


def select_top_eligible_from_lazypredict(
    lazy_models_df: pd.DataFrame,
    top_n: int = 3,
    exclude_patterns: Sequence[str] = (
        "dummy",
        "gaussian",
        "knn",
        "nearestcentroid",
        "labelpropagation",
        "labelspreading",
    ),
) -> pd.DataFrame:
    if lazy_models_df.empty:
        return pd.DataFrame(columns=["lazy_model_name", "family_name", "selection_reason"])  # pragma: no cover

    table = lazy_models_df.copy()
    f1_col = "F1 Score" if "F1 Score" in table.columns else "F1"
    table = table.rename(columns={"index": "lazy_model_name"})
    if "lazy_model_name" not in table.columns:
        table = table.reset_index().rename(columns={"index": "lazy_model_name"})

    table["family_name"] = table["lazy_model_name"].map(map_lazy_model_to_family)
    table["exclude_match"] = table["lazy_model_name"].astype(str).str.lower().apply(
        lambda name: any(pattern in name for pattern in exclude_patterns)
    )

    table = table[(table["family_name"].notna()) & (~table["exclude_match"])].copy()
    table = table.sort_values(f1_col, ascending=False)
    table = table.drop_duplicates(subset=["family_name"], keep="first")
    table = table.head(top_n).copy()
    table["selection_reason"] = table.apply(
        lambda r: f"Top LazyPredict model for family {r['family_name']} with {f1_col}={r[f1_col]:.4f}", axis=1
    )

    return table[["lazy_model_name", "family_name", "selection_reason", f1_col]]


def build_manual_estimator(family_name: str, random_state: int = 42) -> tuple[Any, str]:
    if family_name == "logistic_regression":
        return (
            LogisticRegression(max_iter=2500, class_weight="balanced", n_jobs=-1, random_state=random_state),
            "Interpretable linear coefficients; stable baseline for sparse text.",
        )
    if family_name == "ridge_classifier":
        return (
            RidgeClassifier(class_weight="balanced", random_state=random_state),
            "Low-variance linear model useful under correlated n-gram features.",
        )
    if family_name == "linear_svc":
        return (
            LinearSVC(class_weight="balanced", random_state=random_state),
            "Margin-based linear classifier; fast and robust on high-dimensional text.",
        )
    if family_name == "passive_aggressive":
        return (
            PassiveAggressiveClassifier(class_weight="balanced", random_state=random_state, max_iter=1500),
            "Online large-margin model suitable for drift-prone text streams.",
        )
    if family_name == "sgd_classifier":
        return (
            SGDClassifier(loss="log_loss", class_weight="balanced", random_state=random_state),
            "Scalable linear learner with probabilistic outputs.",
        )
    if family_name == "random_forest":
        return (
            RandomForestClassifier(
                n_estimators=450,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
            "Non-linear ensemble; captures feature interactions at cost of interpretability.",
        )
    if family_name == "extra_trees":
        return (
            ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
            "Highly randomized ensemble; often strong on noisy sparse inputs.",
        )
    if family_name == "decision_tree":
        return (
            DecisionTreeClassifier(
                max_depth=20,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=random_state,
            ),
            "Single-tree baseline for transparent decision path inspection.",
        )
    if family_name == "multinomial_nb":
        return (
            MultinomialNB(alpha=0.4),
            "Fast lexical baseline; strong when class cues are token-frequency driven.",
        )

    raise ValueError(f"Unknown manual family: {family_name}")


def evaluate_manual_family(
    family_name: str,
    x_train: Any,
    y_train: Sequence[Any],
    x_valid: Any,
    y_valid: Sequence[Any],
    project_name: str,
    cv: int = 4,
    random_state: int = 42,
    calibrate_when_needed: bool = True,
    high_priority_labels: Sequence[str] = ("quality_issue", "shipping_service_issue"),
) -> tuple[dict[str, Any], dict[str, Any]]:
    estimator, note = build_manual_estimator(family_name, random_state=random_state)

    cv_mean = np.nan
    cv_std = np.nan
    try:
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(
            clone(estimator),
            x_train,
            y_train,
            cv=splitter,
            scoring="f1_macro",
            n_jobs=-1,
            error_score=np.nan,
        )
        cv_mean = float(np.nanmean(cv_scores))
        cv_std = float(np.nanstd(cv_scores))
    except Exception as exc:
        warnings.warn(f"CV failed for {family_name}: {exc}")

    fit_start = time.perf_counter()
    fitted = clone(estimator).fit(x_train, y_train)
    train_time = time.perf_counter() - fit_start

    used_model = fitted
    y_pred = fitted.predict(x_valid)
    y_prob = _predict_proba_if_available(fitted, x_valid)
    calibration_note = ""

    if y_prob is None and calibrate_when_needed:
        try:
            calibrated = CalibratedClassifierCV(estimator=clone(estimator), method="sigmoid", cv=3)
            calibrated.fit(x_train, y_train)
            y_pred_cal = calibrated.predict(x_valid)
            y_prob_cal = calibrated.predict_proba(x_valid)
            if f1_score(y_valid, y_pred_cal, average="macro", zero_division=0) >= f1_score(
                y_valid, y_pred, average="macro", zero_division=0
            ):
                used_model = calibrated
                y_pred = y_pred_cal
                y_prob = y_prob_cal
                calibration_note = " Calibrated using sigmoid CV."
        except Exception as exc:
            calibration_note = f" Calibration skipped ({type(exc).__name__})."

    metrics = classification_metrics_bundle(
        y_true=y_valid,
        y_pred=y_pred,
        y_prob=y_prob,
        class_order=getattr(used_model, "classes_", None),
        high_priority_labels=high_priority_labels,
    )

    row = make_leaderboard_row(
        project_name=project_name,
        task_type="classification",
        library_source="manual_engineering",
        model_name=MANUAL_FAMILY_DISPLAY.get(family_name, f"Manual::{family_name}"),
        cv_metric_mean=cv_mean,
        cv_metric_std=cv_std,
        holdout_primary_metric=metrics["holdout_primary_metric"],
        holdout_secondary_metric=metrics["holdout_secondary_metric"],
        holdout_tertiary_metric=metrics["holdout_tertiary_metric"],
        calibration_metric=metrics["calibration_metric"],
        train_time_sec=float(train_time),
        infer_latency_ms=_measure_infer_latency_ms(used_model.predict, x_valid),
        model_size_mb=_estimate_model_size_mb(used_model),
        interpretability_note=note + calibration_note,
    )

    artifacts = {
        "family_name": family_name,
        "model": used_model,
        "raw_model": fitted,
        "y_pred": np.asarray(y_pred),
        "y_prob": y_prob,
        "classes_": getattr(used_model, "classes_", None),
        "calibrated": bool(calibration_note),
    }
    return row, artifacts


def optimize_priority_threshold(
    y_true: Sequence[Any],
    y_prob: np.ndarray,
    class_order: Sequence[Any],
    priority_labels: Sequence[str] = ("quality_issue", "shipping_service_issue"),
    beta: float = 2.0,
    grid: Sequence[float] | None = None,
) -> dict[str, float]:
    if y_prob is None or len(y_prob) == 0:
        return {"best_threshold": np.nan, "best_fbeta": np.nan, "priority_recall": np.nan, "priority_precision": np.nan}

    classes = list(class_order)
    class_to_idx = {label: i for i, label in enumerate(classes)}
    priority_idx = [class_to_idx[label] for label in priority_labels if label in class_to_idx]
    if not priority_idx:
        return {"best_threshold": np.nan, "best_fbeta": np.nan, "priority_recall": np.nan, "priority_precision": np.nan}

    y_true_binary = np.array([1 if label in set(priority_labels) else 0 for label in y_true], dtype=int)
    priority_prob = np.clip(y_prob[:, priority_idx].sum(axis=1), 0.0, 1.0)

    thresholds = list(grid) if grid is not None else list(np.linspace(0.10, 0.90, 33))
    best = {"best_threshold": 0.5, "best_fbeta": -1.0, "priority_recall": 0.0, "priority_precision": 0.0}

    for threshold in thresholds:
        pred_binary = (priority_prob >= threshold).astype(int)
        tp = int(((pred_binary == 1) & (y_true_binary == 1)).sum())
        fp = int(((pred_binary == 1) & (y_true_binary == 0)).sum())
        fn = int(((pred_binary == 0) & (y_true_binary == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fbeta = fbeta_score(y_true_binary, pred_binary, beta=beta, zero_division=0)

        if fbeta > best["best_fbeta"]:
            best = {
                "best_threshold": float(threshold),
                "best_fbeta": float(fbeta),
                "priority_recall": float(recall),
                "priority_precision": float(precision),
            }
    return best


def run_flaml_optimization(
    x_train: Any,
    y_train: Sequence[Any],
    x_valid: Any,
    y_valid: Sequence[Any],
    project_name: str,
    time_budget: int = 180,
    random_state: int = 42,
    estimator_list: Sequence[str] = ("lgbm", "xgboost", "rf", "extra_tree", "xgb_limitdepth"),
    high_priority_labels: Sequence[str] = ("quality_issue", "shipping_service_issue"),
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    if AutoML is None:
        return (
            make_leaderboard_row(
                project_name=project_name,
                task_type="classification",
                library_source="flaml",
                model_name="FLAML::not_available",
                cv_metric_mean=np.nan,
                cv_metric_std=np.nan,
                holdout_primary_metric=np.nan,
                holdout_secondary_metric=np.nan,
                holdout_tertiary_metric=np.nan,
                calibration_metric=np.nan,
                train_time_sec=np.nan,
                infer_latency_ms=np.nan,
                model_size_mb=np.nan,
                interpretability_note="FLAML package unavailable.",
            ),
            {"error": "AutoML unavailable"},
            None,
        )

    y_train_series = pd.Series(y_train).astype(str)
    y_valid_series = pd.Series(y_valid).astype(str)
    classes = sorted(pd.unique(pd.concat([y_train_series, y_valid_series], ignore_index=True)))
    label_to_id = {label: idx for idx, label in enumerate(classes)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    y_train_encoded = y_train_series.map(label_to_id).astype(int).to_numpy()

    attempt_estimators = [tuple(estimator_list), ("rf", "extra_tree", "lrl1")]
    errors: list[str] = []
    automl = None
    train_time = np.nan
    used_estimators: tuple[str, ...] = tuple(estimator_list)

    for attempt_list in attempt_estimators:
        used_estimators = tuple(attempt_list)
        try:
            automl = AutoML()
            fit_start = time.perf_counter()
            automl.fit(
                X_train=x_train,
                y_train=y_train_encoded,
                task="classification",
                metric="macro_f1",
                estimator_list=list(attempt_list),
                time_budget=time_budget,
                eval_method="cv",
                n_splits=3,
                n_jobs=-1,
                seed=random_state,
                log_training_metric=True,
                verbose=0,
            )
            train_time = time.perf_counter() - fit_start
            break
        except Exception as exc:  # pragma: no cover - runtime variability
            errors.append(f"{type(exc).__name__}: {exc}")
            automl = None

    if automl is None:
        row = make_leaderboard_row(
            project_name=project_name,
            task_type="classification",
            library_source="flaml",
            model_name="FLAML::failed",
            cv_metric_mean=np.nan,
            cv_metric_std=np.nan,
            holdout_primary_metric=np.nan,
            holdout_secondary_metric=np.nan,
            holdout_tertiary_metric=np.nan,
            calibration_metric=np.nan,
            train_time_sec=np.nan,
            infer_latency_ms=np.nan,
            model_size_mb=np.nan,
            interpretability_note=f"FLAML failed after retries. Errors: {' | '.join(errors[:2])}",
        )
        details = {
            "best_estimator": None,
            "best_config": {},
            "best_loss": np.nan,
            "best_iteration": np.nan,
            "best_config_train_time": np.nan,
            "search_estimators": list(estimator_list),
            "fallback_estimators": ["rf", "extra_tree", "lrl1"],
            "fit_errors": errors,
        }
        return row, details, None

    y_pred_encoded = np.asarray(automl.predict(x_valid))
    y_pred = pd.Series(y_pred_encoded).map(lambda x: id_to_label.get(int(x), str(x))).astype(str).to_numpy()

    y_prob = _predict_proba_if_available(automl, x_valid)
    flaml_classes = getattr(automl, "classes_", None)
    class_order: Sequence[str] | None = None
    if y_prob is not None and flaml_classes is not None:
        try:
            class_order = [id_to_label.get(int(c), str(c)) for c in flaml_classes]
        except Exception:
            class_order = None

    metrics = classification_metrics_bundle(
        y_true=y_valid_series,
        y_pred=y_pred,
        y_prob=y_prob,
        class_order=class_order,
        high_priority_labels=high_priority_labels,
    )

    best_result = getattr(automl, "best_result", {}) or {}
    cv_metric = best_result.get("val_macro_f1")
    if cv_metric is None:
        best_loss = getattr(automl, "best_loss", None)
        cv_metric = (1.0 - best_loss) if isinstance(best_loss, (int, float)) else np.nan

    best_model = getattr(getattr(automl, "model", None), "estimator", automl)
    config = getattr(automl, "best_config", {})
    top_hps = ", ".join(f"{k}={v}" for k, v in list(config.items())[:6])

    retry_note = "" if used_estimators == tuple(estimator_list) else " Used fallback estimator set due initial fit failure."
    row = make_leaderboard_row(
        project_name=project_name,
        task_type="classification",
        library_source="flaml",
        model_name=f"FLAML::{getattr(automl, 'best_estimator', 'unknown')}",
        cv_metric_mean=float(cv_metric) if pd.notna(cv_metric) else np.nan,
        cv_metric_std=np.nan,
        holdout_primary_metric=metrics["holdout_primary_metric"],
        holdout_secondary_metric=metrics["holdout_secondary_metric"],
        holdout_tertiary_metric=metrics["holdout_tertiary_metric"],
        calibration_metric=metrics["calibration_metric"],
        train_time_sec=float(train_time),
        infer_latency_ms=_measure_infer_latency_ms(automl.predict, x_valid),
        model_size_mb=_estimate_model_size_mb(best_model),
        interpretability_note=f"Searched {', '.join(used_estimators)}. Best config: {top_hps}.{retry_note}",
    )

    details = {
        "best_estimator": getattr(automl, "best_estimator", None),
        "best_config": config,
        "best_loss": getattr(automl, "best_loss", np.nan),
        "best_iteration": getattr(automl, "best_iteration", np.nan),
        "best_config_train_time": getattr(automl, "best_config_train_time", np.nan),
        "search_estimators": list(used_estimators),
        "requested_estimators": list(estimator_list),
        "best_result": best_result,
        "fit_errors": errors,
    }
    return row, details, automl


def run_pycaret_experiment(
    x_train_dense: np.ndarray,
    y_train: Sequence[Any],
    x_valid_dense: np.ndarray,
    y_valid: Sequence[Any],
    feature_names: Sequence[str],
    project_name: str,
    artifacts_dir: str | Path,
    session_id: int = 42,
    fold: int = 3,
    compare_top_n: int = 3,
    tune_iter: int = 20,
    include_models: Sequence[str] = ("lr", "ridge", "rf", "et", "nb", "lda"),
    high_priority_labels: Sequence[str] = ("quality_issue", "shipping_service_issue"),
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(x_train_dense, columns=feature_names)
    valid_df = pd.DataFrame(x_valid_dense, columns=feature_names)
    train_df["target_label"] = list(y_train)
    valid_df["target_label"] = list(y_valid)
    train_x = train_df.drop(columns=["target_label"])
    valid_x = valid_df.drop(columns=["target_label"])

    try:
        from pycaret.classification import ClassificationExperiment

        exp = ClassificationExperiment(
            session_id=session_id,
            fold=fold,
            preprocess=True,
            remove_outliers=False,
            n_jobs=-1,
            verbose=False,
        )
        exp.fit(X=train_x, y=train_df["target_label"])

        compare_result = exp.compare_models(
            include=list(include_models),
            n_select=compare_top_n,
            sort="F1",
            turbo=True,
            errors="ignore",
            verbose=False,
        )
        compare_table = (
            compare_result.leaderboard.copy()
            if hasattr(compare_result, "leaderboard")
            else exp.pull().copy()
        )
        if hasattr(compare_result, "models"):
            compare_model_list = list(compare_result.models)
            seed_model = compare_result.best
        else:
            compare_model_list = compare_result if isinstance(compare_result, list) else [compare_result]
            seed_model = compare_model_list[0]

        tune_result = exp.tune_model(
            seed_model,
            n_iter=tune_iter,
            optimize="F1",
            verbose=False,
        )
        tuned_pipeline = tune_result.pipeline if hasattr(tune_result, "pipeline") else tune_result
        tune_table = (
            tune_result.metrics.copy()
            if hasattr(tune_result, "metrics") and isinstance(tune_result.metrics, pd.DataFrame)
            else exp.pull().copy()
        )

        calibrated_model = None
        calibration_note = "Calibration skipped."
        try:
            calibrate_result = exp.calibrate_model(tuned_pipeline, method="sigmoid", verbose=False)
            calibrated_model = (
                calibrate_result.pipeline
                if hasattr(calibrate_result, "pipeline")
                else calibrate_result
            )
            candidate_model = calibrated_model
            calibration_note = "Calibrated with sigmoid."
        except Exception as exc:  # pragma: no cover - version-specific behavior
            candidate_model = tuned_pipeline
            calibration_note = f"Calibration unavailable ({type(exc).__name__})."

        finalize_result = exp.finalize_model(candidate_model)
        finalized_model = finalize_result.pipeline if hasattr(finalize_result, "pipeline") else finalize_result
        model_save_base = artifacts_path / "pycaret_final_model"
        exp.save_model(finalized_model, str(model_save_base), verbose=False)

        pred_result = exp.predict_model(finalized_model, data=valid_x, raw_score=True, verbose=False)
        pred_df = pred_result.predictions.copy() if hasattr(pred_result, "predictions") else pred_result
        pred_col = "prediction_label" if "prediction_label" in pred_df.columns else "Label"

        y_pred = pred_df[pred_col].astype(str)
        y_true_series = valid_df["target_label"].astype(str)

        raw_prob_cols = [
            col
            for col in pred_df.columns
            if col.lower().startswith("prediction_score_") or col.lower().startswith("score_")
        ]
        y_prob = pred_df[raw_prob_cols].to_numpy() if raw_prob_cols else None
        if y_prob is None:
            fallback_score_col = "prediction_score" if "prediction_score" in pred_df.columns else "Score"
            if fallback_score_col in pred_df.columns:
                one_dim = pred_df[fallback_score_col].astype(float).to_numpy()
                y_prob = None if one_dim.ndim != 1 else np.column_stack([1 - one_dim, one_dim])

        class_order_for_metrics: Sequence[str] | None = None
        y_prob_for_metrics = y_prob
        if y_prob is not None and raw_prob_cols:
            suffixes = [col.split("_", 2)[-1] if "prediction_score_" in col else col.split("_", 1)[-1] for col in raw_prob_cols]
            unique_true = sorted(pd.unique(y_true_series))
            if set(suffixes) == set(unique_true):
                class_order_for_metrics = suffixes
            else:
                y_prob_for_metrics = None
        elif y_prob is not None and y_prob.shape[1] == len(pd.unique(y_true_series)):
            class_order_for_metrics = sorted(pd.unique(y_true_series))

        metrics = classification_metrics_bundle(
            y_true=y_true_series,
            y_pred=y_pred,
            y_prob=y_prob_for_metrics,
            class_order=class_order_for_metrics,
            high_priority_labels=high_priority_labels,
        )

        cv_f1 = np.nan
        cv_time = np.nan
        if not tune_table.empty and "F1" in tune_table.columns:
            if "Mean" in tune_table.index:
                cv_f1 = float(tune_table.loc["Mean", "F1"])
            else:
                cv_f1 = float(pd.to_numeric(tune_table["F1"], errors="coerce").mean())
        if "TT (Sec)" in compare_table.columns and not compare_table.empty:
            cv_time = float(compare_table.iloc[0].get("TT (Sec)", np.nan))

        row = make_leaderboard_row(
            project_name=project_name,
            task_type="classification",
            library_source="pycaret",
            model_name=f"PyCaret::{finalized_model.__class__.__name__}",
            cv_metric_mean=cv_f1,
            cv_metric_std=np.nan,
            holdout_primary_metric=metrics["holdout_primary_metric"],
            holdout_secondary_metric=metrics["holdout_secondary_metric"],
            holdout_tertiary_metric=metrics["holdout_tertiary_metric"],
            calibration_metric=metrics["calibration_metric"],
            train_time_sec=cv_time,
            infer_latency_ms=_measure_infer_latency_ms(
                lambda x: exp.predict_model(finalized_model, data=x, verbose=False),
                valid_x,
            ),
            model_size_mb=_estimate_model_size_mb(finalized_model),
            interpretability_note=f"PyCaret compare -> tune -> finalize workflow. {calibration_note}",
        )

        details = {
            "compare_table": compare_table,
            "tune_table": tune_table,
            "final_model_class": finalized_model.__class__.__name__,
            "saved_model_path": str(model_save_base) + ".pkl",
            "include_models": list(include_models),
            "calibration_note": calibration_note,
            "compared_model_count": len(compare_model_list),
        }

        return row, details, finalized_model

    except Exception as exc:
        return (
            make_leaderboard_row(
                project_name=project_name,
                task_type="classification",
                library_source="pycaret",
                model_name="PyCaret::failed",
                cv_metric_mean=np.nan,
                cv_metric_std=np.nan,
                holdout_primary_metric=np.nan,
                holdout_secondary_metric=np.nan,
                holdout_tertiary_metric=np.nan,
                calibration_metric=np.nan,
                train_time_sec=np.nan,
                infer_latency_ms=np.nan,
                model_size_mb=np.nan,
                interpretability_note=f"PyCaret workflow failed: {type(exc).__name__}: {exc}",
            ),
            {"error": f"{type(exc).__name__}: {exc}"},
            None,
        )


def rerun_candidates_with_multiple_seeds(
    texts: Sequence[str],
    labels: Sequence[Any],
    family_names: Sequence[str],
    seeds: Sequence[int],
    project_name: str,
    max_features: int = 8000,
    ngram_range: tuple[int, int] = (1, 2),
    test_size: float = 0.2,
    high_priority_labels: Sequence[str] = ("quality_issue", "shipping_service_issue"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    text_series = pd.Series(texts)
    label_series = pd.Series(labels)

    for family in family_names:
        for seed in seeds:
            x_train_text, x_valid_text, y_train, y_valid = train_test_split(
                text_series,
                label_series,
                test_size=test_size,
                random_state=seed,
                stratify=label_series,
            )
            vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=3)
            x_train = vectorizer.fit_transform(x_train_text)
            x_valid = vectorizer.transform(x_valid_text)

            row, _ = evaluate_manual_family(
                family_name=family,
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                project_name=project_name,
                cv=3,
                random_state=seed,
                calibrate_when_needed=False,
                high_priority_labels=high_priority_labels,
            )
            row["library_source"] = "multi_seed_recheck"
            row["model_name"] = f"{row['model_name']}::seed_{seed}"
            row["interpretability_note"] = f"Seed rerun for stability check on family {family}."
            rows.append(row)

    return ensure_leaderboard_columns(pd.DataFrame(rows))


def save_inference_bundle(
    path: str | Path,
    vectorizer: Any,
    model: Any,
    class_order: Sequence[Any],
    priority_threshold: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    bundle = {
        "vectorizer": vectorizer,
        "model": model,
        "class_order": list(class_order),
        "priority_threshold": priority_threshold,
        "metadata": metadata or {},
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(bundle, f)
    return output_path


def load_inference_bundle(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def predict_with_inference_bundle(bundle: dict[str, Any], texts: Sequence[str]) -> pd.DataFrame:
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]
    class_order = list(bundle.get("class_order", []))
    threshold = bundle.get("priority_threshold")

    x = vectorizer.transform(list(texts))
    labels = model.predict(x)

    y_prob = _predict_proba_if_available(model, x)
    confidence = np.full(len(labels), np.nan)
    priority_probability = np.full(len(labels), np.nan)

    if y_prob is not None:
        confidence = y_prob.max(axis=1)
        if class_order and y_prob.shape[1] == len(class_order):
            priority_idx = [
                idx
                for idx, label in enumerate(class_order)
                if label in {"quality_issue", "shipping_service_issue"}
            ]
            if priority_idx:
                priority_probability = y_prob[:, priority_idx].sum(axis=1)

    out = pd.DataFrame(
        {
            "text": list(texts),
            "predicted_label": labels,
            "confidence": confidence,
            "priority_probability": priority_probability,
        }
    )
    if threshold is not None:
        out["trigger_priority_review"] = (out["priority_probability"] >= threshold).astype(int)
    return out


def rank_leaderboard(
    leaderboard: pd.DataFrame,
    primary_col: str = "holdout_primary_metric",
    secondary_col: str = "holdout_secondary_metric",
    tertiary_col: str = "holdout_tertiary_metric",
    calibration_col: str = "calibration_metric",
) -> pd.DataFrame:
    df = leaderboard.copy()
    for col in (primary_col, secondary_col, tertiary_col, calibration_col):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_rows = len(df)
    if n_rows == 0:
        return ensure_leaderboard_columns(df)

    def scaled_rank(series: pd.Series) -> pd.Series:
        ranked = series.rank(method="average", ascending=False, na_option="bottom")
        if n_rows == 1:
            return pd.Series([1.0], index=series.index)
        return (n_rows - ranked) / (n_rows - 1)

    p_score = scaled_rank(df[primary_col]).fillna(0.0)
    s_score = scaled_rank(df[secondary_col]).fillna(0.0)
    t_score = scaled_rank(df[tertiary_col]).fillna(0.0)
    c_score = scaled_rank(df[calibration_col]).fillna(0.0)

    df["rank_score"] = 0.55 * p_score + 0.25 * s_score + 0.10 * t_score + 0.10 * c_score
    df["final_rank"] = df["rank_score"].rank(method="dense", ascending=False).astype(int)
    df = df.sort_values(["final_rank", "rank_score"], ascending=[True, False])
    return ensure_leaderboard_columns(df.reset_index(drop=True))


def ensure_leaderboard_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in LEADERBOARD_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[LEADERBOARD_COLUMNS]


def save_leaderboard(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    output = ensure_leaderboard_columns(df)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def qualitative_top20_review(
    texts: Sequence[str],
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    confidences: Sequence[float] | None = None,
    severity: Sequence[str] | None = None,
    high_priority_labels: Sequence[str] = ("quality_issue", "shipping_service_issue"),
    top_n: int = 20,
) -> pd.DataFrame:
    confidence_values = list(confidences) if confidences is not None else [np.nan] * len(y_true)
    severity_values = list(severity) if severity is not None else ["unknown"] * len(y_true)
    rows: list[dict[str, Any]] = []
    for idx, (text, truth, pred, conf, sev) in enumerate(
        zip(texts, y_true, y_pred, confidence_values, severity_values)
    ):
        if truth == pred:
            tag = "correct_prediction"
        elif truth in high_priority_labels and pred not in high_priority_labels:
            tag = "missed_critical_issue"
        elif pd.notna(conf) and float(conf) >= 0.80:
            tag = "overconfident_misclassification"
        else:
            tag = "label_overlap_or_ambiguity"
        rows.append(
            {
                "row_id": idx,
                "text_snippet": str(text)[:220],
                "true_label": truth,
                "pred_label": pred,
                "confidence": float(conf) if pd.notna(conf) else np.nan,
                "severity": sev,
                "error_tag": tag,
                "is_error": int(truth != pred),
            }
        )

    review = pd.DataFrame(rows)
    review = review.sort_values(
        by=["is_error", "severity", "confidence"],
        ascending=[False, True, False],
        na_position="last",
    )
    return review.head(top_n).reset_index(drop=True)


def build_actionable_findings(
    topic_summary: pd.DataFrame,
    category_counts: pd.DataFrame,
    top_n: int = 8,
) -> pd.DataFrame:
    findings: list[dict[str, Any]] = []
    category_lookup = dict(
        zip(
            category_counts["action_category"],
            category_counts["count"],
        )
    )
    for _, row in topic_summary.head(top_n).iterrows():
        urgency = "high" if row.get("size", 0) >= np.percentile(topic_summary["size"], 75) else "medium"
        findings.append(
            {
                "theme": row.get("summary", ""),
                "evidence_volume": int(row.get("size", 0)),
                "suggested_action": f"Investigate root causes linked to: {row.get('top_terms', '')}",
                "urgency": urgency,
            }
        )

    for category, count in sorted(category_lookup.items(), key=lambda kv: kv[1], reverse=True)[:3]:
        findings.append(
            {
                "theme": f"High-frequency category: {category}",
                "evidence_volume": int(count),
                "suggested_action": f"Assign owner and weekly mitigation plan for {category}.",
                "urgency": "high" if count >= np.percentile(list(category_lookup.values()), 75) else "medium",
            }
        )

    return pd.DataFrame(findings).drop_duplicates(subset=["theme"]).reset_index(drop=True)
