# Zero to Hero Study Handbook: NLP Unstructured Insights

## Module 1: Foundations & Architecture

- This project builds an end-to-end NLP workflow that converts raw customer review text into operational categories (for example, `quality_issue`, `shipping_service_issue`) and then compares multiple modeling tracks to pick a deployable classifier.
- The main use case is operations intelligence from reviews: support triage, product/service issue detection, and deployment-ready prediction outputs with monitoring artifacts.
- The actual main runtime is the notebook [`nlp_unstructured_insights.ipynb`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/nlp_unstructured_insights.ipynb), not [`main.py`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/main.py) (which only prints a hello message).

Core paradigms and patterns used in this repo:

- Notebook-orchestrated pipeline: execution order is defined cell-by-cell in [`nlp_unstructured_insights.ipynb`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/nlp_unstructured_insights.ipynb).
- Functional utility modules: most logic is in pure-ish functions in [`src/text_prep.py`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/src/text_prep.py) and [`src/insights_pipeline.py`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/src/insights_pipeline.py).
- Dataclass config pattern: `PreprocessingConfig` and `RetrievalResult` use `@dataclass(slots=True)` for typed structured data.
- Weak-labeling heuristic pattern: `infer_actionable_category()` maps text + optional rating to operational labels via keyword cues and fallback rules.
- Multi-track model selection pattern: LazyPredict discovery -> manual top-3 engineering -> FLAML optimization -> PyCaret orchestration -> weighted unified ranking.
- Optional dependency fallback pattern: `try/except ImportError` for `faiss`, `LazyClassifier`, and `AutoML`; when missing, functions return safe empty/failure rows instead of crashing.

Architecture (what talks to what):

- Data source layer:
  - Raw CSV at `data/raw/amazon/Reviews.csv` (downloaded via Kaggle CLI if absent).
- Preprocessing/labeling layer:
  - `prepare_reviews_frame()` cleans text, creates engineered fields, infers `action_category`, computes `severity`.
- Feature layer:
  - TF-IDF sparse features + TruncatedSVD dense features for tracks that need dense input.
- Modeling tracks:
  - LazyPredict discovery (`run_lazypredict_discovery()`).
  - Manual top-3 families (`evaluate_manual_family()`).
  - FLAML AutoML (`run_flaml_optimization()`).
  - PyCaret workflow (`run_pycaret_experiment()`).
- Evaluation and governance:
  - Unified leaderboard (`rank_leaderboard()`, `save_leaderboard()`), multi-seed stability reruns.
- Deployment and monitoring:
  - Inference bundle (`save_inference_bundle()` / `predict_with_inference_bundle()`).
  - Monitoring artifacts (`monitoring_plan.csv`, `drift_baseline.json`).

ASCII architecture flow:

```text
Raw Reviews.csv
    |
    v
prepare_reviews_frame() in src/text_prep.py
    -> clean_text, action_category, severity, needs_action
    |
    v
train_test_split + TfidfVectorizer (+ TruncatedSVD for dense)
    |-----------------------------|---------------------------|---------------------------|
    v                             v                           v                           v
LazyPredict Discovery       Manual Top-3 Families       FLAML Optimization         PyCaret Experiment
run_lazypredict_discovery   evaluate_manual_family      run_flaml_optimization     run_pycaret_experiment
    |                             |                           |                           |
    |----------- selected_top3 ----                           |                           |
                    |                                         |                           |
                    +--------------------- all rows ----------+-----------+---------------+
                                                                  |
                                                                  v
                                                        rank_leaderboard()
                                                                  |
                                                          leaderboard_nlp_unified.csv
                                                                  |
                                                                  v
                              save_inference_bundle() + predict_with_inference_bundle()
                                                                  |
                                            inference_bundle.pkl / inference_preview.csv
                                                                  |
                                                                  v
                                              monitoring_plan.csv / drift_baseline.json
```

## Module 2: Repository Map

| File/Directory Path | Primary Responsibility | Key Classes/Functions | Important Configs/Variables |
|---|---|---|---|
| [`README.md`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/README.md) | Human-readable project overview, workflow narrative, run commands, artifact expectations | N/A | Dataset path `data/raw/amazon/Reviews.csv`, run commands (`uv sync`, notebook execution commands) |
| [`pyproject.toml`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/pyproject.toml) | Project metadata and dependency manifest | N/A | `requires-python = "==3.12.10"`, dependency list, `[tool.uv] prerelease = "allow"` |
| [`nlp_unstructured_insights.ipynb`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/nlp_unstructured_insights.ipynb) | Main orchestrator: ingestion, preprocessing, feature engineering, 4 modeling tracks, ranking, deployment, monitoring | Calls `prepare_reviews_frame`, `run_lazypredict_discovery`, `evaluate_manual_family`, `run_flaml_optimization`, `run_pycaret_experiment`, `rank_leaderboard`, `save_inference_bundle`, `predict_with_inference_bundle` | `PROJECT_NAME`, `RANDOM_STATE`, `PRIMARY_METRIC`, `vectorizer_params`, `TARGET_COL`, artifact/data paths |
| [`src/text_prep.py`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/src/text_prep.py) | Text cleaning, heuristic category inference, severity bucketing, pre-model DataFrame preparation, category term summaries | `PreprocessingConfig`, `normalize_text`, `strip_stopwords`, `infer_actionable_category`, `severity_bucket`, `prepare_reviews_frame`, `top_terms_by_category`, `build_relevance_mask` | `ACTION_KEYWORDS`, `POSITIVE_CUES`, default config (`sample_size=30000`, `random_state=42`) |
| [`src/insights_pipeline.py`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/src/insights_pipeline.py) | Modeling/evaluation utilities, leaderboard mechanics, inference packaging, qualitative review, optional retrieval/topic helpers | `RetrievalResult`, `run_lazypredict_discovery`, `select_top_eligible_from_lazypredict`, `evaluate_manual_family`, `run_flaml_optimization`, `run_pycaret_experiment`, `rank_leaderboard`, `save_inference_bundle`, `predict_with_inference_bundle`, `qualitative_top20_review` | `LEADERBOARD_COLUMNS`, `LAZY_TO_MANUAL_FAMILY`, `MANUAL_FAMILY_DISPLAY`, ranking weights in `rank_leaderboard()` |
| [`main.py`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/main.py) | Minimal script placeholder | `main()` | No runtime config; not the ML pipeline entrypoint |
| [`uv.lock`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/uv.lock) | Resolved lockfile for reproducible dependency versions | N/A | Exact package resolution for `uv` |

Files to learn first, in order:

- [`nlp_unstructured_insights.ipynb`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/nlp_unstructured_insights.ipynb)
- [`src/text_prep.py`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/src/text_prep.py)
- [`src/insights_pipeline.py`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/src/insights_pipeline.py)
- [`pyproject.toml`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/pyproject.toml)

## Module 3: Core Execution Flows

### Flow A: Data Ingestion and Preprocessing

Step-by-step runtime path in notebook:

1. Cell 4 loads `Reviews.csv` into `raw_df` with selected columns.
2. If file is missing, notebook tries a Kaggle CLI download via `subprocess.run([...])`.
3. Cell 6 calls:

```python
clean_df = prepare_reviews_frame(
    raw_df,
    PreprocessingConfig(text_col="Text", score_col="Score", sample_size=14000, random_state=RANDOM_STATE),
)
```

4. `prepare_reviews_frame()` in [`src/text_prep.py`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/src/text_prep.py):
   - Validates required columns (`Text`, `Score` by default).
   - Drops duplicate texts.
   - Creates:
     - `raw_text`
     - `clean_text` via `normalize_text()`
     - `clean_text_nostop` via `strip_stopwords()`
     - `text_len_chars`
     - `text_len_tokens`
     - `action_category` via `infer_actionable_category()`
     - `severity` via `severity_bucket()`
     - `needs_action` (1 unless category is `praise_loyalty`)
   - Optionally drops empty cleaned text.
   - Samples rows if over `sample_size`.

Exact output shape highlights:

- Input DataFrame must include at least columns named by `PreprocessingConfig.text_col` and `.score_col`.
- Output DataFrame includes original columns plus engineered columns listed above.
- `action_category` possible values come from rules:
  - Keyword buckets: `quality_issue`, `shipping_service_issue`, `pricing_value_issue`, `usability_packaging_issue`, `taste_preference`
  - Fallbacks: `praise_loyalty`, `general_negative_experience`, `other_actionable`

### Flow B: Feature Engineering and Validation Split

1. Cell 8 filters low-frequency labels:
   - `TARGET_COL = "action_category"`
   - keeps labels with count `>= min_class_size` (`80`).
2. Cell 10 performs:
   - Stratified train/holdout split (`test_size=0.20`).
   - TF-IDF fitting on train only:

```python
vectorizer = TfidfVectorizer(**vectorizer_params)
X_train_sparse = vectorizer.fit_transform(train_text)
X_valid_sparse = vectorizer.transform(valid_text)
```

3. Dense projection for dense-track models:

```python
svd = TruncatedSVD(n_components=n_svd, random_state=RANDOM_STATE)
X_train_dense = svd.fit_transform(X_train_sparse)
X_valid_dense = svd.transform(X_valid_sparse)
```

Key config keys used:

- `vectorizer_params = {"max_features": 12000, "ngram_range": (1,2), "min_df": 3, "max_df": 0.92, "sublinear_tf": True}`
- `RANDOM_STATE = 42`

### Flow C: Four Modeling Tracks

Track 1: LazyPredict discovery (Cell 12)

- Calls `run_lazypredict_discovery()` with dense features.
- Returns two DataFrames:
  - full LazyPredict table (`lazy_models_df`)
  - normalized leaderboard row table (`lazy_rows_df`)
- Each normalized row follows `make_leaderboard_row()` schema fields.

Track 2: Top-3 family selection + manual engineering (Cells 14 and 16)

1. Calls `select_top_eligible_from_lazypredict(lazy_models_df, top_n=3)`.
2. Applies family mapping using `LAZY_TO_MANUAL_FAMILY`.
3. Excludes patterns like `"dummy"`, `"gaussian"`, `"knn"`.
4. For each selected family, notebook calls `evaluate_manual_family(...)`.
5. `evaluate_manual_family()` internally:
   - Builds estimator via `build_manual_estimator()`.
   - Runs CV `f1_macro` using `StratifiedKFold`.
   - Fits model and predicts holdout labels.
   - Optionally calibrates non-probabilistic models (`CalibratedClassifierCV`).
   - Produces metrics bundle and leaderboard row.
   - Returns `(row_dict, artifacts_dict)`.

`artifacts_dict` keys:

- `family_name`, `model`, `raw_model`, `y_pred`, `y_prob`, `classes_`, `calibrated`

Priority threshold path:

- If `y_prob` exists, notebook calls `optimize_priority_threshold(...)`.
- Output keys: `best_threshold`, `best_fbeta`, `priority_recall`, `priority_precision`.

Qualitative error path:

- `qualitative_top20_review()` returns DataFrame with:
  - `row_id`, `text_snippet`, `true_label`, `pred_label`, `confidence`, `severity`, `error_tag`, `is_error`

Track 3: FLAML optimization (Cell 18)

- Calls `run_flaml_optimization(...)` with sparse features.
- Returns `(flaml_row, flaml_details, flaml_model)`.
- `flaml_row` is leaderboard-compatible.
- `flaml_details` includes:
  - `best_estimator`, `best_config`, `best_loss`, `best_iteration`, `best_config_train_time`, `search_estimators`, `requested_estimators`, `best_result`, `fit_errors`
- Includes fallback estimator list `("rf", "extra_tree", "lrl1")` if first attempt fails.

Track 4: PyCaret experiment (Cell 20)

- Calls `run_pycaret_experiment(...)` with dense arrays + feature names.
- Workflow inside function is explicitly:
  - `ClassificationExperiment(...).fit(...)`
  - `compare_models(...)`
  - `tune_model(...)`
  - `calibrate_model(...)` (inside try/except)
  - `finalize_model(...)`
  - `save_model(...)`
  - `predict_model(...)` on validation
- Returns `(pycaret_row, pycaret_details, pycaret_model)`.
- `pycaret_details` keys include `compare_table`, `tune_table`, `final_model_class`, `saved_model_path`, `include_models`, `calibration_note`, `compared_model_count`.

### Flow D: Unified Ranking, Deployment, and Monitoring

Unified ranking (Cell 22):

1. Builds baseline row using `DummyClassifier(strategy="most_frequent")`.
2. Concatenates baseline + LazyPredict rows + manual rows + FLAML row + PyCaret row.
3. Calls `rank_leaderboard()`.

`rank_leaderboard()` scoring logic:

- Rank-normalizes each metric column.
- Computes `rank_score = 0.55*primary + 0.25*secondary + 0.10*tertiary + 0.10*calibration`.
- Produces dense integer `final_rank`.
- Ensures output column order via `ensure_leaderboard_columns()`.

Leaderboard schema (`LEADERBOARD_COLUMNS`):

- `project_name`
- `task_type`
- `library_source`
- `model_name`
- `cv_metric_mean`
- `cv_metric_std`
- `holdout_primary_metric`
- `holdout_secondary_metric`
- `holdout_tertiary_metric`
- `calibration_metric`
- `train_time_sec`
- `infer_latency_ms`
- `model_size_mb`
- `interpretability_note`
- `rank_score`
- `final_rank`

Deployment path (Cell 26):

1. Selects `deployment_family` from manual candidates.
2. Builds metadata dict and calls:

```python
bundle_path = save_inference_bundle(
    ARTIFACTS / "inference_bundle.pkl",
    vectorizer=vectorizer,
    model=deployment_model,
    class_order=deployment_classes,
    priority_threshold=threshold_result.get("best_threshold"),
    metadata=bundle_metadata,
)
```

3. Loads bundle and runs:

```python
deploy_preview = predict_with_inference_bundle(loaded_bundle, sample_texts)
```

`predict_with_inference_bundle()` output columns:

- Always: `text`, `predicted_label`, `confidence`, `priority_probability`
- Conditional: `trigger_priority_review` if `priority_threshold` is set

Monitoring path (Cell 28):

- Writes `monitoring_plan.csv` with columns:
  - `signal`, `threshold`, `cadence`, `action`
- Writes `drift_baseline.json` with keys:
  - `train_text_len_tokens_quantiles`
  - `train_label_distribution`
  - `vocab_size`

## Module 4: Setup & Run Guide

### 1) Environment and Dependencies

- Python version is pinned in [`pyproject.toml`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/pyproject.toml): `==3.12.10`.
- Package manager is `uv` (lockfile present: [`uv.lock`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/uv.lock)).
- Install command (from README):

```bash
uv sync
```

### 2) Data Prerequisites

- Expected raw dataset file:
  - `data/raw/amazon/Reviews.csv`
- If missing, README/notebook use this command:

```bash
uv run kaggle datasets download -d snap/amazon-fine-food-reviews -p data/raw/amazon --unzip
```

### 3) Run Commands

- Interactive notebook:

```bash
uv run jupyter notebook nlp_unstructured_insights.ipynb
```

- Headless execution:

```bash
uv run jupyter nbconvert --to notebook --execute nlp_unstructured_insights.ipynb --output nlp_unstructured_insights.executed.ipynb
```

### 4) Configuration Surfaces

- Code-level constants in notebook (Cell 2 and later):
  - `PROJECT_NAME`
  - `RANDOM_STATE`
  - metric name constants
  - path constants (`DATA_RAW`, `DATA_PROCESSED`, `ARTIFACTS`)
  - `vectorizer_params`
  - train/validation split and class-size thresholds
- Module-level mappings/constants:
  - `ACTION_KEYWORDS`, `POSITIVE_CUES` in [`src/text_prep.py`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/src/text_prep.py)
  - `LEADERBOARD_COLUMNS`, model family maps in [`src/insights_pipeline.py`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/src/insights_pipeline.py)

### 5) Environment Variables / `.env` Keys

- No `.env` file or explicit `os.environ`/`getenv` keys are defined in this repository.
- Practical note from code behavior:
  - Kaggle CLI authentication is required only if auto-download is used; this repo does not define the auth key names itself.

### 6) Migrations/Seeding/External Services

- No database migrations are present.
- No seed scripts are present.
- External service dependency in active path is Kaggle dataset download (optional when CSV already exists).

## Module 5: Study Plan & Practice Exercises

### Ordered Study Plan

1. Read [`README.md`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/README.md) to understand goals, tracks, and artifact expectations.
2. Walk through notebook markdown sections in [`nlp_unstructured_insights.ipynb`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/nlp_unstructured_insights.ipynb) to map business problem to technical steps.
3. Study [`src/text_prep.py`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/src/text_prep.py) to internalize labeling heuristics and preprocessing outputs.
4. Revisit notebook cells 6, 8, 10 to connect preprocessing outputs to split/features.
5. Study modeling helpers in [`src/insights_pipeline.py`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/src/insights_pipeline.py):
   - LazyPredict selection
   - manual evaluation
   - FLAML and PyCaret paths
6. Revisit notebook cells 12–22 to see how helper functions are orchestrated.
7. Study deployment and monitoring in notebook cells 26 and 28.
8. Read [`pyproject.toml`](/home/ahmad/AI/Github/finetuning-nlp-classification/nlp-unstructured-insights/pyproject.toml) to learn dependency scope and runtime assumptions.

### Practice Exercises (with Solution Outlines)

1. Exercise: In `prepare_reviews_frame()`, list all columns that are newly created and explain one business purpose for each.
   - Solution outline: New columns are `raw_text`, `clean_text`, `clean_text_nostop`, `text_len_chars`, `text_len_tokens`, `action_category`, `severity`, `needs_action`; they support model input cleaning, weak-label targets, and operational triage.

2. Exercise: Explain exactly how a review becomes `praise_loyalty` instead of `other_actionable`.
   - Solution outline: `infer_actionable_category()` first checks keyword category scores; if none matched, it checks positive cue words (`POSITIVE_CUES`) and rating logic (`rating >= 4` + positive cue, or positive cue without rating) to return `praise_loyalty`; otherwise fallback is `other_actionable` or `general_negative_experience` when rating `<= 2`.

3. Exercise: Trace how leakage prevention is implemented before model training.
   - Solution outline: Notebook cell 6 creates leakage report and keeps text-only features; cell 10 fits `TfidfVectorizer` only on train split (`fit_transform(train_text)`, `transform(valid_text)`), avoiding holdout leakage.

4. Exercise: How are manual candidate models selected from LazyPredict?
   - Solution outline: `select_top_eligible_from_lazypredict()` normalizes model names, maps to families with `LAZY_TO_MANUAL_FAMILY`, excludes disallowed patterns, sorts by F1 column, removes duplicate families, and keeps top `n`.

5. Exercise: Compare how calibration is handled in manual track vs PyCaret track.
   - Solution outline: Manual path calibrates only when `predict_proba` is unavailable and only keeps calibrated model if macro F1 is not worse; PyCaret path attempts `calibrate_model(...)` in try/except and continues with tuned model if unavailable.

6. Exercise: Reconstruct the leaderboard scoring formula and explain why a model with best macro F1 might still lose overall.
   - Solution outline: `rank_score = 0.55*primary + 0.25*secondary + 0.10*tertiary + 0.10*calibration` using rank-normalized metrics; a model can lose if it dominates macro F1 but underperforms on weighted F1, critical recall, or calibration.

7. Exercise: Describe the serialized inference bundle contract and how predictions are generated from it.
   - Solution outline: `save_inference_bundle()` stores dict keys `vectorizer`, `model`, `class_order`, `priority_threshold`, `metadata`; `predict_with_inference_bundle()` vectorizes text, predicts labels, computes confidence and priority probability, and conditionally emits `trigger_priority_review`.

8. Exercise: Identify all artifacts produced for governance/monitoring and what question each answers.
   - Solution outline:
     - `leaderboard_nlp_unified.csv`: model ranking decision basis.
     - `top3_candidates_multiseed*.csv`: stability across seeds.
     - `inference_bundle.pkl` and `inference_preview.csv`: deployability check.
     - `monitoring_plan.csv` and `drift_baseline.json`: post-deploy health and drift thresholds.

9. Exercise: Which code path is authoritative for production logic: README claims or notebook code? Justify with specific examples.
   - Solution outline: Notebook code is authoritative runtime path; for example, class-size filter (`min_class_size = 80`), vectorizer settings, and exact track invocation are in notebook cells, while README is descriptive.

10. Exercise: If LazyPredict is unavailable, what deterministic fallback behavior exists in this repo?
   - Solution outline: Notebook cell 14 builds a fallback DataFrame with three families (`logistic_regression`, `linear_svc`, `random_forest`) and proceeds through manual lab so the pipeline remains executable.

## Learner Verification Checklist

- Can you explain the complete path from raw `Reviews.csv` row to `action_category` and `severity`?
- Can you describe exactly why TF-IDF is fit on train split only and where this happens?
- Can you trace one manual model from family selection to leaderboard row creation?
- Can you explain all fields in the unified leaderboard and how `final_rank` is computed?
- Can you describe when calibration is attempted and when it is skipped in both manual and PyCaret tracks?
- Can you explain the inference bundle schema and each output column in `predict_with_inference_bundle()`?
- Can you list the monitoring signals and the corresponding trigger actions written by the notebook?
- Can you state which files are orchestration (`.ipynb`) vs reusable logic (`src/*.py`) vs packaging (`pyproject.toml`, `uv.lock`)?
