# NLP Unstructured Insights (Upgraded Four-Track Workflow)

Practical NLP project for extracting actionable operations insights from messy customer reviews.

Primary deliverable: [`nlp_unstructured_insights.ipynb`](./nlp_unstructured_insights.ipynb)

## Project Goal

Build an end-to-end, decision-ready NLP system that goes beyond sentiment and supports:

- support intelligence,
- product issue discovery,
- service/packaging root-cause signals,
- deployable inference + monitoring plan.

## Setup and Dataset

```bash
git clone https://github.com/pypi-ahmad/nlp-unstructured-insights.git
cd nlp-unstructured-insights
```


Primary dataset: Amazon Fine Food Reviews

- Source: <https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews>
- Local path: `data/raw/amazon/Reviews.csv`

Optional validation dataset:

- Yelp Open Dataset: <https://www.yelp.com/dataset>

## Project Structure

```
nlp-unstructured-insights/
├── artifacts/
├── data/
│   ├── processed/
│   └── raw/
│       ├── amazon/
│       └── yelp/
├── src/
│   ├── insights_pipeline.py
│   └── text_prep.py
├── nlp_unstructured_insights.ipynb
├── pyproject.toml
└── uv.lock
```

## Current Workflow (Notebook Sections)

1. Business Problem and Success Criteria
2. Dataset Access and Data Dictionary
3. Data Cleaning and Leakage Checks
4. Feature Engineering
5. Validation Strategy
6. LazyPredict Discovery Lab
7. Selection of Top 3 Eligible Models
8. Manual Engineering Lab
9. FLAML Optimization Lab
10. PyCaret Experiment Lab
11. Unified Leaderboard and Final Model Ranking
12. Business Recommendation
13. Inference / Deployment Path
14. Monitoring / Drift / Retraining Plan
15. Limitations and Next Steps

## Four Serious Modeling Tracks

### 1) LazyPredict Discovery Lab

- Runs after feature matrix + validation strategy are defined.
- Produces ranked model-family discovery table.
- Excludes unsuitable families when needed.

### 2) LazyPredict -> Top 3 Manual Model Rule

- Only top 3 eligible families from LazyPredict can enter manual engineering.
- Manual model choices are not random.

### 3) Manual Engineering Lab

- Implements the selected top 3 families with explicit training and evaluation.
- Includes CV + holdout metrics, calibration handling, threshold optimization for critical issue routing, and qualitative error analysis.

### 4) FLAML Optimization Lab

- Uses FLAML as a full optimization track (not a single comparison row).
- Uses explicit `time_budget` and project primary metric (`macro_f1`).
- Logs best estimator, key hyperparameters, and comparison vs manual track.

### 5) PyCaret Experiment Lab

- Uses orchestration flow: `setup -> compare_models -> tune_model -> calibrate_model (if available) -> finalize_model -> save_model`.
- Retains and evaluates final PyCaret artifact.

## Final Leaderboard Logic

Unified leaderboard includes:

- top LazyPredict discovery entries,
- manual top-3 engineered models,
- best FLAML result,
- best PyCaret finalized result,
- explicit baseline model.

Saved file:

- `artifacts/leaderboard_nlp_unified.csv`

Leaderboard columns:

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

Ranking weights (rank-normalized):

- 55% primary metric (macro F1)
- 25% secondary metric (weighted F1)
- 10% tertiary metric (critical issue recall)
- 10% calibration/confidence quality

Top candidates are also rerun across multiple seeds and saved to:

- `artifacts/top3_candidates_multiseed.csv`
- `artifacts/top3_candidates_multiseed_summary.csv`

## Deployment / Inference Path

Notebook saves an inference bundle for deployment:

- `artifacts/inference_bundle.pkl`

Bundle contents:

- fitted vectorizer,
- selected deployable model,
- class order,
- priority threshold,
- metadata.

Prediction preview file:

- `artifacts/inference_preview.csv`

## Monitoring / Drift / Retraining

Notebook exports:

- `artifacts/monitoring_plan.csv`
- `artifacts/drift_baseline.json`

Monitoring covers:

- macro F1 decay,
- critical-issue recall drop,
- text/input drift (length/OOV),
- error-tag mix shifts.

## Exact Run Instructions

From project root:

```bash
cd nlp-unstructured-insights
uv sync
```

Dataset download (if missing):

```bash
uv run kaggle datasets download -d snap/amazon-fine-food-reviews -p data/raw/amazon --unzip
```

Run notebook:

```bash
uv run jupyter notebook nlp_unstructured_insights.ipynb
```

Optional headless execution:

```bash
uv run jupyter nbconvert --to notebook --execute nlp_unstructured_insights.ipynb --output nlp_unstructured_insights.executed.ipynb
```

## Notes and Limitations

- Labels are weakly supervised and partially heuristic-derived.
- Leaderboard is holdout-based; maintain periodic human QA and relabeling.
- AutoML tracks depend on available compute and package versions.
