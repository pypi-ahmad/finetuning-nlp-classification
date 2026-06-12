"""End-to-end experiment pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from transformers import set_seed

from lora_qlora_lab.config import Settings
from lora_qlora_lab.data.emotion_dataset import (
    build_eval_records,
    load_emotion_splits,
    sample_splits,
    save_eval_records,
    save_raw_splits,
)
from lora_qlora_lab.eval.inference import evaluate_model, save_predictions
from lora_qlora_lab.reporting.reporting import (
    compute_deltas,
    render_markdown_report,
    save_charts,
    save_metrics,
    save_metrics_json,
)
from lora_qlora_lab.training.fine_tune import train_lora, train_qlora


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(settings: Settings) -> dict[str, Any]:
    """Run full baseline/train/evaluate/report flow."""
    set_seed(settings.seed)
    settings.ensure_directories()

    logger.info("Loading and sampling dataset {}", settings.dataset_name)
    dataset = load_emotion_splits(settings.dataset_name, settings.seed)
    sampled = sample_splits(
        dataset=dataset,
        train_n=settings.train_samples,
        val_n=settings.validation_samples,
        test_n=settings.test_samples,
        seed=settings.seed,
    )
    save_raw_splits(sampled, settings.resolved_raw_dir)

    baseline_records = build_eval_records(sampled["test"], settings.baseline_eval_samples)
    tuned_records = build_eval_records(sampled["test"], settings.tuned_eval_samples)
    save_eval_records(baseline_records, settings.resolved_artifacts_dir / "tables" / "eval_records_baseline.json")
    save_eval_records(tuned_records, settings.resolved_artifacts_dir / "tables" / "eval_records_tuned.json")

    logger.info("Evaluating LoRA baseline model")
    baseline_lora_metrics, baseline_lora_preds = evaluate_model(
        model_name=settings.lora_model_name,
        records=baseline_records,
        hf_token=settings.hf_token,
        run_name="baseline_lora",
        adapter_path=None,
        quantized_4bit=False,
    )

    logger.info("Training LoRA model")
    lora_output_dir = settings.resolved_models_dir / "lora_distilgpt2"
    lora_adapter_path, lora_train_metrics = train_lora(settings, sampled, lora_output_dir)

    logger.info("Evaluating tuned LoRA model")
    tuned_lora_metrics, tuned_lora_preds = evaluate_model(
        model_name=settings.lora_model_name,
        records=tuned_records,
        hf_token=settings.hf_token,
        run_name="tuned_lora",
        adapter_path=lora_adapter_path,
        quantized_4bit=False,
    )

    logger.info("Evaluating QLoRA baseline model")
    baseline_qlora_metrics, baseline_qlora_preds = evaluate_model(
        model_name=settings.qlora_model_name,
        records=baseline_records,
        hf_token=settings.hf_token,
        run_name="baseline_qlora",
        adapter_path=None,
        quantized_4bit=True,
    )

    logger.info("Training QLoRA model")
    qlora_output_dir = settings.resolved_models_dir / "qlora_opt350m"
    qlora_adapter_path, qlora_train_metrics = train_qlora(settings, sampled, qlora_output_dir)

    logger.info("Evaluating tuned QLoRA model")
    tuned_qlora_metrics, tuned_qlora_preds = evaluate_model(
        model_name=settings.qlora_model_name,
        records=tuned_records,
        hf_token=settings.hf_token,
        run_name="tuned_qlora",
        adapter_path=qlora_adapter_path,
        quantized_4bit=True,
    )

    results_df = pd.DataFrame(
        [baseline_lora_metrics, tuned_lora_metrics, baseline_qlora_metrics, tuned_qlora_metrics]
    )

    predictions_df = pd.concat(
        [baseline_lora_preds, tuned_lora_preds, baseline_qlora_preds, tuned_qlora_preds],
        ignore_index=True,
    )

    save_predictions(predictions_df, settings.resolved_artifacts_dir / "tables" / "predictions.csv")
    save_metrics(results_df, settings.resolved_artifacts_dir / "metrics" / "evaluation_metrics.csv")
    save_charts(results_df, settings.resolved_artifacts_dir / "charts")
    render_markdown_report(
        dataset_name=settings.dataset_name,
        lora_model=settings.lora_model_name,
        qlora_model=settings.qlora_model_name,
        seed=settings.seed,
        results=results_df,
        output_path=settings.resolved_artifacts_dir / "reports" / "lora_qlora_report.md",
    )

    summary = {
        "dataset": settings.dataset_name,
        "seed": settings.seed,
        "lora_model": settings.lora_model_name,
        "qlora_model": settings.qlora_model_name,
        "results": results_df.to_dict(orient="records"),
        "deltas": compute_deltas(results_df),
        "lora_train_metrics": lora_train_metrics,
        "qlora_train_metrics": qlora_train_metrics,
    }

    save_metrics_json(summary, settings.resolved_artifacts_dir / "metrics" / "summary.json")
    _save_json(settings.resolved_artifacts_dir / "metrics" / "lora_train_metrics.json", lora_train_metrics)
    _save_json(settings.resolved_artifacts_dir / "metrics" / "qlora_train_metrics.json", qlora_train_metrics)

    logger.info("Pipeline completed. Metrics saved to artifacts/metrics")
    return summary
