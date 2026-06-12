"""End-to-end pipeline: data -> baseline eval -> LoRA SFT -> DPO -> final eval."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from .data import build_preference_split, build_splits
from .evaluate import evaluate_stage
from .settings import ProjectConfig, load_config
from .train import train_dpo, train_sft
from .utils import ensure_dir, set_seed, write_json, write_jsonl


def _make_run_dir(cfg: ProjectConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_root) / f"run_{timestamp}"
    ensure_dir(run_dir)
    return run_dir


def _save_data_artifacts(run_dir: Path, splits: dict[str, list[dict[str, Any]]], prefs: list[dict[str, Any]]) -> None:
    data_dir = run_dir / "data"
    ensure_dir(data_dir)
    write_jsonl(data_dir / "train_sft.jsonl", splits["train"])
    write_jsonl(data_dir / "val_sft.jsonl", splits["val"])
    write_jsonl(data_dir / "test_eval.jsonl", splits["test"])
    write_jsonl(data_dir / "train_dpo_prefs.jsonl", prefs)


def _metrics_table(metrics_by_stage: list[dict[str, Any]]) -> str:
    lines = [
        "| Stage | Valid JSON | Schema Match | Exact Match | Field Accuracy | Avg Latency (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_by_stage:
        lines.append(
            "| {stage} | {valid:.4f} | {schema:.4f} | {exact:.4f} | {field:.4f} | {latency:.2f} |".format(
                stage=row["stage"],
                valid=row["valid_json_rate"],
                schema=row["schema_match_rate"],
                exact=row["exact_match_rate"],
                field=row["field_accuracy"],
                latency=row["avg_latency_ms"],
            )
        )
    return "\n".join(lines)


def _save_markdown_report(run_dir: Path, metrics: dict[str, Any]) -> None:
    table = _metrics_table(metrics["stages"])
    deltas = metrics["deltas"]

    report = f"""# Run Report

- Run directory: `{run_dir}`
- Model: `{metrics['model_name']}`
- CUDA available: `{metrics['cuda_available']}`
- QLoRA requested: `{metrics['qlora_requested']}`
- Data source: `{metrics['data_source']}`

## Metrics

{table}

## Improvements

- LoRA vs Base exact match: `{deltas['sft_vs_base_exact_match_delta']:+.4f}`
- DPO vs LoRA exact match: `{deltas['dpo_vs_sft_exact_match_delta']:+.4f}`
- DPO vs Base exact match: `{deltas['dpo_vs_base_exact_match_delta']:+.4f}`
- DPO vs Base field accuracy: `{deltas['dpo_vs_base_field_accuracy_delta']:+.4f}`
"""
    (run_dir / "RUN_REPORT.md").write_text(report, encoding="utf-8")


def run(cfg: ProjectConfig) -> Path:
    set_seed(cfg.seed)
    run_dir = _make_run_dir(cfg)
    logger.add(run_dir / "run.log", level="INFO")

    logger.info("Run directory: {}", run_dir)
    logger.info("Loading data splits")

    splits = build_splits(
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
        seed=cfg.seed,
        source=cfg.data.source,
        hf_dataset=cfg.data.hf_dataset,
        hf_train_split=cfg.data.hf_train_split,
        hf_test_split=cfg.data.hf_test_split,
    )
    preference_rows = build_preference_split(splits["train"], seed=cfg.seed)
    _save_data_artifacts(run_dir, splits, preference_rows)

    write_json(run_dir / "config_snapshot.json", json.loads(cfg.model_dump_json()))

    logger.info("Stage 1/5: baseline evaluation")
    base_metrics = evaluate_stage(
        cfg=cfg,
        stage="base",
        rows=splits["test"],
        output_dir=run_dir,
    )

    logger.info("Stage 2/5: LoRA SFT training")
    sft_summary = train_sft(cfg, splits["train"], splits["val"], run_dir)

    logger.info("Stage 3/5: evaluate SFT adapter")
    sft_metrics = evaluate_stage(
        cfg=cfg,
        stage="sft",
        rows=splits["test"],
        output_dir=run_dir,
        adapter_dir=sft_summary["adapter_dir"],
    )

    logger.info("Stage 4/5: DPO training")
    dpo_summary = train_dpo(
        cfg=cfg,
        preference_rows=preference_rows,
        run_dir=run_dir,
        sft_adapter_dir=sft_summary["adapter_dir"],
    )

    logger.info("Stage 5/5: evaluate DPO adapter")
    dpo_metrics = evaluate_stage(
        cfg=cfg,
        stage="dpo",
        rows=splits["test"],
        output_dir=run_dir,
        adapter_dir=dpo_summary["adapter_dir"],
    )

    summary = {
        "model_name": cfg.model.name,
        "cuda_available": torch.cuda.is_available(),
        "qlora_requested": cfg.model.use_qlora,
        "data_source": cfg.data.source,
        "stages": [base_metrics, sft_metrics, dpo_metrics],
        "sft_training": sft_summary,
        "dpo_training": dpo_summary,
        "deltas": {
            "sft_vs_base_exact_match_delta": sft_metrics["exact_match_rate"] - base_metrics["exact_match_rate"],
            "dpo_vs_sft_exact_match_delta": dpo_metrics["exact_match_rate"] - sft_metrics["exact_match_rate"],
            "dpo_vs_base_exact_match_delta": dpo_metrics["exact_match_rate"] - base_metrics["exact_match_rate"],
            "dpo_vs_base_field_accuracy_delta": dpo_metrics["field_accuracy"] - base_metrics["field_accuracy"],
        },
    }

    write_json(run_dir / "metrics_summary.json", summary)
    _save_markdown_report(run_dir, summary)
    logger.info("Pipeline completed. Metrics saved to {}", run_dir / "metrics_summary.json")

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA + DPO JSON extraction pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = run(cfg)
    logger.info("Completed run at {}", run_dir)


if __name__ == "__main__":
    main()
