"""Metrics aggregation and report rendering."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
from jinja2 import Template

_REPORT_TEMPLATE = """# LoRA vs QLoRA Fine-Tuning Report

Generated at: `{{ generated_at }}`

## Experiment Setup

- Dataset: `{{ dataset_name }}`
- Task: Emotion classification with generative label prediction
- LoRA model: `{{ lora_model }}`
- QLoRA model: `{{ qlora_model }}`
- Seed: `{{ seed }}`

## Evaluation Results

| Run | Model | Accuracy | Macro F1 | Samples |
|---|---|---:|---:|---:|
{% for row in rows %}| {{ row.run_name }} | {{ row.model_name }} | {{ '%.4f'|format(row.accuracy) }} | {{ '%.4f'|format(row.macro_f1) }} | {{ row.n_samples }} |
{% endfor %}

## Delta vs Baseline

- LoRA gain (accuracy): **{{ '%.4f'|format(deltas.lora_accuracy_gain) }}**
- QLoRA gain (accuracy): **{{ '%.4f'|format(deltas.qlora_accuracy_gain) }}**
- LoRA gain (macro F1): **{{ '%.4f'|format(deltas.lora_f1_gain) }}**
- QLoRA gain (macro F1): **{{ '%.4f'|format(deltas.qlora_f1_gain) }}**

## Notes

- Baselines are measured before adapter training on the same held-out split.
- Train/validation/test are sampled deterministically with a fixed seed.
"""


def compute_deltas(results: pd.DataFrame) -> dict[str, float]:
    """Compute fine-tuning gains over baselines."""
    baseline_lora = results.loc[results["run_name"] == "baseline_lora"].iloc[0]
    tuned_lora = results.loc[results["run_name"] == "tuned_lora"].iloc[0]
    baseline_qlora = results.loc[results["run_name"] == "baseline_qlora"].iloc[0]
    tuned_qlora = results.loc[results["run_name"] == "tuned_qlora"].iloc[0]

    return {
        "lora_accuracy_gain": float(tuned_lora["accuracy"] - baseline_lora["accuracy"]),
        "qlora_accuracy_gain": float(tuned_qlora["accuracy"] - baseline_qlora["accuracy"]),
        "lora_f1_gain": float(tuned_lora["macro_f1"] - baseline_lora["macro_f1"]),
        "qlora_f1_gain": float(tuned_qlora["macro_f1"] - baseline_qlora["macro_f1"]),
    }


def save_metrics(results: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)


def _sanitize_for_json(value: Any) -> Any:
    """Convert non-JSON scalar sentinels (for example NaN) into JSON-safe values."""
    if isinstance(value, dict):
        return {key: _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    if value is pd.NA:
        return None
    try:
        if math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def save_metrics_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_payload = _sanitize_for_json(payload)
    output_path.write_text(json.dumps(clean_payload, indent=2, allow_nan=False), encoding="utf-8")


def save_charts(results: pd.DataFrame, chart_dir: Path) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)

    chart_df = results.copy()
    fig_acc = px.bar(
        chart_df,
        x="run_name",
        y="accuracy",
        color="model_name",
        title="Accuracy by Run",
        text="accuracy",
    )
    fig_acc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_acc.write_html(chart_dir / "accuracy_by_run.html", include_plotlyjs="cdn")

    fig_f1 = px.bar(
        chart_df,
        x="run_name",
        y="macro_f1",
        color="model_name",
        title="Macro F1 by Run",
        text="macro_f1",
    )
    fig_f1.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_f1.write_html(chart_dir / "macro_f1_by_run.html", include_plotlyjs="cdn")


def render_markdown_report(
    dataset_name: str,
    lora_model: str,
    qlora_model: str,
    seed: int,
    results: pd.DataFrame,
    output_path: Path,
) -> str:
    rows = results.to_dict(orient="records")
    deltas = compute_deltas(results)
    text = Template(_REPORT_TEMPLATE).render(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        dataset_name=dataset_name,
        lora_model=lora_model,
        qlora_model=qlora_model,
        seed=seed,
        rows=rows,
        deltas=deltas,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text
