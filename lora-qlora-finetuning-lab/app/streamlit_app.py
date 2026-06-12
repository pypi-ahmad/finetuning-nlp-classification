"""Streamlit dashboard for LoRA/QLoRA results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"

st.set_page_config(page_title="LoRA vs QLoRA Lab", layout="wide")
st.title("LoRA vs QLoRA Fine-Tuning Lab")
st.caption("Real training outputs from notebook-first project execution")

summary_path = ARTIFACTS / "metrics" / "summary.json"
metrics_csv = ARTIFACTS / "metrics" / "evaluation_metrics.csv"
preds_csv = ARTIFACTS / "tables" / "predictions.csv"

if not summary_path.exists() or not metrics_csv.exists():
    st.warning("Artifacts not found. Run `uv run lora-qlora-lab run-all` first.")
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
results = pd.read_csv(metrics_csv)

col1, col2 = st.columns(2)
col1.metric("LoRA Accuracy Gain", f"{summary['deltas']['lora_accuracy_gain']:.4f}")
col2.metric("QLoRA Accuracy Gain", f"{summary['deltas']['qlora_accuracy_gain']:.4f}")

st.subheader("Evaluation Metrics")
st.dataframe(results, use_container_width=True, hide_index=True)

fig_acc = px.bar(results, x="run_name", y="accuracy", color="model_name", title="Accuracy by Run")
st.plotly_chart(fig_acc, use_container_width=True)

fig_f1 = px.bar(results, x="run_name", y="macro_f1", color="model_name", title="Macro F1 by Run")
st.plotly_chart(fig_f1, use_container_width=True)

if preds_csv.exists():
    st.subheader("Prediction Samples")
    preds = pd.read_csv(preds_csv)
    selected_run = st.selectbox("Run", sorted(preds["run_name"].unique().tolist()))
    view = preds[preds["run_name"] == selected_run].head(50)
    st.dataframe(view, use_container_width=True, hide_index=True)
