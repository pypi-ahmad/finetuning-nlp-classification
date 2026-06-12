from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lora_qlora_lab.reporting.reporting import compute_deltas, save_metrics_json


def test_compute_deltas() -> None:
    df = pd.DataFrame(
        [
            {"run_name": "baseline_lora", "accuracy": 0.2, "macro_f1": 0.1},
            {"run_name": "tuned_lora", "accuracy": 0.3, "macro_f1": 0.2},
            {"run_name": "baseline_qlora", "accuracy": 0.25, "macro_f1": 0.2},
            {"run_name": "tuned_qlora", "accuracy": 0.35, "macro_f1": 0.3},
        ]
    )
    deltas = compute_deltas(df)

    assert round(deltas["lora_accuracy_gain"], 4) == 0.1
    assert round(deltas["qlora_accuracy_gain"], 4) == 0.1


def test_save_metrics_json_normalizes_nan(tmp_path: Path) -> None:
    out_path = tmp_path / "summary.json"
    payload = {"results": [{"adapter_path": float("nan"), "accuracy": 0.5}]}

    save_metrics_json(payload, out_path)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))

    assert loaded["results"][0]["adapter_path"] is None
