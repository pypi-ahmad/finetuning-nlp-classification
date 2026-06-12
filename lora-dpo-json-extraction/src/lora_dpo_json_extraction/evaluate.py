"""Evaluation helpers for baseline, SFT, and DPO checkpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from transformers import AutoModelForCausalLM

from .models import load_adapter_for_inference, load_base_model, load_tokenizer
from .settings import ProjectConfig
from .utils import extract_first_json_object, ensure_dir, write_jsonl

EXPECTED_KEYS = ["intent", "priority", "product", "needs_human"]


def _coerce_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in EXPECTED_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if key == "needs_human" and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                value = True
            elif lowered in {"false", "no", "0"}:
                value = False
        normalized[key] = value
    return normalized


def _safe_parse_json(text: str) -> dict[str, Any] | None:
    fragment = extract_first_json_object(text)
    if not fragment:
        return None
    try:
        parsed = json.loads(fragment)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return _coerce_prediction(parsed)


def _generate_prediction(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    output_tokens = generated[0][input_len:]
    return tokenizer.decode(output_tokens, skip_special_tokens=True)


def _build_model(cfg: ProjectConfig, stage: str, adapter_dir: str | None):
    if stage == "base":
        model = load_base_model(cfg.model, for_training=False)
        model.eval()
        return model
    if not adapter_dir:
        raise ValueError(f"adapter_dir is required for stage={stage}")
    return load_adapter_for_inference(cfg.model, adapter_dir)


def evaluate_stage(
    cfg: ProjectConfig,
    stage: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    adapter_dir: str | None = None,
) -> dict[str, Any]:
    """Run generation-based evaluation and persist per-example predictions."""

    ensure_dir(output_dir)
    tokenizer = load_tokenizer(cfg.model.name)
    model = _build_model(cfg, stage=stage, adapter_dir=adapter_dir)

    valid_json = 0
    exact_match = 0
    schema_match = 0
    field_hits = 0
    total_fields = len(rows) * len(EXPECTED_KEYS)
    latencies_ms: list[float] = []

    prediction_rows: list[dict[str, Any]] = []

    for row in rows:
        prompt = row["prompt"]
        target = row["target"]

        start = time.perf_counter()
        raw_output = _generate_prediction(model, tokenizer, prompt, cfg.eval.max_new_tokens)
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(latency_ms)

        parsed = _safe_parse_json(raw_output)
        is_valid = parsed is not None
        if is_valid:
            valid_json += 1
            key_set_ok = set(parsed.keys()) == set(EXPECTED_KEYS)
            schema_match += int(key_set_ok)
            for key in EXPECTED_KEYS:
                field_hits += int(parsed.get(key) == target.get(key))
            exact_match += int(parsed == target)

        prediction_rows.append(
            {
                "prompt": prompt,
                "target": target,
                "prediction_text": raw_output,
                "prediction_json": parsed,
                "valid_json": is_valid,
                "exact_match": bool(parsed == target) if parsed is not None else False,
                "latency_ms": latency_ms,
            }
        )

    count = max(len(rows), 1)
    metrics = {
        "stage": stage,
        "num_examples": len(rows),
        "valid_json_rate": valid_json / count,
        "schema_match_rate": schema_match / count,
        "exact_match_rate": exact_match / count,
        "field_accuracy": field_hits / max(total_fields, 1),
        "avg_latency_ms": sum(latencies_ms) / max(len(latencies_ms), 1),
    }

    file_name = f"predictions_{stage}.jsonl"
    write_jsonl(output_dir / file_name, prediction_rows)
    logger.info("{} metrics: {}", stage, {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    return metrics
