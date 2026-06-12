"""Inference and evaluation utilities."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lora_qlora_lab.data.emotion_dataset import EvalRecord, extract_label


def _load_model_and_tokenizer(
    model_name: str,
    hf_token: str | None,
    adapter_path: Path | None = None,
    quantized_4bit: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantized_4bit:
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=qconf,
            device_map="auto",
            token=hf_token,
        )
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, token=hf_token)
        if torch.cuda.is_available():
            base_model = base_model.to("cuda")

    if adapter_path is not None:
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
    else:
        model = base_model

    model.eval()
    return model, tokenizer


def evaluate_model(
    model_name: str,
    records: list[EvalRecord],
    hf_token: str | None,
    run_name: str,
    adapter_path: Path | None = None,
    quantized_4bit: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run deterministic generation-based evaluation and return metrics + predictions."""
    model, tokenizer = _load_model_and_tokenizer(
        model_name=model_name,
        hf_token=hf_token,
        adapter_path=adapter_path,
        quantized_4bit=quantized_4bit,
    )

    predictions: list[dict[str, Any]] = []

    for record in records:
        encoded = tokenizer(record.prompt, return_tensors="pt")
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated[0][encoded["input_ids"].shape[1] :]
        raw_completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred_label = extract_label(raw_completion)
        predictions.append(
            {
                "run_name": run_name,
                "text": record.text,
                "gold_label": record.gold_label,
                "raw_completion": raw_completion,
                "pred_label": pred_label,
            }
        )

    df = pd.DataFrame(predictions)
    accuracy = accuracy_score(df["gold_label"], df["pred_label"])
    macro_f1 = f1_score(df["gold_label"], df["pred_label"], average="macro", zero_division=0)

    metrics = {
        "run_name": run_name,
        "model_name": model_name,
        "quantized_4bit": quantized_4bit,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "n_samples": len(df),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }

    # Explicitly free memory between runs
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, df


def save_predictions(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def eval_records_to_frame(records: list[EvalRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records])
