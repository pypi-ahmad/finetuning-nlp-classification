from __future__ import annotations

from lora_qlora_lab.data.emotion_dataset import build_prompt, build_train_text, extract_label


def test_prompt_and_train_text() -> None:
    prompt = build_prompt("I am very happy today")
    train_text = build_train_text("I am very happy today", "joy")

    assert "Label:" in prompt
    assert train_text.endswith("joy")


def test_extract_label() -> None:
    assert extract_label("The correct label is fear") == "fear"
    assert extract_label("unknown answer") == "unknown"
