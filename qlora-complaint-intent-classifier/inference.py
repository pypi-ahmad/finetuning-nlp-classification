"""
inference.py — load the saved LoRA adapter and classify banking queries.

Usage:
    uv run python inference.py                      # interactive prompt
    uv run python inference.py "I lost my card"     # single query
    uv run python inference.py --batch queries.txt  # one query per line
"""

import argparse
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR   = "./qlora-banking77-adapter"
MAX_NEW_TOKENS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_MSG = (
    "You are a banking intent classifier. "
    "Given a customer query, respond with exactly one intent label using "
    "lowercase letters and underscores. Output only the label, nothing else."
)

# Banking77 label set (77 intents, alphabetical order matching HF dataset)
LABEL_NAMES = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
    "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge",
    "exchange_rate", "exchange_via_app", "extra_charge_on_statement", "failed_transfer",
    "fiat_currency_support", "get_disposable_virtual_card", "get_physical_card",
    "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card",
    "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
    "pending_card_payment", "pending_cash_withdrawal", "pending_top_up",
    "pending_transfer", "pin_blocked", "receiving_money", "refund_not_showing_up",
    "request_refund", "reverted_card_payment?", "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
    "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


def load_model(adapter_dir: str = ADAPTER_DIR):
    print(f"Loading base model: {BASE_MODEL_ID}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"Loading adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model ready.\n")
    return model, tokenizer


def _normalise(text: str) -> str:
    n = re.sub(r"[^a-z0-9_]", "_", text.strip().lower())
    return re.sub(r"_+", "_", n).strip("_")


def predict(text: str, model, tokenizer) -> str:
    messages = [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    norm = _normalise(generated)

    if norm in LABEL_NAMES:
        return norm
    for label in LABEL_NAMES:
        if norm.startswith(label):
            return label
    best, best_score = LABEL_NAMES[0], -1
    for label in LABEL_NAMES:
        score = sum(c in norm for c in label.replace("_", ""))
        if score > best_score:
            best, best_score = label, score
    return best


def main():
    parser = argparse.ArgumentParser(description="Banking intent classifier (QLoRA adapter)")
    parser.add_argument("query", nargs="?", help="Single query string")
    parser.add_argument("--batch", metavar="FILE", help="Text file with one query per line")
    parser.add_argument("--adapter", default=ADAPTER_DIR, help="Path to saved LoRA adapter")
    args = parser.parse_args()

    if not Path(args.adapter).exists():
        print(f"Adapter not found at '{args.adapter}'.")
        print("Run the notebook first to fine-tune and save the adapter.")
        sys.exit(1)

    model, tokenizer = load_model(args.adapter)

    if args.batch:
        queries = Path(args.batch).read_text().splitlines()
        queries = [q.strip() for q in queries if q.strip()]
        for q in queries:
            label = predict(q, model, tokenizer)
            print(f"{label}\t{q}")
        return

    if args.query:
        label = predict(args.query, model, tokenizer)
        print(f"Query  : {args.query}")
        print(f"Intent : {label}")
        return

    # Interactive mode
    print("Banking Intent Classifier — interactive mode  (Ctrl+C to quit)")
    print("─" * 60)
    while True:
        try:
            query = input("\nQuery> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not query:
            continue
        print(f"Intent: {predict(query, model, tokenizer)}")


if __name__ == "__main__":
    main()
