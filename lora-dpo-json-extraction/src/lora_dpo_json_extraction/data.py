"""Dataset builders for synthetic and internet-sourced JSON extraction tasks."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
from loguru import logger

from .schemas import TicketExtraction

INTENT_PHRASES: dict[str, list[str]] = {
    "refund_request": [
        "charged twice and need a refund",
        "want my money back for an accidental payment",
        "requesting a refund after duplicate billing",
    ],
    "bug_report": [
        "the submit button crashes with a 500 error",
        "app freezes whenever I open transaction history",
        "dashboard shows blank widgets after login",
    ],
    "feature_request": [
        "please add CSV export for reports",
        "we need role-based filters in the dashboard",
        "would like scheduled email summaries",
    ],
    "account_help": [
        "cannot reset my password",
        "2FA code never arrives",
        "locked out of my account after device change",
    ],
    "security_incident": [
        "suspicious login from another country",
        "possible credential leak detected",
        "unknown API key used in my project",
    ],
}

PRODUCT_PHRASES: dict[str, list[str]] = {
    "billing_portal": ["billing portal", "invoice center", "subscription billing page"],
    "mobile_app": ["mobile app", "android app", "iOS application"],
    "api_gateway": ["API gateway", "public API", "developer API"],
    "analytics_dashboard": ["analytics dashboard", "reporting dashboard", "metrics workspace"],
}

PRIORITY_PHRASES: dict[str, list[str]] = {
    "low": [
        "this is low urgency",
        "not urgent, can be scheduled",
        "minor issue, no rush",
    ],
    "medium": [
        "this blocks part of our workflow",
        "moderate urgency for our team",
        "needs attention this week",
    ],
    "high": [
        "critical outage for production",
        "urgent and revenue-impacting",
        "P1 incident needing immediate response",
    ],
}

HUMAN_PHRASES = {
    True: [
        "Please escalate to a human specialist.",
        "Need a real support engineer on this now.",
        "Manager callback requested.",
    ],
    False: [
        "Self-serve guidance is fine.",
        "A bot response is acceptable for now.",
        "No escalation needed yet.",
    ],
}

PROMPT_TEMPLATE = """You are an information extraction assistant.
Extract the ticket into JSON only.
Schema:
{{
  "intent": string,
  "priority": "low" | "medium" | "high",
  "product": string,
  "needs_human": boolean
}}

Ticket:
{ticket}

Return JSON only, no markdown and no extra text.
JSON:
"""

BANKING77_PRODUCT_RULES: list[tuple[tuple[str, ...], str]] = [
    (("card", "cash_withdrawal", "cash_deposit", "cash"), "card_and_cash"),
    (("beneficiary", "transfer", "bank_transfer", "recipient"), "bank_transfer"),
    (("apple_pay", "cashback", "top_up", "exchange", "pending_cash_withdrawal"), "wallet_and_funding"),
    (("beneficiary", "direct_debit", "chargeback"), "billing_and_debits"),
]

BANKING77_INTENT_GROUP_RULES: list[tuple[tuple[str, ...], str]] = [
    (
        (
            "compromised",
            "not_recognised",
            "card_stolen",
            "cash_withdrawal_not_recognised",
            "beneficiary_not_allowed",
            "beneficiary_not_verified",
        ),
        "security_fraud",
    ),
    (("bank_transfer", "beneficiary", "transfer"), "payments_transfers"),
    (("cash_withdrawal", "cash_deposit", "cash"), "cash_atm"),
    (("top_up", "apple_pay", "cashback"), "topup_wallet"),
    (("card",), "card_operations"),
    (("pin", "passcode", "beneficiary", "country_support"), "account_access"),
    (("charge", "fee", "declined", "pending"), "fees_charges"),
]


@dataclass(slots=True)
class Example:
    prompt: str
    response: str
    target: dict[str, Any]


def _determine_needs_human(intent: str, priority: str, explicit_human: bool) -> bool:
    """Rule used to create consistent supervision targets."""

    if explicit_human:
        return True
    return priority == "high" or intent == "security_incident"


def _build_ticket(intent: str, priority: str, product: str, explicit_human: bool, rng: random.Random) -> str:
    intent_phrase = rng.choice(INTENT_PHRASES[intent])
    priority_phrase = rng.choice(PRIORITY_PHRASES[priority])
    product_phrase = rng.choice(PRODUCT_PHRASES[product])
    human_phrase = rng.choice(HUMAN_PHRASES[explicit_human])

    templates = [
        "Customer report from {product}: {intent}. Priority note: {priority}. {human}",
        "Issue on our {product}: {intent}. Context: {priority}. {human}",
        "Ticket says that in the {product}, user reports '{intent}'. Severity: {priority}. {human}",
    ]
    return rng.choice(templates).format(
        product=product_phrase,
        intent=intent_phrase,
        priority=priority_phrase,
        human=human_phrase,
    )


def _to_response_dict(intent: str, priority: str, product: str, needs_human: bool) -> dict[str, Any]:
    payload = {
        "intent": intent,
        "priority": priority,
        "product": product,
        "needs_human": needs_human,
    }
    validated = TicketExtraction.model_validate(payload)
    return validated.model_dump()


def build_prompt(ticket: str) -> str:
    """Render the instruction prompt used for SFT and DPO.

    Example:
        >>> prompt = build_prompt("Issue on API gateway: cannot reset password")
        >>> "JSON:" in prompt
        True
    """

    return PROMPT_TEMPLATE.format(ticket=ticket.strip())


def generate_examples(total_size: int, seed: int) -> list[Example]:
    """Generate synthetic extraction examples.

    The generation is fully deterministic for a given seed.
    """

    rng = random.Random(seed)
    intents = list(INTENT_PHRASES.keys())
    priorities = list(PRIORITY_PHRASES.keys())
    products = list(PRODUCT_PHRASES.keys())

    rows: list[Example] = []
    for _ in range(total_size):
        intent = rng.choice(intents)
        priority = rng.choice(priorities)
        product = rng.choice(products)
        explicit_human = rng.random() < 0.4
        needs_human = _determine_needs_human(intent, priority, explicit_human)

        ticket = _build_ticket(intent, priority, product, explicit_human, rng)
        prompt = build_prompt(ticket)
        target = _to_response_dict(intent, priority, product, needs_human)
        response = json.dumps(target, separators=(",", ":"), sort_keys=True)
        rows.append(Example(prompt=prompt, response=response, target=target))
    return rows


def _normalize_intent(label_name: str) -> str:
    return label_name.strip().lower().replace(" ", "_")


def _derive_product_from_banking77_intent(intent: str) -> str:
    for keywords, product in BANKING77_PRODUCT_RULES:
        if any(keyword in intent for keyword in keywords):
            return product
    return "account_services"


def _derive_intent_group_from_banking77_label(raw_intent: str) -> str:
    for keywords, intent_group in BANKING77_INTENT_GROUP_RULES:
        if any(keyword in raw_intent for keyword in keywords):
            return intent_group
    return "general_support"


def _derive_priority_from_banking77(intent_group: str, raw_intent: str, text: str) -> str:
    text_lower = text.lower()
    if intent_group == "security_fraud":
        return "high"
    if any(x in text_lower for x in ["urgent", "fraud", "stolen", "locked out", "cannot access"]):
        return "high"
    if intent_group in {"payments_transfers", "fees_charges", "cash_atm"}:
        return "medium"
    if any(x in raw_intent for x in ["declined", "pending", "charged"]):
        return "medium"
    return "low"


def _derive_needs_human_from_banking77(intent_group: str, priority: str, raw_intent: str) -> bool:
    if intent_group in {"security_fraud", "payments_transfers"}:
        return True
    if any(x in raw_intent for x in ["not_recognised", "beneficiary_not_allowed", "compromised"]):
        return True
    return priority == "high"


def _banking77_row_to_payload(text: str, label_name: str) -> dict[str, Any]:
    raw_intent = _normalize_intent(label_name)
    intent = _derive_intent_group_from_banking77_label(raw_intent)
    priority = _derive_priority_from_banking77(intent_group=intent, raw_intent=raw_intent, text=text)
    product = _derive_product_from_banking77_intent(raw_intent)
    needs_human = _derive_needs_human_from_banking77(intent_group=intent, priority=priority, raw_intent=raw_intent)
    target = _to_response_dict(intent=intent, priority=priority, product=product, needs_human=needs_human)
    prompt = build_prompt(text)
    response = json.dumps(target, separators=(",", ":"), sort_keys=True)
    return {"prompt": prompt, "response": response, "target": target}


def build_banking77_splits(
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    dataset_name: str,
    train_split: str,
    test_split: str,
) -> dict[str, list[dict[str, Any]]]:
    """Build train/val/test using internet-sourced Banking77 data from HF."""

    logger.info("Loading internet dataset {} (splits: {}, {})", dataset_name, train_split, test_split)
    dataset = load_dataset(dataset_name)

    if train_split not in dataset or test_split not in dataset:
        available = list(dataset.keys())
        raise ValueError(f"Requested splits not found. available={available}")

    train_ds = dataset[train_split]
    test_ds = dataset[test_split]

    required_train = train_size + val_size
    if required_train > len(train_ds):
        raise ValueError(f"train_size+val_size={required_train} exceeds {len(train_ds)} rows")
    if test_size > len(test_ds):
        raise ValueError(f"test_size={test_size} exceeds {len(test_ds)} rows")

    label_feature = train_ds.features["label"]
    label_names = getattr(label_feature, "names", None)
    if not label_names:
        raise ValueError("Expected class-label names in train split for label column")

    rng = random.Random(seed)
    train_val_indices = rng.sample(range(len(train_ds)), required_train)
    train_indices = train_val_indices[:train_size]
    val_indices = train_val_indices[train_size:]
    test_indices = rng.sample(range(len(test_ds)), test_size)

    def _convert(ds, indices: list[int]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx in indices:
            item = ds[int(idx)]
            text = str(item["text"])
            label_id = int(item["label"])
            label_name = str(label_names[label_id])
            rows.append(_banking77_row_to_payload(text=text, label_name=label_name))
        return rows

    return {
        "train": _convert(train_ds, train_indices),
        "val": _convert(train_ds, val_indices),
        "test": _convert(test_ds, test_indices),
    }


def _mutate_field_value(current_value: Any, candidates: list[Any], rng: random.Random) -> Any:
    alternatives = [value for value in candidates if value != current_value]
    if alternatives:
        return rng.choice(alternatives)
    if isinstance(current_value, bool):
        return not current_value
    if isinstance(current_value, str):
        return f"{current_value}_alt"
    return current_value


def _build_rejected(
    target: dict[str, Any],
    rng: random.Random,
    field_values: dict[str, list[Any]],
) -> str:
    """Create an intentionally weaker output for DPO preference learning."""

    corrupted = dict(target)
    mutation = rng.choice(["wrong_field", "missing_field", "extra_text"])

    if mutation == "wrong_field":
        field = rng.choice(["intent", "priority", "product", "needs_human"])
        corrupted[field] = _mutate_field_value(
            current_value=target.get(field),
            candidates=field_values.get(field, []),
            rng=rng,
        )
        return json.dumps(corrupted, separators=(",", ":"), sort_keys=True)

    if mutation == "missing_field":
        field = rng.choice(["intent", "priority", "product", "needs_human"])
        corrupted.pop(field, None)
        return json.dumps(corrupted, separators=(",", ":"), sort_keys=True)

    return (
        "Sure, here is the JSON you asked for: "
        + json.dumps(corrupted, separators=(",", ":"), sort_keys=True)
    )


def build_splits(
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    source: str = "synthetic",
    hf_dataset: str = "PolyAI/banking77",
    hf_train_split: str = "train",
    hf_test_split: str = "test",
) -> dict[str, list[dict[str, Any]]]:
    """Build train/val/test splits for SFT and evaluation."""

    source_key = source.strip().lower()

    if source_key == "synthetic":
        total = train_size + val_size + test_size
        examples = generate_examples(total_size=total, seed=seed)
        payload = [
            {"prompt": ex.prompt, "response": ex.response, "target": ex.target}
            for ex in examples
        ]
        return {
            "train": payload[:train_size],
            "val": payload[train_size : train_size + val_size],
            "test": payload[train_size + val_size :],
        }

    if source_key in {"banking77", "internet_banking77", "hf_banking77"}:
        return build_banking77_splits(
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            dataset_name=hf_dataset,
            train_split=hf_train_split,
            test_split=hf_test_split,
        )

    raise ValueError(f"Unsupported data source: {source}")


def build_preference_split(train_rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Create preference pairs from SFT train split."""

    rng = random.Random(seed + 101)

    field_values: dict[str, list[Any]] = {
        "intent": sorted({row["target"]["intent"] for row in train_rows}),
        "priority": sorted({row["target"]["priority"] for row in train_rows}),
        "product": sorted({row["target"]["product"] for row in train_rows}),
        "needs_human": sorted({row["target"]["needs_human"] for row in train_rows}),
    }

    prefs: list[dict[str, Any]] = []
    for row in train_rows:
        chosen = row["response"]
        rejected = _build_rejected(row["target"], rng, field_values=field_values)
        prefs.append(
            {
                "prompt": row["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "target": row["target"],
            }
        )
    return prefs
