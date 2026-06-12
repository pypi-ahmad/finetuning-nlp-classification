"""
routing_taxonomy.py
─────────────────────────────────────────────────────────────────────────────
Maps the 77 fine-grained Banking77 *intents* onto 8 coarse, realistic
*support-ticket routing queues*.

Why remap?
    Banking77 is an intent-detection benchmark with 77 narrow labels. A real
    triage / routing system does not need 77 destinations — it needs to send a
    ticket to the *team* that can resolve it. Collapsing 77 intents into 8
    operational queues turns an academic benchmark into a measurable routing
    task where:
      * Top-3 accuracy is meaningful (3 of 8, not 3 of 77).
      * A confusion matrix is human-readable.
      * "Misrouting" maps to a concrete operational cost (ticket bounces to the
        wrong team).

The mapping below is an explicit, auditable dictionary (no magic). Every one of
the 77 intents is listed exactly once. `assert_full_coverage()` is called by the
notebook against the live dataset so a future Banking77 revision that adds /
renames an intent fails loudly instead of silently dropping data.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# The 8 routing queues (stable order → stable label ids)
# ─────────────────────────────────────────────────────────────────────────────
ROUTES: list[str] = [
    "Card Disputes & Fraud",
    "Payments & Transfers",
    "Card Operations",
    "ATM & Cash",
    "Account Access & Security",
    "Top-Up & Funding",
    "Identity & Verification",
    "Fees, Rates & Product Info",
]

# Short operational definitions — injected into the prompt so the model (and a
# human reader) knows exactly what each queue owns.
ROUTE_DESCRIPTIONS: dict[str, str] = {
    "Card Disputes & Fraud":
        "Unrecognised / duplicate / reverted charges, refunds, compromised or "
        "lost/stolen cards.",
    "Payments & Transfers":
        "Bank transfers and direct debits: timing, fees, failed/declined/pending "
        "transfers, beneficiaries, incoming money.",
    "Card Operations":
        "Ordering, delivery, activation, linking and physical malfunction of "
        "physical/virtual cards (contactless, expiry, declined card use).",
    "ATM & Cash":
        "Cash withdrawals and ATMs: charges, declines, wrong amount dispensed, "
        "swallowed cards, withdrawal exchange rates.",
    "Account Access & Security":
        "PIN/passcode, personal-detail edits, lost phone, account closure.",
    "Top-Up & Funding":
        "Adding money to the account by card/bank/cash/cheque: fees, limits, "
        "failures, reversals, automatic top-ups.",
    "Identity & Verification":
        "KYC: identity / source-of-funds verification, why & how to verify, age "
        "eligibility.",
    "Fees, Rates & Product Info":
        "General product questions: exchange rates/charges, supported cards & "
        "currencies, country support, card acceptance, digital wallets.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Explicit intent → route map (keys lower-cased for robustness against the
# capitalised "Refund_not_showing_up" label and any future case drift).
# ─────────────────────────────────────────────────────────────────────────────
INTENT_TO_ROUTE: dict[str, str] = {
    # ── Card Disputes & Fraud ────────────────────────────────────────────────
    "card_payment_not_recognised":          "Card Disputes & Fraud",
    "cash_withdrawal_not_recognised":       "Card Disputes & Fraud",
    "direct_debit_payment_not_recognised":  "Card Disputes & Fraud",
    "compromised_card":                     "Card Disputes & Fraud",
    "lost_or_stolen_card":                  "Card Disputes & Fraud",
    "transaction_charged_twice":            "Card Disputes & Fraud",
    "extra_charge_on_statement":            "Card Disputes & Fraud",
    "reverted_card_payment?":               "Card Disputes & Fraud",
    "refund_not_showing_up":                "Card Disputes & Fraud",
    "request_refund":                       "Card Disputes & Fraud",

    # ── Payments & Transfers ─────────────────────────────────────────────────
    "balance_not_updated_after_bank_transfer": "Payments & Transfers",
    "beneficiary_not_allowed":              "Payments & Transfers",
    "cancel_transfer":                      "Payments & Transfers",
    "declined_transfer":                    "Payments & Transfers",
    "failed_transfer":                      "Payments & Transfers",
    "pending_transfer":                     "Payments & Transfers",
    "receiving_money":                      "Payments & Transfers",
    "transfer_fee_charged":                 "Payments & Transfers",
    "transfer_into_account":                "Payments & Transfers",
    "transfer_not_received_by_recipient":   "Payments & Transfers",
    "transfer_timing":                      "Payments & Transfers",

    # ── Card Operations ──────────────────────────────────────────────────────
    "activate_my_card":                     "Card Operations",
    "apple_pay_or_google_pay":              "Card Operations",
    "card_about_to_expire":                 "Card Operations",
    "card_arrival":                         "Card Operations",
    "card_delivery_estimate":               "Card Operations",
    "card_linking":                         "Card Operations",
    "card_not_working":                     "Card Operations",
    "contactless_not_working":              "Card Operations",
    "declined_card_payment":                "Card Operations",
    "get_disposable_virtual_card":          "Card Operations",
    "get_physical_card":                    "Card Operations",
    "getting_spare_card":                   "Card Operations",
    "getting_virtual_card":                 "Card Operations",
    "order_physical_card":                  "Card Operations",
    "pending_card_payment":                 "Card Operations",
    "virtual_card_not_working":             "Card Operations",

    # ── ATM & Cash ───────────────────────────────────────────────────────────
    "atm_support":                          "ATM & Cash",
    "card_swallowed":                       "ATM & Cash",
    "cash_withdrawal_charge":               "ATM & Cash",
    "declined_cash_withdrawal":             "ATM & Cash",
    "pending_cash_withdrawal":              "ATM & Cash",
    "wrong_amount_of_cash_received":        "ATM & Cash",
    "wrong_exchange_rate_for_cash_withdrawal": "ATM & Cash",

    # ── Account Access & Security ────────────────────────────────────────────
    "change_pin":                           "Account Access & Security",
    "edit_personal_details":                "Account Access & Security",
    "lost_or_stolen_phone":                 "Account Access & Security",
    "passcode_forgotten":                   "Account Access & Security",
    "pin_blocked":                          "Account Access & Security",
    "terminate_account":                    "Account Access & Security",

    # ── Top-Up & Funding ─────────────────────────────────────────────────────
    "automatic_top_up":                     "Top-Up & Funding",
    "balance_not_updated_after_cheque_or_cash_deposit": "Top-Up & Funding",
    "pending_top_up":                       "Top-Up & Funding",
    "top_up_by_bank_transfer_charge":       "Top-Up & Funding",
    "top_up_by_card_charge":                "Top-Up & Funding",
    "top_up_by_cash_or_cheque":             "Top-Up & Funding",
    "top_up_failed":                        "Top-Up & Funding",
    "top_up_limits":                        "Top-Up & Funding",
    "top_up_reverted":                      "Top-Up & Funding",
    "topping_up_by_card":                   "Top-Up & Funding",

    # ── Identity & Verification ──────────────────────────────────────────────
    "age_limit":                            "Identity & Verification",
    "unable_to_verify_identity":            "Identity & Verification",
    "verify_my_identity":                   "Identity & Verification",
    "verify_source_of_funds":               "Identity & Verification",
    "verify_top_up":                        "Identity & Verification",
    "why_verify_identity":                  "Identity & Verification",

    # ── Fees, Rates & Product Info ───────────────────────────────────────────
    "card_acceptance":                      "Fees, Rates & Product Info",
    "card_payment_fee_charged":             "Fees, Rates & Product Info",
    "card_payment_wrong_exchange_rate":     "Fees, Rates & Product Info",
    "country_support":                      "Fees, Rates & Product Info",
    "disposable_card_limits":               "Fees, Rates & Product Info",
    "exchange_charge":                      "Fees, Rates & Product Info",
    "exchange_rate":                        "Fees, Rates & Product Info",
    "exchange_via_app":                     "Fees, Rates & Product Info",
    "fiat_currency_support":                "Fees, Rates & Product Info",
    "supported_cards_and_currencies":       "Fees, Rates & Product Info",
    "visa_or_mastercard":                   "Fees, Rates & Product Info",
}

# ─────────────────────────────────────────────────────────────────────────────
# Stable id <-> label maps for the routing task
# ─────────────────────────────────────────────────────────────────────────────
ROUTE2ID: dict[str, int] = {r: i for i, r in enumerate(ROUTES)}
ID2ROUTE: dict[int, str] = {i: r for i, r in enumerate(ROUTES)}


def route_for_intent(intent_name: str) -> str:
    """Return the routing queue for a Banking77 intent name (case-insensitive)."""
    try:
        return INTENT_TO_ROUTE[intent_name.lower()]
    except KeyError as exc:
        raise KeyError(
            f"Banking77 intent {intent_name!r} is not mapped in INTENT_TO_ROUTE. "
            "Update routing_taxonomy.py to cover it."
        ) from exc


def assert_full_coverage(banking77_intent_names: list[str]) -> None:
    """
    Fail loudly if the live dataset and our mapping disagree. Catches a renamed
    or newly-added Banking77 intent before it silently corrupts the labels.
    """
    live = {n.lower() for n in banking77_intent_names}
    mapped = set(INTENT_TO_ROUTE)
    missing = live - mapped          # in dataset, not mapped
    extra = mapped - live            # mapped, not in dataset
    if missing or extra:
        raise AssertionError(
            f"Routing taxonomy out of sync with dataset.\n"
            f"  Unmapped intents (in data, not in map): {sorted(missing)}\n"
            f"  Stale mappings (in map, not in data):   {sorted(extra)}"
        )


if __name__ == "__main__":
    # Quick self-check / summary you can run standalone.
    from collections import Counter

    assert len(INTENT_TO_ROUTE) == 77, f"expected 77 intents, got {len(INTENT_TO_ROUTE)}"
    assert set(INTENT_TO_ROUTE.values()) == set(ROUTES), "route name mismatch"
    counts = Counter(INTENT_TO_ROUTE.values())
    print(f"{len(INTENT_TO_ROUTE)} intents -> {len(ROUTES)} routes\n")
    for r in ROUTES:
        print(f"  {counts[r]:2d} intents  |  {r}")
