#!/usr/bin/env python
"""
download_cfpb.py
─────────────────────────────────────────────────────────────────────────────
Pull a *curated, class-balanced* subset of the CFPB Consumer Complaint Database
via the public search API (no auth, no multi-GB bulk download).

    https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/

Why the API and not the bulk CSV?
    The bulk complaints.csv.zip is ~hundreds of MB zipped / several GB unzipped
    and ~3.8M rows, ~90% of which are credit-reporting. For a *local, 8 GB,
    learning-focused* triage project we instead fetch a few thousand
    narrative-bearing complaints **per canonical product**, which gives a
    balanced hierarchy without drowning in one class. The bulk command is still
    documented in the README for anyone who wants the full set.

Output: data/raw/cfpb/cfpb_subset.parquet  with columns
    product, issue, sub_issue, complaint_what_happened, date_received, company

Run:
    uv run python download_cfpb.py --per-product 2000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "raw" / "cfpb" / "cfpb_subset.parquet"

# Canonical L1 products → the historical raw `product` strings that map to each.
# (CFPB renamed several products over the years; we consolidate them.)
CANONICAL_PRODUCTS: dict[str, list[str]] = {
    "Credit reporting & repair": [
        "Credit reporting or other personal consumer reports",
        "Credit reporting, credit repair services, or other personal consumer reports",
        "Credit reporting",
    ],
    "Debt collection": ["Debt collection"],
    "Mortgage": ["Mortgage"],
    "Credit card / prepaid": [
        "Credit card", "Credit card or prepaid card", "Prepaid card",
    ],
    "Bank account / savings": [
        "Checking or savings account", "Bank account or service",
    ],
    "Money transfer & virtual currency": [
        "Money transfer, virtual currency, or money service", "Money transfers",
    ],
    "Student loan": ["Student loan"],
    "Vehicle / consumer loan": ["Vehicle loan or lease", "Consumer Loan"],
    "Payday / title / personal loan": [
        "Payday loan, title loan, or personal loan",
        "Payday loan, title loan, personal loan, or advance loan",
        "Payday loan",
    ],
}

FIELDS = ["product", "issue", "sub_issue", "complaint_what_happened",
          "date_received", "company"]


def fetch_product(raw_products: list[str], target: int, page: int = 1000) -> list[dict]:
    """Fetch up to `target` narrative complaints for a list of raw product names."""
    rows: list[dict] = []
    for raw in raw_products:
        frm = 0
        while len(rows) < target and frm < 9000:  # stay within ES from+size window
            params = [
                ("product", raw), ("has_narrative", "true"),
                ("size", min(page, target - len(rows))), ("from", frm),
                ("sort", "created_date_desc"), ("no_aggs", "true"),
            ] + [("field", f) for f in FIELDS]
            try:
                r = requests.get(API, params=params, timeout=90)
                r.raise_for_status()
                hits = r.json().get("hits", {}).get("hits", [])
            except Exception as e:  # transient network / rate limit → back off once
                print(f"    ! {raw}: {e}; retrying in 5s")
                time.sleep(5)
                continue
            if not hits:
                break
            for h in hits:
                src = h.get("_source", {})
                rows.append({k: src.get(k) for k in FIELDS})
            frm += len(hits)
            time.sleep(0.3)  # be polite to the public API
            if len(hits) < params[2][1]:  # last page
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-product", type=int, default=2000,
                    help="target narrative complaints per canonical L1 product")
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for canon, raws in CANONICAL_PRODUCTS.items():
        print(f"• {canon} ...", flush=True)
        rows = fetch_product(raws, args.per_product)
        for row in rows:
            row["product_l1"] = canon
        all_rows.extend(rows)
        print(f"    got {len(rows)}")

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["complaint_what_happened"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"\nSaved {len(df)} complaints → {OUT}")
    print(df["product_l1"].value_counts().to_string())


if __name__ == "__main__":
    main()
