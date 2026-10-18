#!/usr/bin/env python3
"""Generate separate, fully synthetic transaction CSVs for three demo banks.

No real customer records are used. Python standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

BANKS = ("bank_a", "bank_b", "bank_c")
FIELDNAMES = [
    "transaction_id",
    "timestamp_utc",
    "amount",
    "transaction_type",
    "channel",
    "hour_utc",
    "is_international",
    "account_age_days",
    "failed_attempts_24h",
    "distance_from_usual_km",
    "merchant_risk_score",
    "device_changed",
    "fraud_label",
]


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def generate_row(rng: random.Random, bank: str, row_num: int, bank_profile: dict) -> dict:
    amount = min(15000.0, rng.lognormvariate(bank_profile["amount_mu"], 1.0))
    txn_type = rng.choices(
        ["card_payment", "online_payment", "bank_transfer", "atm_withdrawal"],
        weights=bank_profile["type_weights"], k=1
    )[0]
    channel = rng.choices(
        ["pos", "mobile", "web", "atm"],
        weights=bank_profile["channel_weights"], k=1
    )[0]
    hour = rng.randrange(24)
    is_international = int(rng.random() < bank_profile["international_rate"])
    account_age_days = min(3650, int(rng.expovariate(1 / bank_profile["account_age_mean"])))
    failed_attempts = min(5, int(rng.choices([0, 1, 2, 3, 4, 5], [0.76, 0.13, 0.06, 0.025, 0.015, 0.01])[0]))
    distance = min(1200.0, rng.expovariate(1 / bank_profile["distance_mean_km"]))
    merchant_risk = round(rng.betavariate(2, 4), 4)
    device_changed = int(rng.random() < bank_profile["device_change_rate"])

    # Synthetic target only: a probabilistic rule simulates plausible fraud patterns.
    # It is not a model of real banking fraud and must be described as synthetic data.
    score = (
        -4.05
        + 0.75 * is_international
        + 0.75 * device_changed
        + 0.52 * failed_attempts
        + 1.35 * merchant_risk
        + 0.0028 * distance
        + 0.38 * int(amount >= 1500)
        + 0.55 * int(account_age_days < 30)
        + 0.22 * int(channel in ("mobile", "web"))
        + 0.20 * int(hour <= 4)
    )
    fraud_label = int(rng.random() < sigmoid(score))

    # Deterministic date range makes runs reproducible and avoids actual personal data.
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamp = start + timedelta(minutes=rng.randrange(365 * 24 * 60))
    return {
        "transaction_id": f"{bank}_{row_num:07d}",
        "timestamp_utc": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "amount": f"{amount:.2f}",
        "transaction_type": txn_type,
        "channel": channel,
        "hour_utc": hour,
        "is_international": is_international,
        "account_age_days": account_age_days,
        "failed_attempts_24h": failed_attempts,
        "distance_from_usual_km": f"{distance:.2f}",
        "merchant_risk_score": f"{merchant_risk:.4f}",
        "device_changed": device_changed,
        "fraud_label": fraud_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-bank", type=int, default=5000, help="Rows to generate per simulated bank (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/clients"), help="Root directory for local client datasets")
    args = parser.parse_args()
    if args.rows_per_bank < 100:
        parser.error("--rows-per-bank must be at least 100")

    profiles = {
        "bank_a": {"amount_mu": 4.25, "type_weights": [0.42, 0.30, 0.20, 0.08], "channel_weights": [0.38, 0.30, 0.24, 0.08], "international_rate": 0.07, "account_age_mean": 650, "distance_mean_km": 32, "device_change_rate": 0.07},
        "bank_b": {"amount_mu": 4.05, "type_weights": [0.34, 0.28, 0.27, 0.11], "channel_weights": [0.27, 0.34, 0.28, 0.11], "international_rate": 0.11, "account_age_mean": 520, "distance_mean_km": 42, "device_change_rate": 0.09},
        "bank_c": {"amount_mu": 4.45, "type_weights": [0.38, 0.24, 0.26, 0.12], "channel_weights": [0.30, 0.26, 0.30, 0.14], "international_rate": 0.05, "account_age_mean": 800, "distance_mean_km": 26, "device_change_rate": 0.05},
    }

    summary = {"dataset_type": "synthetic_demo_only", "seed": args.seed, "rows_per_bank": args.rows_per_bank, "banks": {}}
    for bank_index, bank in enumerate(BANKS):
        rng = random.Random(args.seed + bank_index * 1009)
        bank_dir = args.output_dir / bank
        bank_dir.mkdir(parents=True, exist_ok=True)
        out_path = bank_dir / "transactions.csv"
        fraud_count = 0
        with out_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row_num in range(1, args.rows_per_bank + 1):
                row = generate_row(rng, bank, row_num, profiles[bank])
                fraud_count += row["fraud_label"]
                writer.writerow(row)
        summary["banks"][bank] = {
            "file": str(out_path),
            "rows": args.rows_per_bank,
            "fraud_rows": fraud_count,
            "fraud_rate": round(fraud_count / args.rows_per_bank, 5),
        }
        print(f"Created {out_path} | rows={args.rows_per_bank} | synthetic_fraud_rate={fraud_count / args.rows_per_bank:.2%}")

    metadata_dir = args.output_dir.parent / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metadata_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote summary: {summary_path}")
    print("NOTE: These are synthetic records generated for prototyping, not real transaction data.")


if __name__ == "__main__":
    main()
