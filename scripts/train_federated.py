#!/usr/bin/env python3
"""Standard-library federated logistic regression demo for FederatedShield AI.

Each simulated client reads its own CSV and trains locally. The coordinator receives
only model weights and training sample counts, aggregates weights with FedAvg, and
receives aggregate confusion-matrix counts for evaluation. This is a single-process
simulation, NOT an isolated or network-secured deployment, and it has NO differential
privacy yet. Those are explicit follow-up phases.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

BANKS = ("bank_a", "bank_b", "bank_c")
TXN_TYPES = ("card_payment", "online_payment", "bank_transfer", "atm_withdrawal")
CHANNELS = ("pos", "mobile", "web", "atm")

# Feature order is fixed for all clients to keep model parameters compatible.
FEATURE_NAMES = [
    "bias",
    "amount_log_scaled",
    "hour_scaled",
    "is_international",
    "account_age_scaled",
    "failed_attempts_scaled",
    "distance_scaled",
    "merchant_risk_score",
    "device_changed",
] + [f"type_{x}" for x in TXN_TYPES] + [f"channel_{x}" for x in CHANNELS]


def sigmoid(z: float) -> float:
    # Stable sigmoid avoids exp overflow for large negative/positive values.
    if z >= 0:
        e = math.exp(-min(z, 700.0))
        return 1.0 / (1.0 + e)
    e = math.exp(max(z, -700.0))
    return e / (1.0 + e)


def row_to_features(row: dict[str, str]) -> list[float]:
    amount = max(0.0, float(row["amount"]))
    hour = int(row["hour_utc"])
    age = max(0.0, float(row["account_age_days"]))
    failed = max(0.0, float(row["failed_attempts_24h"]))
    distance = max(0.0, float(row["distance_from_usual_km"]))
    merchant_risk = min(1.0, max(0.0, float(row["merchant_risk_score"])))

    features = [
        1.0,
        math.log1p(amount) / math.log1p(15000.0),
        min(23, max(0, hour)) / 23.0,
        float(int(row["is_international"])),
        min(3650.0, age) / 3650.0,
        min(5.0, failed) / 5.0,
        min(1200.0, distance) / 1200.0,
        merchant_risk,
        float(int(row["device_changed"])),
    ]
    txn_type = row["transaction_type"]
    channel = row["channel"]
    features.extend(1.0 if txn_type == category else 0.0 for category in TXN_TYPES)
    features.extend(1.0 if channel == category else 0.0 for category in CHANNELS)
    return features


def load_rows(path: Path) -> list[tuple[list[float], int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Run Phase 1 data generation first.")
    samples: list[tuple[list[float], int]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"amount", "transaction_type", "channel", "hour_utc", "is_international",
                    "account_age_days", "failed_attempts_24h", "distance_from_usual_km",
                    "merchant_risk_score", "device_changed", "fraud_label"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Unexpected columns in {path}; required columns: {sorted(required)}")
        for row in reader:
            samples.append((row_to_features(row), int(row["fraud_label"])))
    if len(samples) < 20:
        raise ValueError(f"Not enough rows in {path}: {len(samples)}")
    return samples


class SimulatedBankClient:
    """Client reads and keeps its own samples; coordinator only gets weights/counts."""

    def __init__(self, bank_name: str, data_path: Path, seed: int):
        self.bank_name = bank_name
        all_samples = load_rows(data_path)
        rng = random.Random(seed)
        rng.shuffle(all_samples)
        cut = int(len(all_samples) * 0.8)
        self.train_samples = all_samples[:cut]
        self.test_samples = all_samples[cut:]
        self.rng = random.Random(seed + 37)

    def train(self, global_weights: list[float], epochs: int, learning_rate: float,
              batch_size: int, l2: float) -> tuple[list[float], int]:
        weights = list(global_weights)
        indices = list(range(len(self.train_samples)))
        for _ in range(epochs):
            self.rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                gradients = [0.0] * len(weights)
                for idx in batch_indices:
                    x, y = self.train_samples[idx]
                    p = sigmoid(sum(w * v for w, v in zip(weights, x)))
                    err = p - y
                    for j, value in enumerate(x):
                        gradients[j] += err * value
                denom = max(1, len(batch_indices))
                for j in range(len(weights)):
                    grad = gradients[j] / denom
                    if j != 0:  # no L2 penalty on the bias
                        grad += l2 * weights[j]
                    weights[j] -= learning_rate * grad
        return weights, len(self.train_samples)

    def evaluate_counts(self, weights: list[float], threshold: float) -> dict[str, int]:
        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for x, y in self.test_samples:
            pred = int(sigmoid(sum(w * v for w, v in zip(weights, x))) >= threshold)
            if pred and y:
                counts["tp"] += 1
            elif pred and not y:
                counts["fp"] += 1
            elif not pred and y:
                counts["fn"] += 1
            else:
                counts["tn"] += 1
        return counts


def fedavg(client_updates: list[tuple[list[float], int]]) -> list[float]:
    total = sum(count for _, count in client_updates)
    if total <= 0:
        raise ValueError("Cannot aggregate updates with zero training samples")
    n_weights = len(client_updates[0][0])
    return [sum(weights[j] * count for weights, count in client_updates) / total
            for j in range(n_weights)]


def summarize_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2_den = 5 * precision + recall
    f2 = 5 * precision * recall / f2_den if f2_den else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {**counts, "total": total, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "f2": round(f2, 4), "accuracy": round(accuracy, 4)}


def sum_counts(parts: list[dict[str, int]]) -> dict[str, int]:
    return {key: sum(part[key] for part in parts) for key in ("tp", "fp", "fn", "tn")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/clients"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--threshold", type=float, default=0.07,
                        help="Fixed demo decision threshold for the sparse synthetic labels; not calibrated")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rounds < 1 or args.local_epochs < 1 or args.batch_size < 1:
        parser.error("rounds, local-epochs, and batch-size must be positive")
    if not 0.0 < args.threshold < 1.0:
        parser.error("threshold must be between 0 and 1")

    clients = [SimulatedBankClient(bank, args.data_root / bank / "transactions.csv",
                                   args.seed + (i * 1009)) for i, bank in enumerate(BANKS)]
    weights = [0.0] * len(FEATURE_NAMES)
    round_history: list[dict[str, Any]] = []
    print("FederatedShield AI — Phase 2 federated-learning prototype")
    print("Mode: single-process simulation; raw rows are handled by clients and are not passed to FedAvg.")
    print("Privacy status: differential privacy is NOT implemented in this phase.")
    print(f"Clients: {len(clients)} | features: {len(FEATURE_NAMES)} | rounds: {args.rounds}")

    start_all = time.perf_counter()
    for round_no in range(1, args.rounds + 1):
        round_start = time.perf_counter()
        client_updates: list[tuple[list[float], int]] = []
        for client in clients:
            local_weights, train_count = client.train(
                weights, args.local_epochs, args.learning_rate, args.batch_size, args.l2
            )
            # Only local model parameters and the training-set size are sent to FedAvg.
            client_updates.append((local_weights, train_count))
            print(f"  round {round_no} | {client.bank_name}: local training complete; updates sent")
        weights = fedavg(client_updates)
        round_seconds = time.perf_counter() - round_start
        round_history.append({"round": round_no, "duration_seconds": round(round_seconds, 4),
                              "clients": len(clients),
                              "training_samples_reported": sum(n for _, n in client_updates)})
        print(f"Round {round_no}/{args.rounds} aggregated in {round_seconds:.2f}s")

    # Each client evaluates locally and returns confusion-matrix counts, not test rows.
    local_counts = {client.bank_name: client.evaluate_counts(weights, args.threshold) for client in clients}
    aggregate_counts = sum_counts(list(local_counts.values()))
    metrics = summarize_counts(aggregate_counts)
    bank_metrics = {name: summarize_counts(counts) for name, counts in local_counts.items()}
    total_seconds = time.perf_counter() - start_all

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "federated_model.json"
    metrics_path = args.output_dir / "phase2_metrics.json"
    model_payload = {
        "project": "FederatedShield AI",
        "model_type": "federated_logistic_regression_demo",
        "feature_names": FEATURE_NAMES,
        "weights": [round(w, 10) for w in weights],
        "threshold": args.threshold,
        "threshold_note": "Fixed demonstration setting for this synthetic dataset; not a calibrated operational threshold.",
        "seed": args.seed,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "privacy_status": "differential_privacy_not_implemented_in_phase_2",
        "scope_note": "Synthetic-data, single-process federated-learning simulation; not production security.",
    }
    metrics_payload = {
        "project": "FederatedShield AI",
        "dataset_type": "synthetic_demo_only",
        "scope_note": "Metrics measure this synthetic demo only and do not establish real-world fraud performance.",
        "privacy_status": "No differential privacy in Phase 2.",
        "simulation_note": "Three clients run in one process; directories are not security boundaries.",
        "configuration": {"rounds": args.rounds, "local_epochs": args.local_epochs,
                          "learning_rate": args.learning_rate, "batch_size": args.batch_size,
                          "l2": args.l2, "decision_threshold": args.threshold, "seed": args.seed},
        "test_metrics_threshold": metrics,
        "per_bank_test_metrics": bank_metrics,
        "round_history": round_history,
        "total_training_seconds": round(total_seconds, 4),
        "total_test_samples": metrics["total"],
    }
    model_path.write_text(json.dumps(model_payload, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")

    print("\n=== Aggregate test metrics (fixed demo threshold) ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nTraining runtime: {total_seconds:.2f}s")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("Reminder: synthetic-demo metrics only; no differential privacy or process/network isolation yet.")


if __name__ == "__main__":
    main()
