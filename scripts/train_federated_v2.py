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
    "large_amount_ge_1500",
    "hour_scaled",
    "hour_le_4",
    "is_international",
    "account_age_scaled",
    "new_account_lt_30d",
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
        float(amount >= 1500.0),
        min(23, max(0, hour)) / 23.0,
        float(hour <= 4),
        float(int(row["is_international"])),
        min(3650.0, age) / 3650.0,
        float(age < 30.0),
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
        train_end = int(len(all_samples) * 0.6)
        val_end = int(len(all_samples) * 0.8)
        self.train_samples = all_samples[:train_end]
        self.validation_samples = all_samples[train_end:val_end]
        self.test_samples = all_samples[val_end:]
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

    def evaluate_samples(self, samples: list[tuple[list[float], int]], weights: list[float], threshold: float) -> dict[str, int]:
        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for x, y in samples:
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
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    return {**counts, "total": total, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "f2": round(f2, 4), "accuracy": round(accuracy, 4),
            "specificity": round(specificity, 4), "false_positive_rate": round(false_positive_rate, 4),
            "balanced_accuracy": round(balanced_accuracy, 4)}


def sum_counts(parts: list[dict[str, int]]) -> dict[str, int]:
    return {key: sum(part[key] for part in parts) for key in ("tp", "fp", "fn", "tn")}


def evaluate_ranking_metrics(samples: list[tuple[list[float], int]], weights: list[float]) -> dict[str, float | int]:
    """Compute client-local ROC-AUC and average precision from prediction scores."""
    scored = [(sigmoid(sum(w * v for w, v in zip(weights, x))), y) for x, y in samples]
    positives = sum(y for _, y in scored)
    negatives = len(scored) - positives
    if positives == 0 or negatives == 0:
        return {"roc_auc": 0.0, "average_precision": 0.0,
                "positive_count": positives, "negative_count": negatives}

    # Rank-sum ROC-AUC with average ranks for tied scores.
    ascending = sorted(scored, key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    i = 0
    while i < len(ascending):
        j = i + 1
        while j < len(ascending) and ascending[j][0] == ascending[i][0]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0  # ranks are 1-indexed; j is exclusive
        positive_rank_sum += sum(label for _, label in ascending[i:j]) * average_rank
        i = j
    roc_auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)

    # Average precision uses the precision at each unique-score threshold.
    descending = sorted(scored, key=lambda pair: pair[0], reverse=True)
    tp = fp = 0
    ap = 0.0
    i = 0
    while i < len(descending):
        j = i + 1
        while j < len(descending) and descending[j][0] == descending[i][0]:
            j += 1
        group_pos = sum(label for _, label in descending[i:j])
        group_neg = (j - i) - group_pos
        tp += group_pos
        fp += group_neg
        ap += (group_pos / positives) * (tp / (tp + fp))
        i = j
    return {"roc_auc": round(roc_auc, 4), "average_precision": round(ap, 4),
            "positive_count": positives, "negative_count": negatives}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/clients"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional fixed threshold; mutually exclusive with --min-precision")
    parser.add_argument("--min-precision", type=float, default=None,
                        help="Select the threshold on validation data to maximize recall subject to this minimum precision")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rounds < 1 or args.local_epochs < 1 or args.batch_size < 1:
        parser.error("rounds, local-epochs, and batch-size must be positive")
    if args.threshold is not None and not 0.0 < args.threshold < 1.0:
        parser.error("threshold must be between 0 and 1")
    if args.min_precision is not None and not 0.0 < args.min_precision < 1.0:
        parser.error("min-precision must be between 0 and 1")
    if args.threshold is not None and args.min_precision is not None:
        parser.error("--threshold and --min-precision are mutually exclusive")

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

    # Select operating threshold on validation data only; test data remains untouched.
    validation_summary = None
    candidates = [round(i / 100, 2) for i in range(1, 81)]
    if args.threshold is None and args.min_precision is not None:
        feasible = []
        for candidate in candidates:
            counts = sum_counts([client.evaluate_samples(client.validation_samples, weights, candidate) for client in clients])
            summary = summarize_counts(counts)
            if summary["precision"] >= args.min_precision:
                # Among thresholds meeting the precision floor, maximize recall; then F2, then precision.
                rank = (summary["recall"], summary["f2"], summary["precision"])
                feasible.append((rank, candidate, summary))
        if not feasible:
            parser.error("No validation threshold met --min-precision; lower the precision target or retrain.")
        best = max(feasible, key=lambda item: item[0])
        threshold = best[1]
        validation_summary = {"threshold": threshold, "metrics": best[2],
                              "selection_rule": "maximum validation recall subject to precision >= min_precision; tie-break F2 then precision",
                              "min_precision": args.min_precision}
    elif args.threshold is None:
        best = None
        for candidate in candidates:
            counts = sum_counts([client.evaluate_samples(client.validation_samples, weights, candidate) for client in clients])
            summary = summarize_counts(counts)
            # Optimize F2 because missing a fraud case is considered more costly in this demo.
            rank = (summary["f2"], summary["recall"], summary["precision"])
            if best is None or rank > best[0]:
                best = (rank, candidate, summary)
        assert best is not None
        threshold = best[1]
        validation_summary = {"threshold": threshold, "metrics": best[2], "selection_rule": "maximum validation F2; tie-break recall then precision"}
    else:
        threshold = args.threshold
        validation_summary = {"threshold": threshold, "selection_rule": "fixed user-specified threshold; not selected internally on validation"}

    # Evaluate on holdout test rows only after the threshold has been selected.
    local_counts = {client.bank_name: client.evaluate_samples(client.test_samples, weights, threshold) for client in clients}
    aggregate_counts = sum_counts(list(local_counts.values()))
    metrics = summarize_counts(aggregate_counts)
    bank_metrics = {name: summarize_counts(counts) for name, counts in local_counts.items()}
    # Ranking metrics are calculated locally at each client, then only the metrics are summarized.
    bank_ranking_metrics = {client.bank_name: evaluate_ranking_metrics(client.test_samples, weights) for client in clients}
    ranking_clients = list(bank_ranking_metrics.values())
    macro_ranking_metrics = {
        "roc_auc_macro_average": round(sum(m["roc_auc"] for m in ranking_clients) / len(ranking_clients), 4),
        "average_precision_macro_average": round(sum(m["average_precision"] for m in ranking_clients) / len(ranking_clients), 4),
        "note": "Unweighted average of client-local metrics; not the pooled global metric.",
    }
    total_seconds = time.perf_counter() - start_all

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "federated_model.json"
    metrics_path = args.output_dir / "phase2_metrics.json"
    model_payload = {
        "project": "FederatedShield AI",
        "model_type": "federated_logistic_regression_demo",
        "feature_names": FEATURE_NAMES,
        "weights": [round(w, 10) for w in weights],
        "threshold": threshold,
        "threshold_note": "Selected on validation data by F2 unless explicitly provided; synthetic demo only, not an operationally calibrated threshold.",
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
                          "l2": args.l2, "decision_threshold": threshold, "min_precision_target": args.min_precision,
                          "threshold_policy": "min_precision" if args.min_precision is not None else ("fixed" if args.threshold is not None else "max_f2"),
                          "seed": args.seed,
                          "split": "60% train / 20% validation / 20% test per bank"},
        "validation_threshold_selection": validation_summary,
        "test_metrics_threshold": metrics,
        "per_bank_test_metrics": bank_metrics,
        "per_bank_test_ranking_metrics": bank_ranking_metrics,
        "macro_average_test_ranking_metrics": macro_ranking_metrics,
        "round_history": round_history,
        "total_training_seconds": round(total_seconds, 4),
        "total_test_samples": metrics["total"],
    }
    model_path.write_text(json.dumps(model_payload, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")

    print(f"\nValidation-selected threshold: {threshold:.2f}")
    if validation_summary and "metrics" in validation_summary:
        print("Validation metrics:")
        print(json.dumps(validation_summary["metrics"], indent=2))
    print("\n=== Aggregate held-out test metrics ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nTraining runtime: {total_seconds:.2f}s")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("Reminder: synthetic-demo metrics only; no differential privacy or process/network isolation yet.")


if __name__ == "__main__":
    main()
