#!/usr/bin/env python3
"""FederatedShield AI Phase 3: full-batch DP-SGD local training + FedAvg.

This is an educational/synthetic-data prototype. It uses record-level clipping
and Gaussian noise for each local full-batch gradient step. The accountant
covers training-record influence on released model parameters under a
replace-one-record adjacency assumption, public fixed client training-set sizes,
public hyperparameters, and the implemented Gaussian mechanism. It does NOT
privatize validation/test evaluation outputs, threshold selection, row counts,
client metadata, or external logs. It is not production banking security.
"""
from __future__ import annotations
import argparse
import json
import math
import secrets
import sys
import time
from pathlib import Path
from typing import Any

# Run from project root. This module is the Phase 2.1 script already installed
# in scripts/ and provides the feature processing/client splits/metrics.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from train_federated import (  # type: ignore
    BANKS, FEATURE_NAMES, sigmoid, fedavg, summarize_counts, sum_counts, load_rows,
)


class Phase3BankClient:
    """Phase-3 client with separate train/validation/test splits.

    This wrapper intentionally avoids depending on Phase 2 client methods, whose
    API only exposes an 80/20 train/test split and evaluate_counts().
    """
    def __init__(self, bank_name: str, data_path: Path, seed: int):
        import random
        self.bank_name = bank_name
        all_samples = load_rows(data_path)
        split_rng = random.Random(seed)
        split_rng.shuffle(all_samples)
        n = len(all_samples)
        train_end = int(n * 0.60)
        val_end = int(n * 0.80)
        self.train_samples = all_samples[:train_end]
        self.validation_samples = all_samples[train_end:val_end]
        self.test_samples = all_samples[val_end:]
        self.rng = random.Random(seed + 37)

    @staticmethod
    def evaluate_samples(samples: list[tuple[list[float], int]],
                         weights: list[float], threshold: float) -> dict[str, int]:
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

_SECURE_RANDOM = secrets.SystemRandom()


def l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def train_dp(client: SimulatedBankClient, initial_weights: list[float], *, epochs: int,
              learning_rate: float, l2: float, clip_norm: float,
              noise_multiplier: float, batch_size: int) -> tuple[list[float], int]:
    """Train locally with disjoint mini-batches, per-record clipping, and Gaussian noise.

    Each record is included in one batch per local epoch. For a batch of size B,
    the mean of clipped per-record gradients has replace-one sensitivity <= 2*C/B;
    Gaussian noise with standard deviation 2*sigma*C/B is added to each mean-gradient
    coordinate. No intermediate model weights leave the client between batches.
    The stated RDP accounting composes one affected Gaussian mechanism per record per
    epoch (disjoint batches within an epoch; later steps in that epoch are post-processing
    of the model after the record's batch). Noise uses OS-backed randomness.
    """
    weights = list(initial_weights)
    samples = client.train_samples
    n = len(samples)
    if n <= 0:
        raise ValueError(f"{client.bank_name} has no training records")

    indices = list(range(n))
    for _ in range(epochs):
        client.rng.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_n = len(batch_indices)
            grad_sum = [0.0] * len(weights)
            for idx in batch_indices:
                x, y = samples[idx]
                p = sigmoid(sum(w * v for w, v in zip(weights, x)))
                err = p - y
                per_record = [err * value for value in x]
                norm = l2_norm(per_record)
                scale = min(1.0, clip_norm / max(norm, 1e-12))
                for j, value in enumerate(per_record):
                    grad_sum[j] += value * scale

            grads = [g / batch_n for g in grad_sum]
            noise_std = (2.0 * noise_multiplier * clip_norm) / batch_n
            grads = [g + _SECURE_RANDOM.gauss(0.0, noise_std) for g in grads]
            for j in range(len(weights)):
                if j != 0:  # L2 penalty is data-independent given current weights.
                    grads[j] += l2 * weights[j]
                weights[j] -= learning_rate * grads[j]
    return weights, n


def epsilon_for_gaussian_composition(noise_multiplier: float, steps: int,
                                     delta: float) -> tuple[float, float]:
    """Conservative RDP-to-(epsilon,delta) conversion for composed Gaussian steps."""
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be between 0 and 1")
    if noise_multiplier <= 0 or steps < 1:
        raise ValueError("noise multiplier and number of steps must be positive")
    orders = [1.1, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
              12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0]
    best_eps = float("inf")
    best_order = orders[0]
    for alpha in orders:
        rdp = steps * alpha / (2.0 * noise_multiplier * noise_multiplier)
        eps = rdp + math.log(1.0 / delta) / (alpha - 1.0)
        if eps < best_eps:
            best_eps, best_order = eps, alpha
    return best_eps, best_order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/clients"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_dp"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=5.0,
                        help="Gaussian noise multiplier sigma; larger means more privacy and more noise")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional fixed threshold; default selects by validation F2")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for split/client training order; DP Gaussian noise remains OS-random")
    args = parser.parse_args()
    if args.rounds < 1 or args.local_epochs < 1:
        parser.error("rounds and local-epochs must be positive")
    if args.learning_rate <= 0 or args.clip_norm <= 0 or args.noise_multiplier <= 0 or args.batch_size < 1:
        parser.error("learning-rate, clip-norm, noise-multiplier and batch-size must be positive")
    if args.threshold is not None and not 0.0 < args.threshold < 1.0:
        parser.error("threshold must be between 0 and 1")

    clients = [Phase3BankClient(bank, args.data_root / bank / "transactions.csv",
                                args.seed + i * 1009) for i, bank in enumerate(BANKS)]
    weights = [0.0] * len(FEATURE_NAMES)
    history: list[dict[str, Any]] = []
    print("FederatedShield AI — Phase 3 DP federated-learning prototype")
    print("Data: synthetic only | mode: single-process simulation")
    print("Protection: record-level DP-SGD on local training records; validation/test reports are not DP")
    print(f"Clients={len(clients)} rounds={args.rounds} local_epochs={args.local_epochs} "
          f"clip_norm={args.clip_norm} noise_multiplier={args.noise_multiplier} delta={args.delta}")

    start_all = time.perf_counter()
    for round_no in range(1, args.rounds + 1):
        round_start = time.perf_counter()
        updates: list[tuple[list[float], int]] = []
        for client in clients:
            local_weights, count = train_dp(
                client, weights, epochs=args.local_epochs,
                learning_rate=args.learning_rate, l2=args.l2,
                clip_norm=args.clip_norm, noise_multiplier=args.noise_multiplier,
                batch_size=args.batch_size,
            )
            updates.append((local_weights, count))
            print(f"  round {round_no} | {client.bank_name}: DP local training complete; model update sent")
        weights = fedavg(updates)
        elapsed = time.perf_counter() - round_start
        history.append({"round": round_no, "duration_seconds": round(elapsed, 4),
                        "clients": len(clients), "training_samples_reported": sum(n for _, n in updates)})
        print(f"Round {round_no}/{args.rounds} aggregated in {elapsed:.2f}s")

    # This selection and test reporting uses validation/test labels, so those outputs
    # are explicitly outside the model-training DP guarantee in this prototype.
    validation_summary = None
    if args.threshold is None:
        candidates = [i / 100 for i in range(1, 81)]
        best = None
        for candidate in candidates:
            counts = sum_counts([c.evaluate_samples(c.validation_samples, weights, candidate) for c in clients])
            summary = summarize_counts(counts)
            rank = (summary["f2"], summary["recall"], summary["precision"])
            if best is None or rank > best[0]:
                best = (rank, candidate, summary)
        assert best is not None
        threshold = best[1]
        validation_summary = {"threshold": threshold, "metrics": best[2],
                              "selection_rule": "maximum validation F2; tie-break recall then precision",
                              "privacy_note": "validation-based threshold selection is not covered by training-model DP guarantee"}
    else:
        threshold = args.threshold
        validation_summary = {"threshold": threshold, "selection_rule": "user-specified threshold",
                              "privacy_note": "validation-based threshold selection not performed"}

    per_bank_counts = {c.bank_name: c.evaluate_samples(c.test_samples, weights, threshold) for c in clients}
    aggregate = summarize_counts(sum_counts(list(per_bank_counts.values())))
    by_bank = {name: summarize_counts(counts) for name, counts in per_bank_counts.items()}
    total_seconds = time.perf_counter() - start_all
    # Under the documented fixed disjoint-batch assumption, one training record
    # appears in one batch per local epoch; the subsequent local steps in that epoch
    # are post-processing with respect to that record. Compose across rounds/epochs.
    steps_per_client = args.rounds * args.local_epochs
    epsilon, best_order = epsilon_for_gaussian_composition(args.noise_multiplier, steps_per_client, args.delta)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "project": "FederatedShield AI", "model_type": "federated_logistic_regression_dp_sgd_demo",
        "feature_names": FEATURE_NAMES, "weights": [round(w, 12) for w in weights],
        "threshold": threshold, "seed": args.seed, "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "privacy": {"mechanism": "disjoint-mini-batch per-record gradient clipping + Gaussian noise",
                    "adjacency": "replace-one training record within one client's fixed-size training set",
                    "clip_norm": args.clip_norm, "noise_multiplier": args.noise_multiplier,
                    "delta": args.delta, "epsilon_per_client_training_records": round(epsilon, 4),
                    "rdp_order_selected": best_order,
                    "steps_per_client": steps_per_client,
                    "not_covered": ["validation/test evaluation outputs", "threshold selection based on validation labels",
                                    "dataset size/client metadata", "network transport or process isolation"]},
        "scope_note": "Synthetic data and single-process prototype; not production banking security."}
    metrics_payload = {
        "project": "FederatedShield AI", "dataset_type": "synthetic_demo_only",
        "scope_note": "Does not establish real-world fraud performance or production security.",
        "simulation_note": "Three clients run in one process; directories are not security boundaries.",
        "configuration": {"rounds": args.rounds, "local_epochs": args.local_epochs,
                          "learning_rate": args.learning_rate, "l2": args.l2,
                          "batch_size": args.batch_size,
                          "clip_norm": args.clip_norm, "noise_multiplier": args.noise_multiplier,
                          "delta": args.delta, "decision_threshold": threshold, "seed": args.seed,
                          "split": "60% train / 20% validation / 20% test per bank (Phase 3 split; distinct from Phase 2)"},
        "privacy_accounting": {"mechanism": "Gaussian mechanism on clipped mini-batch mean gradients; disjoint batches per epoch, no batch reuse within an epoch",
                               "adjacency": "replace-one training record within one client's fixed-size training set",
                               "epsilon_per_client_training_records": round(epsilon, 4), "delta": args.delta,
                               "rdp_order_selected": best_order, "steps_per_client": steps_per_client,
                               "training_model_only": True,
                               "not_covered": ["validation/test metrics and threshold selection", "metadata and row counts",
                                               "network/process isolation", "other system logging"]},
        "validation_threshold_selection": validation_summary,
        "test_metrics_threshold": aggregate,
        "per_bank_test_metrics": by_bank,
        "round_history": history, "total_training_seconds": round(total_seconds, 4),
        "total_test_samples": aggregate["total"],
    }
    (args.output_dir / "federated_model_dp.json").write_text(json.dumps(model_payload, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "phase3_dp_metrics.json").write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
    print("\n=== Held-out synthetic test metrics ===")
    print(json.dumps(aggregate, indent=2))
    print(f"\nTraining runtime: {total_seconds:.2f}s")
    print(f"Training-record privacy accounting (per client): epsilon={epsilon:.4f}, delta={args.delta:g}; "
          f"RDP order={best_order}, composed steps={steps_per_client}")
    print(f"Saved model: {args.output_dir / 'federated_model_dp.json'}")
    print(f"Saved metrics: {args.output_dir / 'phase3_dp_metrics.json'}")
    print("Caution: the privacy accounting covers training-record influence on model parameters only, not validation/test outputs or deployment security.")

if __name__ == "__main__":
    main()
