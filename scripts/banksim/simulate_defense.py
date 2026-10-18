import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_public/processed"
CLIENT_DIR = DATA / "clients"
OUT = ROOT / "outputs_public"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
ROUNDS = 5
LOCAL_EPOCHS = 2
BATCH_SIZE = 1024
LEARNING_RATE = 0.03
MALICIOUS_CLIENT = 2       # Client C, zero-based index
ATTACK_SCALE = 10.0        # Deliberately obvious model-update amplification
NORM_MULTIPLIER = 3.0      # Reject if update norm exceeds 3x peer median


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        names = [name for name in reader.fieldnames if name != "fraud"]
        rows = list(reader)

    X = np.asarray(
        [[float(row[name]) for name in names] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([int(row["fraud"]) for row in rows], dtype=np.float64)
    return X, y, names


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def local_train(X, y, start_w, start_b, rng):
    w = start_w.copy()
    b = float(start_b)
    n = len(y)

    positives = max(float(y.sum()), 1.0)
    negatives = max(float(n - y.sum()), 1.0)
    positive_weight = min(negatives / positives, 20.0)
    sample_weights = np.where(y == 1, positive_weight, 1.0)
    sample_weights /= sample_weights.mean()

    for _ in range(LOCAL_EPOCHS):
        order = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb, yb = X[idx], y[idx]
            sw = sample_weights[idx]
            p = sigmoid(xb @ w + b)
            error = (p - yb) * sw

            w -= LEARNING_RATE * (xb.T @ error / len(idx))
            b -= LEARNING_RATE * float(error.mean())

    return w, b


def evaluate(y, scores, threshold):
    pred = scores >= threshold
    tp = int(np.sum((y == 1) & pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    tn = int(np.sum((y == 0) & ~pred))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(y),
    }


def choose_threshold(y, scores):
    candidates = np.arange(0.05, 0.501, 0.01)
    results = [
        (evaluate(y, scores, float(t)), float(t))
        for t in candidates
    ]
    result, threshold = max(
        results,
        key=lambda item: (
            item[0]["f1"],
            item[0]["precision"],
            -item[0]["fp"],
        ),
    )
    return threshold, result


def train_scenario(clients, X_val, y_val, X_test, y_test,
                   rng, attacked=False, defended=False):
    w = np.zeros(X_val.shape[1], dtype=np.float64)
    b = 0.0
    audit = []

    for round_no in range(1, ROUNDS + 1):
        updates = []

        for client_id, (X, y) in enumerate(clients):
            local_w, local_b = local_train(X, y, w, b, rng)
            delta_w = local_w - w
            delta_b = local_b - b

            malicious = attacked and client_id == MALICIOUS_CLIENT
            if malicious:
                delta_w *= ATTACK_SCALE
                delta_b *= ATTACK_SCALE

            norm = float(np.sqrt(np.dot(delta_w, delta_w) + delta_b**2))
            updates.append({
                "client": f"client_{chr(97 + client_id)}",
                "delta_w": delta_w,
                "delta_b": delta_b,
                "count": len(y),
                "norm": norm,
                "malicious_injected": malicious,
            })

        accepted = [True] * len(updates)
        threshold_norm = None

        if defended:
            # Median peer norm is robust to one very large outlier among
            # three clients. This detects the specific amplification attack,
            # not every possible malicious update.
            norms = np.array([u["norm"] for u in updates])
            threshold_norm = NORM_MULTIPLIER * float(np.median(norms))

            for i, update in enumerate(updates):
                finite = (
                    np.all(np.isfinite(update["delta_w"]))
                    and np.isfinite(update["delta_b"])
                    and np.isfinite(update["norm"])
                )
                too_large = update["norm"] > threshold_norm
                accepted[i] = bool(finite and not too_large)

                audit.append({
                    "round": round_no,
                    "client": update["client"],
                    "update_norm": update["norm"],
                    "norm_limit": threshold_norm,
                    "accepted": accepted[i],
                    "reason": (
                        "non_finite_update" if not finite
                        else "update_norm_exceeded" if too_large
                        else "accepted"
                    ),
                    "attack_injected": update["malicious_injected"],
                })

        selected = [
            (u, keep) for u, keep in zip(updates, accepted) if keep
        ]
        if not selected:
            raise RuntimeError("All client updates were rejected")

        total = sum(u["count"] for u, _ in selected)
        aggregate_w = sum(
            u["delta_w"] * u["count"] for u, _ in selected
        ) / total
        aggregate_b = sum(
            u["delta_b"] * u["count"] for u, _ in selected
        ) / total

        w += aggregate_w
        b += aggregate_b

        val_result = evaluate(y_val, sigmoid(X_val @ w + b), 0.5)
        print(
            f"  Round {round_no}/{ROUNDS}: "
            f"validation F1@0.5={val_result['f1']:.4f}; "
            f"accepted={sum(accepted)}/{len(accepted)}"
        )

    val_scores = sigmoid(X_val @ w + b)
    threshold, val_result = choose_threshold(y_val, val_scores)
    test_scores = sigmoid(X_test @ w + b)
    test_result = evaluate(y_test, test_scores, threshold)

    return {
        "selected_threshold": threshold,
        "validation_metrics": val_result,
        "test_metrics": test_result,
        "audit_log": audit,
        "weights": w,
        "bias": b,
    }


def main():
    started = time.time()
    rng = np.random.default_rng(SEED)

    X_train, y_train, names = load_csv(DATA / "train.csv")
    X_val, y_val, val_names = load_csv(DATA / "validation.csv")
    X_test, y_test, test_names = load_csv(DATA / "test.csv")

    if names != val_names or names != test_names:
        raise ValueError("Feature mismatch across splits")

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-12] = 1.0

    def scale(X):
        return np.clip((X - mean) / std, -10, 10)

    X_val, X_test = scale(X_val), scale(X_test)
    clients = []

    for name in ("client_a_train.csv", "client_b_train.csv",
                 "client_c_train.csv"):
        X, y, client_names = load_csv(CLIENT_DIR / name)
        if client_names != names:
            raise ValueError(f"Feature mismatch in {name}")
        clients.append((scale(X), y))

    scenarios = {}
    settings = [
        ("clean_fedavg", False, False),
        ("attacked_fedavg", True, False),
        ("defended_fedavg", True, True),
    ]

    for name, attacked, defended in settings:
        print(f"\n=== {name} ===")
        result = train_scenario(
            clients, X_val, y_val, X_test, y_test, rng,
            attacked=attacked, defended=defended,
        )

        scenarios[name] = {
            "selected_threshold": result["selected_threshold"],
            "validation_metrics": result["validation_metrics"],
            "test_metrics": result["test_metrics"],
            "audit_log": result["audit_log"],
        }

        if name == "defended_fedavg":
            np.savez(
                OUT / "banksim_defended_model.npz",
                weights=result["weights"],
                bias=result["bias"],
                mean=mean,
                std=std,
            )

    summary = {
        "dataset": "BankSim",
        "synthetic_demo_data": True,
        "experiment": "controlled client update amplification",
        "malicious_client": "client_c",
        "attack_scale": ATTACK_SCALE,
        "defense": (
            "reject non-finite updates and updates with norm above "
            f"{NORM_MULTIPLIER} times the round median"
        ),
        "scenarios": scenarios,
        "elapsed_seconds": round(time.time() - started, 3),
        "limitations": [
            "All clients are simulated in one process on one server.",
            "Attack is an obvious scaled-update attack, not a comprehensive threat model.",
            "Norm checks cannot detect every poisoned or backdoored update.",
            "No differential privacy or client isolation is provided by this experiment.",
            "Thresholds are selected on validation data; test data is held out.",
        ],
    }

    path = OUT / "banksim_defense_comparison.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (OUT / "banksim_defense_audit.jsonl").open("w") as f:
        for row in scenarios["defended_fedavg"]["audit_log"]:
            f.write(json.dumps(row) + "\n")

    print("\n=== Held-out test comparison ===")
    for name, result in scenarios.items():
        print(
            f"{name}: "
            + json.dumps({
                "threshold": result["selected_threshold"],
                **result["test_metrics"],
            })
        )

    print("\nSaved comparison:", path)
    print("Saved audit log:", OUT / "banksim_defense_audit.jsonl")
    print("Elapsed seconds:", round(time.time() - started, 3))


if __name__ == "__main__":
    main()
