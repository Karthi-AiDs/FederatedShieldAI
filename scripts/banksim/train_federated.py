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

ROUNDS = 5
LOCAL_EPOCHS = 2
BATCH_SIZE = 1024
LEARNING_RATE = 0.03
SEED = 42


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        feature_names = [x for x in reader.fieldnames if x != "fraud"]
        rows = list(reader)

    X = np.array(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    y = np.array([int(row["fraud"]) for row in rows], dtype=np.float64)
    return X, y, feature_names


def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def train_local(X, y, w, b, rng):
    w = w.copy()
    n = len(y)

    # Inverse-frequency weighting helps the model learn the rare fraud class.
    positives = max(float(y.sum()), 1.0)
    negatives = max(float(n - y.sum()), 1.0)
    positive_weight = min(negatives / positives, 20.0)

    sample_weights = np.where(y == 1, positive_weight, 1.0)
    sample_weights = sample_weights / sample_weights.mean()

    for _ in range(LOCAL_EPOCHS):
        order = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb, yb = X[idx], y[idx]
            sw = sample_weights[idx]

            p = sigmoid(xb @ w + b)
            error = (p - yb) * sw

            grad_w = xb.T @ error / len(idx)
            grad_b = float(error.mean())

            w -= LEARNING_RATE * grad_w
            b -= LEARNING_RATE * grad_b

    return w, b


def metrics(y, scores, threshold):
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
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(y),
    }


def main():
    started = time.time()
    rng = np.random.default_rng(SEED)

    X_train, y_train, feature_names = load_csv(DATA / "train.csv")
    X_val, y_val, _ = load_csv(DATA / "validation.csv")
    X_test, y_test, _ = load_csv(DATA / "test.csv")

    # Standardize using training data only. Share the same transformation
    # across clients to keep parameter aggregation meaningful.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-12] = 1.0

    X_train = np.clip((X_train - mean) / std, -10, 10)
    X_val = np.clip((X_val - mean) / std, -10, 10)
    X_test = np.clip((X_test - mean) / std, -10, 10)

    client_paths = [
        CLIENT_DIR / "client_a_train.csv",
        CLIENT_DIR / "client_b_train.csv",
        CLIENT_DIR / "client_c_train.csv",
    ]
    clients = []
    for path in client_paths:
        X, y, names = load_csv(path)
        if names != feature_names:
            raise ValueError(f"Feature mismatch in {path}")
        X = np.clip((X - mean) / std, -10, 10)
        clients.append((X, y))

    w = np.zeros(X_train.shape[1], dtype=np.float64)
    b = 0.0

    print(f"Training on {len(clients)} simulated clients")
    print(f"Train={len(y_train)}, validation={len(y_val)}, test={len(y_test)}")

    for round_no in range(1, ROUNDS + 1):
        local_models = []
        total_examples = 0

        for client_no, (X, y) in enumerate(clients, start=1):
            local_w, local_b = train_local(X, y, w, b, rng)
            local_models.append((local_w, local_b, len(y)))
            total_examples += len(y)

        # FedAvg: aggregate model parameters weighted by client sample count.
        w = sum(
            local_w * count
            for local_w, _, count in local_models
        ) / total_examples
        b = sum(
            local_b * count
            for _, local_b, count in local_models
        ) / total_examples

        val_scores = sigmoid(X_val @ w + b)
        val_result = metrics(y_val, val_scores, 0.5)
        print(
            f"Round {round_no}/{ROUNDS}: "
            f"validation F1@0.5={val_result['f1']:.4f}"
        )

    # Select threshold on validation only; test data remains untouched.
    val_scores = sigmoid(X_val @ w + b)
    candidates = np.arange(0.05, 0.501, 0.01)
    threshold_results = [
        metrics(y_val, val_scores, float(t)) for t in candidates
    ]
    best = max(
        threshold_results,
        key=lambda m: (m["f1"], m["precision"], -m["fp"]),
    )
    threshold = best["threshold"]

    test_scores = sigmoid(X_test @ w + b)
    test_result = metrics(y_test, test_scores, threshold)

    result = {
        "dataset": "BankSim",
        "synthetic_demo_data": True,
        "method": "single-process federated simulation with FedAvg",
        "model": "logistic regression",
        "rounds": ROUNDS,
        "local_epochs_per_round": LOCAL_EPOCHS,
        "client_count": len(clients),
        "feature_count": len(feature_names),
        "selected_threshold": threshold,
        "validation_metrics": best,
        "test_metrics": test_result,
        "training_seconds": round(time.time() - started, 3),
        "limitations": [
            "Clients are simulated in one process on one machine.",
            "This is not a production deployment or proof of data privacy.",
            "Threshold was selected on validation data, not test data.",
            "Metrics are from synthetic BankSim data."
        ],
    }

    with (OUT / "banksim_federated_metrics.json").open("w") as f:
        json.dump(result, f, indent=2)

    np.savez(OUT / "banksim_federated_model.npz", weights=w, bias=b,
             mean=mean, std=std)

    print("\nFinal held-out test metrics:")
    print(json.dumps(test_result, indent=2))
    print(f"\nSaved metrics to {OUT / 'banksim_federated_metrics.json'}")
    print(f"Saved model to {OUT / 'banksim_federated_model.npz'}")


if __name__ == "__main__":
    main()
