import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data_public/raw/bs140513_032310.csv"
OUT = ROOT / "data_public/processed"
CLIENTS = OUT / "clients"
SEED = 42

random.seed(SEED)
OUT.mkdir(parents=True, exist_ok=True)
CLIENTS.mkdir(parents=True, exist_ok=True)


def clean(value):
    return (value or "").strip().strip("'\"").strip()


def load_rows():
    rows = []
    with RAW.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {
            "step", "age", "gender", "category", "amount", "fraud"
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")

        for line_no, row in enumerate(reader, start=2):
            try:
                label = int(clean(row["fraud"]))
                amount = float(clean(row["amount"]))
                step = float(clean(row["step"]))
                age = clean(row["age"])
                gender = clean(row["gender"])
                category = clean(row["category"])

                if label not in (0, 1):
                    raise ValueError("fraud label must be 0 or 1")
                if amount < 0 or not math.isfinite(amount):
                    raise ValueError("invalid amount")
                if not math.isfinite(step):
                    raise ValueError("invalid step")
                if not age or not gender or not category:
                    raise ValueError("missing feature")

                rows.append({
                    "step": step,
                    "amount": amount,
                    "age": age,
                    "gender": gender,
                    "category": category,
                    "fraud": label,
                })
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid row {line_no}: {e}") from e

    return rows


rows = load_rows()
if not rows:
    raise ValueError("Dataset is empty")

# Fixed vocabularies learned from feature values, not from fraud labels.
ages = sorted({r["age"] for r in rows})
genders = sorted({r["gender"] for r in rows})
categories = sorted({r["category"] for r in rows})

feature_names = ["step", "log_amount"]
feature_names += [f"age_{x}" for x in ages]
feature_names += [f"gender_{x}" for x in genders]
feature_names += [f"category_{x}" for x in categories]

def transform(row):
    values = {
        "step": row["step"],
        "log_amount": math.log1p(row["amount"]),
    }
    for age in ages:
        values[f"age_{age}"] = int(row["age"] == age)
    for gender in genders:
        values[f"gender_{gender}"] = int(row["gender"] == gender)
    for category in categories:
        values[f"category_{category}"] = int(row["category"] == category)
    return [values[name] for name in feature_names] + [row["fraud"]]


# Stratify by label before splitting, keeping the test distribution realistic.
by_label = {0: [], 1: []}
for row in rows:
    by_label[row["fraud"]].append(row)

train_rows, val_rows, test_rows = [], [], []
for label, group in by_label.items():
    random.shuffle(group)
    n = len(group)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_rows.extend(group[:n_train])
    val_rows.extend(group[n_train:n_train + n_val])
    test_rows.extend(group[n_train + n_val:])

random.shuffle(train_rows)
random.shuffle(val_rows)
random.shuffle(test_rows)

def write_csv(path, data):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(feature_names + ["fraud"])
        for row in data:
            writer.writerow(transform(row))

write_csv(OUT / "train.csv", train_rows)
write_csv(OUT / "validation.csv", val_rows)
write_csv(OUT / "test.csv", test_rows)

# Divide only the training set among three simulated institutions.
# Each client contains both labels, with proportions kept approximately similar.
client_summaries = {}
for label in (0, 1):
    group = [r for r in train_rows if r["fraud"] == label]
    random.shuffle(group)
    for i, row in enumerate(group):
        # Round-robin allocation prevents one client receiving all fraud cases.
        pass
    buckets = [[], [], []]
    for i, row in enumerate(group):
        buckets[i % 3].append(row)

    for i, bucket in enumerate(buckets):
        client_summaries.setdefault(i, []).extend(bucket)

for i in range(3):
    client_rows = client_summaries[i]
    random.shuffle(client_rows)
    write_csv(CLIENTS / f"client_{chr(97+i)}_train.csv", client_rows)

def counts(data):
    c = Counter(r["fraud"] for r in data)
    return {
        "rows": len(data),
        "legitimate": c.get(0, 0),
        "fraud": c.get(1, 0),
        "fraud_rate_percent": round(100 * c.get(1, 0) / len(data), 4)
            if data else 0,
    }

summary = {
    "dataset": "BankSim",
    "source_file": str(RAW.relative_to(ROOT)),
    "synthetic_demo_data": True,
    "seed": SEED,
    "features": feature_names,
    "excluded_columns": [
        "customer", "merchant", "zipcodeOri", "zipMerchant"
    ],
    "splits": {
        "train": counts(train_rows),
        "validation": counts(val_rows),
        "test": counts(test_rows),
    },
    "clients": {
        f"client_{chr(97+i)}": counts(client_summaries[i])
        for i in range(3)
    },
    "note": (
        "Synthetic transactions. Validation/test are held out from client "
        "training. Client partition is a simulation, not real banks."
    ),
}

with (OUT / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

with (OUT / "schema.json").open("w", encoding="utf-8") as f:
    json.dump({
        "feature_names": feature_names,
        "target": "fraud",
        "encoding": "one-hot age, gender, category; log1p amount",
    }, f, indent=2)

print(json.dumps(summary, indent=2))
print("\nPreprocessing complete.")
