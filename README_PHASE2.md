# FederatedShield AI — Phase 2: Federated Learning Prototype

## What it does

- Loads each bank's own synthetic CSV into a simulated local client.
- Splits each bank's rows locally into 80% training and 20% test data.
- Trains a logistic-regression model locally with a fixed shared feature schema.
- Aggregates only trained model parameters using sample-weighted FedAvg.
- Evaluates the shared model within each client and sends confusion-matrix counts for aggregation.
- Saves a demo model and metrics JSON.

## Run on Ubuntu / IBM LinuxONE

No pip packages are required; this phase uses Python's standard library.

```bash
cd ~/federatedshield_phase1
source .venv/bin/activate
python3 scripts/train_federated.py --rounds 5 --local-epochs 10 --threshold 0.07
```

Outputs:

- `outputs/federated_model.json`
- `outputs/phase2_metrics.json`

Optional smaller smoke test:

```bash
python3 scripts/train_federated.py --rounds 2 --local-epochs 1 --batch-size 256 --threshold 0.07
```

## Important limitations

This phase is a **single-process simulation** of three clients. Their local CSVs are loaded into objects in the same process; separate folders are not security boundaries. Raw rows are not passed to the FedAvg aggregation function, but this is not yet a real distributed deployment.

Differential privacy, secure transport, authentication, true process/network isolation, malicious-client robust aggregation, and production model validation are **not implemented** in this phase. The reported metrics use synthetic labels and must not be presented as evidence of real-world banking fraud performance. The default probability threshold of 0.07 is a fixed demonstration setting chosen to surface more candidates with these sparse synthetic labels; it is not a calibrated or operational fraud-risk threshold. It must be tuned on validation data before any real use.
