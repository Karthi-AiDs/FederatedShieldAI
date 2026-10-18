# FederatedShield AI — Phase 2 Prediction Improvement

This add-on keeps the existing standard-library logistic-regression FedAvg baseline and adds a validation threshold policy that can prioritize alert quality. It does not install packages and does not claim production-grade fraud detection.

## Recommended run on IBM LinuxONE

From the project root (`~/federatedshield_phase1`):

```bash
source .venv/bin/activate
python3 scripts/generate_synthetic_data.py --rows-per-bank 20000 --seed 42 --output-dir data_v2/clients
python3 scripts/train_federated_v2.py \
  --data-root data_v2/clients \
  --output-dir outputs_v2 \
  --rounds 5 \
  --local-epochs 10 \
  --learning-rate 0.15 \
  --batch-size 128 \
  --l2 0.0005 \
  --min-precision 0.14
```

The command creates a separate `data_v2/` dataset and `outputs_v2/` folder, so the previous 5,000-row-per-bank data and baseline outputs are preserved.

## Threshold selection

`--min-precision 0.14` means the operating threshold is selected using validation data only: among thresholds whose validation precision is at least 14%, choose the one with highest recall (tie-break using F2 and precision). The held-out test set is only evaluated after that threshold is fixed. If no threshold meets the requested precision floor, the program stops rather than silently selecting a different rule.

Accuracy by itself can mislead on imbalanced fraud data. Review precision, recall, F1/F2, specificity, false-positive rate, and balanced accuracy together.

## Scope and limitations

- All transactions are synthetic demonstration data; results do not establish real-world fraud performance.
- All three clients run in one process; separate folders are not security boundaries.
- This script is not differentially private. Keep Phase 2 and Phase 3 results separate.
- The threshold policy improves the precision/recall operating point; it does not make the underlying model intrinsically more accurate.
