# FederatedShield AI — Phase 3

## What this adds
- A record-level DP-SGD-style training step for the local logistic regression clients.
- Per-record L2 gradient clipping followed by Gaussian noise on each mini-batch mean gradient.
- RDP composition assuming fixed-size client training sets, replace-one-record adjacency, disjoint batches per epoch, and no intermediate model weights leaving the client between batches.
- Separate outputs under `outputs_dp/` so the Phase 2 baseline is preserved.

## Important scope limits
- Data is synthetic.
- Three clients still execute in one Python process. They are not isolated security boundaries.
- The DP accounting applies to the released model parameters with respect to each client's training records under the documented assumptions. It does not cover validation/test metrics, validation-based threshold selection, row counts/metadata, logs, transport, or process isolation.
- DP does not make the entire application secure. Do not claim production-grade privacy.
- Each training record is used in one mini-batch per local epoch. The accountant composes `rounds * local_epochs` Gaussian mechanisms per client, relying on disjoint batches within an epoch and the fact that later local updates in that epoch are post-processing with respect to a record already used. Review this assumption carefully before presenting a formal privacy claim.

## Step A — Re-run the non-private baseline correctly
Use Phase 2.1's validation-based threshold selection by NOT passing `--threshold`:

```bash
source .venv/bin/activate
python3 scripts/train_federated.py --rounds 5 --local-epochs 10
cat outputs/phase2_metrics.json
```

This preserves validation-based threshold selection and reports the held-out test metrics. Do not use the old fixed `--threshold 0.07` unless you intentionally want that fixed operating point.

## Step B — Install this script
Unzip this archive into the project root, so `scripts/train_federated_dp.py` overlays into the existing `scripts/` directory and `README_PHASE3.md` lands in the root.

## Step C — Run the DP model

```bash
python3 scripts/train_federated_dp.py \
  --rounds 5 \
  --local-epochs 10 \
  --batch-size 128 \
  --clip-norm 1.0 \
  --noise-multiplier 5 \
  --delta 0.00001
```

Noise multiplier 5 is a starting experiment, not necessarily the best privacy/utility balance. Compare it with 3 and 10 using the same seed and protocol; noise sampling is intentionally OS-random, so DP runs are not bit-for-bit reproducible. Smaller multiplier means less noise and weaker privacy (larger epsilon); larger multiplier means more noise and stronger privacy (smaller epsilon).

Outputs:
- `outputs_dp/federated_model_dp.json`
- `outputs_dp/phase3_dp_metrics.json`

## Phase 3 interpretation
Record the actual epsilon, delta, precision, recall, F1/F2, false positives/negatives and runtime. Compare with Phase 2 using the same split and validation procedure. Since the data is synthetic and validation/test outputs are not DP, present these results as an educational prototype rather than a formal privacy assessment of a real banking system. The RDP accounting implementation is a prototype; have it independently reviewed before claiming a formal guarantee in a real deployment.
