# FederatedShield AI — Phase 3 bug fix

## What was fixed

The previous Phase 3 script imported the Phase 2 `SimulatedBankClient`, but that class only provides an 80/20 train/test split and the method `evaluate_counts()`. Phase 3 expected `validation_samples` and `evaluate_samples()`, so it crashed after training. This corrected script defines a Phase 3-specific client with a 60/20/20 train/validation/test split and evaluation helper. The Phase 2 script and its outputs are not modified.

## Install

From Windows PowerShell, upload the ZIP to the server (adjust the PEM path if needed):

```powershell
scp -i "C:\Users\admin\Desktop\IBM Z\MyKey1.pem" "$env:USERPROFILE\Downloads\federatedshield_phase3_fixed.zip" linux1@148.100.108.154:~
```

On LinuxONE:

```bash
cd ~/federatedshield_phase1
unzip -o ~/federatedshield_phase3_fixed.zip -d .
source .venv/bin/activate
```

Replace only the broken Phase 3 script and run it:

```bash
python3 scripts/train_federated_dp.py --rounds 5 --local-epochs 10 --batch-size 128 --clip-norm 1.0 --noise-multiplier 5 --delta 0.00001
```

Check outputs:

```bash
cat outputs_dp/phase3_dp_metrics.json
```

## Important limitations

- The input data is synthetic; the reported metrics do not establish real-world fraud performance.
- The three clients are simulated in one process. Their directories are not separate security boundaries and this is not a networked federated deployment.
- Training-model privacy accounting applies only under the stated fixed-size replace-one adjacency and implementation assumptions. Validation/test reporting, threshold selection, metadata, logs, and deployment are not covered by the model-training privacy guarantee.
- A privacy budget is not a guarantee that customer data can never be inferred. Report the calculated epsilon/delta and the assumptions with the demo.
