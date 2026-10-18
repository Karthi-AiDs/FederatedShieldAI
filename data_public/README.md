# Dataset

The project uses the BankSim synthetic transaction dataset file `bs140513_032310.csv`.

The raw CSV is intentionally not committed by default: it is approximately 47 MB, and redistribution must follow the upstream dataset's terms. See [DATASET_LICENSE.md](../DATASET_LICENSE.md) for source attribution and license notes.

## Use an existing local copy

Place the file at:

```text
data_public/raw/bs140513_032310.csv
```

Then prepare the train/validation/test splits with:

```bash
source .venv/bin/activate
python3 scripts/banksim/prepare_banksim.py
```

Do not upload private, real customer transaction data. This project is designed to use the public synthetic dataset.

If you decide to publish a copy of the raw dataset in this repository, first verify the current upstream license and attribution requirements, then intentionally force-add the file because CSVs and raw data are ignored by `.gitignore`. GitHub's ordinary Git repository has a 100 MB per-file limit; use Git LFS or a dataset-hosting service for larger files, subject to license terms.
