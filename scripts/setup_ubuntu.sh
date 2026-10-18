#!/usr/bin/env bash
set -euo pipefail

if ! grep -qi 'ubuntu' /etc/os-release; then
  echo 'WARNING: This script was designed for the Ubuntu Datathon image.'
  echo 'Detected OS:'
  grep -E '^(PRETTY_NAME|ID)=' /etc/os-release || true
  read -r -p 'Continue anyway? Type yes to continue: ' answer
  [[ "$answer" == "yes" ]] || exit 1
fi

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git unzip ca-certificates

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

printf '\nEnvironment created. Activate it with:\n  source .venv/bin/activate\n\n'
printf 'Python version in the new environment:\n'
.venv/bin/python --version
printf '\nSystem checks:\n'
bash scripts/check_environment.sh
printf '\nPhase 1 setup complete. No pip packages are required yet.\n'
