#!/usr/bin/env bash
set -euo pipefail
printf '=== OS ===\n'
cat /etc/os-release | sed -n '1,8p'
printf '\n=== Architecture ===\n'
uname -m
printf '\n=== Python ===\n'
python3 --version
printf '\n=== Disk ===\n'
df -h "$HOME" | tail -n 1
printf '\n=== Optional ML packages visible to system Python ===\n'
python3 - <<'PY'
import importlib.util
for name in ("numpy", "pandas", "sklearn", "streamlit", "fastapi"):
    print(f"{name}: {'available' if importlib.util.find_spec(name) else 'not installed'}")
PY
