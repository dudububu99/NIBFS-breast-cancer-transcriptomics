#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "================================================================"
echo "NIBFS v1.2.5 - release verification (no manuscript rerun)"
echo "================================================================"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN was not found. Install Python 3.10+ first." >&2
  exit 1
fi

if [[ ! -x .venv_verify/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv_verify
fi

. .venv_verify/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-verify.txt
python scripts/verify_repository.py --with-tests

echo "================================================================"
echo "REPOSITORY VERIFICATION COMPLETED SUCCESSFULLY"
echo "No manuscript experiment was rerun."
echo "================================================================"
