#!/usr/bin/env bash
# One-click reproduction: integrity check, metrics, statistics and live decoding check.
# Usage: bash run_all.sh [--skip-decode-check] [--device cpu|cuda:0]
# Set PYTHON to an interpreter that has the packages of requirements.txt (default: python3).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
mkdir -p outputs
"$PY" reproduce.py "$@" 2>&1 | tee outputs/run_log.txt
exit "${PIPESTATUS[0]}"
