#!/usr/bin/env bash
# Download C4 calibration data to hcsmoe/data/c4-train.00000-of-01024.json
# Run from HC-SMoE repo root:  bash experiment/download_c4.sh
# Optional: NUM=512 bash experiment/download_c4.sh  (fewer examples)
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
python experiment/download_c4.py --num_examples "${NUM:-2048}"
