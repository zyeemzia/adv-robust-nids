#!/usr/bin/env bash
# Runs the whole pipeline on small synthetic data. Outputs go to */smoke/.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export NIDS_RAW_DIR="$PWD/tests/synthetic_raw" NIDS_OUT_TAG=smoke
export NIDS_N_ATTACK=200 NIDS_N_HSJ=5 NIDS_N_ADV=1000
python tests/make_synthetic.py "$NIDS_RAW_DIR"
cd src
for step in inspect_data data train_baselines run_attacks run_poisoning run_defense report; do
  echo "=== $step ===" ; python "$step.py"
done
