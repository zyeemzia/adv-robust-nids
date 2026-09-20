#!/usr/bin/env bash
# Full Project-2 pipeline on the real data in data/raw.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p logs
cd src
for step in inspect_data data train_baselines run_attacks run_poisoning run_defense run_defense_madry run_hsj run_detectors run_backdoor report; do
  echo "=== $step ===" ; python "$step.py" 2>&1 | tee "../logs/$step.log"
done
