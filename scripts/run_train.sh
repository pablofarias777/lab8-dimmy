#!/usr/bin/env bash
set -euo pipefail

python3 src/train_dpo.py \
  --dataset-path data/hhh_preferences.jsonl \
  --model-name gpt2 \
  --output-dir outputs/dpo-lab8 \
  --beta 0.1
