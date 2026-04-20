#!/usr/bin/env bash
set -euo pipefail

python3 src/infer.py \
  --model-path outputs/dpo-lab8 \
  --dataset-path data/hhh_preferences.jsonl
