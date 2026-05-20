#!/usr/bin/env bash
set -euo pipefail

# Resolve project root from script location so relative paths work from any cwd.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python scripts/finetune_llm.py \
  --config-name full-data-all \
  --train-on-full-data \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --trust-remote-code \
  --stdout-file results/finetuned_llm/logs/full-data-all.log
