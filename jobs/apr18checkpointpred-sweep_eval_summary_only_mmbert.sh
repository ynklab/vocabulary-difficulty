#!/bin/bash
# DO NOT RUN AS A JOB: 
# IGNORE:
#PJM -g gh35
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=3:00:00
#PJM -N apr18c-sweep-eval-summary-mmbert
#PJM -j
#PJM -o results/finetuned_llm/logs/apr18checkpointpred-sweep_eval_summary_only_mmbert.out

set -euo pipefail

OVERWRITE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  jobs/apr18checkpointpred-sweep_eval_summary_only_mmbert.sh [--overwrite]

Description:
  Evaluate existing sweep prediction CSVs (no model loading/inference) and
  write epoch summary CSVs for the apr18 checkpoint sweep run set.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# DO NOT RUN AS A JOB: cd "$HOME/bea2026st"
# DO NOT RUN AS A JOB: module load miniconda/py39_4.9.2

OVERWRITE_ARGS=()
if [[ "$OVERWRITE" -eq 1 ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

conda run --no-capture-output -n bea2026st \
  python -u scripts/evaluate_finetune_llm_epoch_sweep_predictions.py \
  --config-name mar26-mmbert-ep16-cnesde \
  --model-name jhu-clsp/mmBERT-base \
  --epochs 16 \
  "${OVERWRITE_ARGS[@]}"

conda run --no-capture-output -n bea2026st \
  python -u scripts/evaluate_finetune_llm_epoch_sweep_predictions.py \
  --config-name apr18-mmbert-ep16-cnesde-allinone \
  --model-name jhu-clsp/mmBERT-base \
  --epochs 16 \
  "${OVERWRITE_ARGS[@]}"
