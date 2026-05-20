#!/bin/bash
# DO NOT RUN AS A JOB: 
# IGNORE:
#PJM -g gh35
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=3:00:00
#PJM -N apr17c-sweep-eval-summary
#PJM -j
#PJM -o results/finetuned_llm/logs/apr17checkpointpred-sweep_eval_summary_only.out

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
  jobs/apr17checkpointpred-sweep_eval_summary_only.sh [--overwrite]

Description:
  Evaluate existing sweep prediction CSVs (no model loading/inference) and
  write epoch summary CSVs for the apr17 checkpoint sweep run set.
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
  --config-name apr16calibrated-Ministral-3-8B-Base-allinone-lr1x-full \
  --model-name mistralai/Ministral-3-8B-Base-2512 \
  --epochs 4 \
  "${OVERWRITE_ARGS[@]}"

conda run --no-capture-output -n bea2026st \
  python -u scripts/evaluate_finetune_llm_epoch_sweep_predictions.py \
  --config-name apr16calibrated-Qwen2.5-7B-allinone-lr1x-full \
  --model-name Qwen/Qwen2.5-7B \
  --epochs 4 \
  "${OVERWRITE_ARGS[@]}"

conda run --no-capture-output -n bea2026st \
  python -u scripts/evaluate_finetune_llm_epoch_sweep_predictions.py \
  --config-name apr16calibrated-glm-4-9b-allinone-lr1x-full \
  --model-name zai-org/glm-4-9b \
  --epochs 4 \
  "${OVERWRITE_ARGS[@]}"

conda run --no-capture-output -n bea2026st \
  python -u scripts/evaluate_finetune_llm_epoch_sweep_predictions.py \
  --config-name apr16calibrated-Ministral-3-14B-abl-raft \
  --model-name mistralai/Ministral-3-14B-Base-2512 \
  --epochs 4 \
  "${OVERWRITE_ARGS[@]}"

conda run --no-capture-output -n bea2026st \
  python -u scripts/evaluate_finetune_llm_epoch_sweep_predictions.py \
  --config-name mar20calibrated-Ministral-3-14B-Base-allinone-lr1x-full \
  --model-name mistralai/Ministral-3-14B-Base-2512 \
  --epochs 4 \
  "${OVERWRITE_ARGS[@]}"
