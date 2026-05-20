#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=1:30:00
#PJM -N ftllm_feb24calibrated_merge_32b_lr1x
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_feb24calibrated_merge_32b_lr1x.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

# Merge folds 1..5 (assuming fold1 pilot + folds2to5 runs are complete).
conda run --no-capture-output -n llm_newer python -u \
  scripts/merge_finetune_llm_cv_results.py \
  --results-path results/finetuned_llm/feb24calibrated-cn32b-lr1x.csv \
  --stdout-file results/finetuned_llm/logs/feb24calibrated-cn32b-lr1x.log \
  --folds 1 2 3 4 5

conda run --no-capture-output -n llm_newer python -u \
  scripts/merge_finetune_llm_cv_results.py \
  --results-path results/finetuned_llm/feb24calibrated-esde32b-lr1x.csv \
  --stdout-file results/finetuned_llm/logs/feb24calibrated-esde32b-lr1x.log \
  --folds 1 2 3 4 5

# Full combined run merge requires complete folds 1..5 for both cn and esde runs.
# Uncomment when folds 2-3 also exist:
# conda run --no-capture-output -n llm_newer python -u \
#   scripts/merge_finetune_llm_runs.py \
#   --dry-run \
#   --run-names \
#   feb24calibrated-esde32b-lr1x--zai-org--GLM-4-32B-Base-0414 \
#   feb24calibrated-cn32b-lr1x--zai-org--GLM-4-32B-Base-0414 \
#   --target-run-name feb24calibrated-all32b-lr1x--zai-org--GLM-4-32B-Base-0414
