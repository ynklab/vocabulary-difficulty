#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=1:30:00
#PJM -N ftllm_feb23cv1_merge_all
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_feb23cv1_merge_all.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_env python -u \
  scripts/merge_finetune_llm_cv_results.py \
  --results-path results/finetuned_llm/feb23cv1-esde.csv \
  --stdout-file results/finetuned_llm/logs/feb23cv1-esde.log \
  --folds 1 2 3 4 5

conda run --no-capture-output -n llm_env python -u \
  scripts/merge_finetune_llm_cv_results.py \
  --results-path results/finetuned_llm/feb23cv1-cn.csv \
  --stdout-file results/finetuned_llm/logs/feb23cv1-cn.log \
  --folds 1 2 3 4 5

# Optional dry run for the final combined merge (uncomment to validate first):
# conda run --no-capture-output -n llm_env python -u \
#   scripts/merge_finetune_llm_runs.py \
#   --dry-run \
#   --run-names \
#   feb23cv1-esde--zai-org--glm-4-9b \
#   feb23cv1-cn--zai-org--glm-4-9b \
#   --target-run-name feb23cv1-all--zai-org--glm-4-9b

conda run --no-capture-output -n llm_env python -u \
  scripts/merge_finetune_llm_runs.py \
  --run-names \
  feb23cv1-esde--zai-org--glm-4-9b \
  feb23cv1-cn--zai-org--glm-4-9b \
  --target-run-name feb23cv1-all--zai-org--glm-4-9b
