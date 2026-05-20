#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=1:30:00
#PJM -N ftllm_feb24calibrated-Ministral-3-8B-Base_merge_all
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_feb24calibrated-Ministral-3-8B-Base_merge_all.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u \
  scripts/merge_finetune_llm_cv_results.py \
  --results-path results/finetuned_llm/feb24calibrated-Ministral-3-8B-Base-es.csv \
  --stdout-file results/finetuned_llm/logs/feb24calibrated-Ministral-3-8B-Base-es.log \
  --folds 1 2 3 4 5

conda run --no-capture-output -n llm_plus python -u \
  scripts/merge_finetune_llm_cv_results.py \
  --results-path results/finetuned_llm/feb24calibrated-Ministral-3-8B-Base-de.csv \
  --stdout-file results/finetuned_llm/logs/feb24calibrated-Ministral-3-8B-Base-de.log \
  --folds 1 2 3 4 5

conda run --no-capture-output -n llm_plus python -u \
  scripts/merge_finetune_llm_cv_results.py \
  --results-path results/finetuned_llm/feb24calibrated-Ministral-3-8B-Base-cn.csv \
  --stdout-file results/finetuned_llm/logs/feb24calibrated-Ministral-3-8B-Base-cn.log \
  --folds 1 2 3 4 5

# Optional dry run for the final combined merge (uncomment to validate first):
# conda run --no-capture-output -n llm_plus python -u \
#   scripts/merge_finetune_llm_runs.py \
#   --dry-run \
#   --run-names \
#   feb24calibrated-Ministral-3-8B-Base-es--mistralai--Ministral-3-8B-Base-2512 \
#   feb24calibrated-Ministral-3-8B-Base-de--mistralai--Ministral-3-8B-Base-2512 \
#   feb24calibrated-Ministral-3-8B-Base-cn--mistralai--Ministral-3-8B-Base-2512 \
#   --target-run-name feb24calibrated-Ministral-3-8B-Base-all--mistralai--Ministral-3-8B-Base-2512

conda run --no-capture-output -n llm_plus python -u \
  scripts/merge_finetune_llm_runs.py \
  --run-names \
  feb24calibrated-Ministral-3-8B-Base-es--mistralai--Ministral-3-8B-Base-2512 \
  feb24calibrated-Ministral-3-8B-Base-de--mistralai--Ministral-3-8B-Base-2512 \
  feb24calibrated-Ministral-3-8B-Base-cn--mistralai--Ministral-3-8B-Base-2512 \
  --target-run-name feb24calibrated-Ministral-3-8B-Base-all--mistralai--Ministral-3-8B-Base-2512
