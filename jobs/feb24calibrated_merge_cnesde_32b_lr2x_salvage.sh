#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=0:30:00
#PJM -N ftllm_feb24calibrated_merge_cnesde_32b_lr2x_salvage
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_feb24calibrated_merge_cnesde_32b_lr2x_salvage.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

# Merge folds 1..5 within each single-language run.
conda run --no-capture-output -n llm_newer python -u \
  scripts/merge_finetune_llm_cv_results.py \
  --results-path results/finetuned_llm/feb24calibrated-cn32b-lr2x.csv \
  --stdout-file results/finetuned_llm/logs/feb24calibrated-cn32b-lr2x.log \
  --folds 1 2 3 4 5

conda run --no-capture-output -n llm_newer python -u \
  scripts/merge_finetune_llm_cv_results.py \
  --results-path results/finetuned_llm/feb24calibrated-es32b-lr2x.csv \
  --stdout-file results/finetuned_llm/logs/feb24calibrated-es32b-lr2x.log \
  --folds 1 2 3 4 5

conda run --no-capture-output -n llm_newer python -u \
  scripts/merge_finetune_llm_cv_results.py \
  --results-path results/finetuned_llm/feb24calibrated-de32b-lr2x.csv \
  --stdout-file results/finetuned_llm/logs/feb24calibrated-de32b-lr2x.log \
  --folds 1 2 3 4 5

# Merge cn + es + de into one combined run (moves model language dirs).
conda run --no-capture-output -n llm_newer python -u \
  scripts/merge_finetune_llm_runs.py \
  --run-names \
  feb24calibrated-cn32b-lr2x--zai-org--GLM-4-32B-Base-0414 \
  feb24calibrated-es32b-lr2x--zai-org--GLM-4-32B-Base-0414 \
  feb24calibrated-de32b-lr2x--zai-org--GLM-4-32B-Base-0414 \
  --target-run-name feb24calibrated-all32b-lr2x--zai-org--GLM-4-32B-Base-0414
