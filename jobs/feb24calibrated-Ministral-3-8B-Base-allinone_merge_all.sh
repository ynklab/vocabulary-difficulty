#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=1:00:00
#PJM -N ftllm_feb24calibrated-Ministral-3-8B-Base-allinone_merge_all
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_feb24calibrated-Ministral-3-8B-Base-allinone_merge_all.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u \
  scripts/merge_finetune_llm_cv_results.py \
  --results-path results/finetuned_llm/feb24calibrated-Ministral-3-8B-Base-allinone.csv \
  --stdout-file results/finetuned_llm/logs/feb24calibrated-Ministral-3-8B-Base-allinone.log \
  --folds 1 2 3 4 5

conda run --no-capture-output -n llm_plus python -u \
  scripts/merge_finetune_llm_predictions.py \
  --run-names feb24calibrated-Ministral-3-8B-Base-allinone--mistralai--Ministral-3-8B-Base-2512
