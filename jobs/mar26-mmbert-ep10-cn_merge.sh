#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=1:00:00
#PJM -N mar26-mmbert-ep10-cn_merge
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_mar26-mmbert-ep10-cn_merge.out

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u \
  scripts/merge_finetune_llm_cv_results.py \
  --results-path results/finetuned_llm/mar26-mmbert-ep10-cn.csv \
  --stdout-file results/finetuned_llm/logs/mar26-mmbert-ep10-cn.log \
  --folds 1 2 3 4 5

conda run --no-capture-output -n llm_plus python -u \
  scripts/merge_finetune_llm_predictions.py \
  --run-names mar26-mmbert-ep10-cn--jhu-clsp--mmBERT-base
