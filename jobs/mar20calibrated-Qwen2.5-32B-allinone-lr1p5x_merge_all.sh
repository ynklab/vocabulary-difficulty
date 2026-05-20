#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=1:00:00
#PJM -N mar20c-Qwen25-32B-lr1p5x_merge
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_mar20calibrated-Qwen2.5-32B-allinone-lr1p5x_merge_all.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_newer python -u \
  scripts/merge_finetune_llm_cv_results.py \
  --results-path results/finetuned_llm/mar20calibrated-Qwen2.5-32B-allinone-lr1p5x.csv \
  --stdout-file results/finetuned_llm/logs/mar20calibrated-Qwen2.5-32B-allinone-lr1p5x.log \
  --folds 1 2 3 4 5

conda run --no-capture-output -n llm_newer python -u \
  scripts/merge_finetune_llm_predictions.py \
  --run-names mar20calibrated-Qwen2.5-32B-allinone-lr1p5x--Qwen--Qwen2.5-32B
