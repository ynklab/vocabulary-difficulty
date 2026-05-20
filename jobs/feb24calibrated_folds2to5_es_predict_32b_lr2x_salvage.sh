#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=4
#PJM -L elapse=3:00:00
#PJM -N ftllm_feb24calibrated_folds2to5_es_predict_32b_lr2x_salvage
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_feb24calibrated_folds2to5_es_predict_32b_lr2x_salvage.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

FOLDS=2-5 conda run --no-capture-output -n llm_newer \
  bash scripts/run_finetune_llm_cv_parallel.sh \
  -v \
  --mode predict \
  --config-name feb24calibrated-es32b-lr2x \
  --model-name zai-org/GLM-4-32B-Base-0414 \
  --languages es \
  --batch-size 2 \
  --loss-type ce_prob \
  --predict prob \
  --calibrate \
  --trust-remote-code \
  --results-path results/finetuned_llm/feb24calibrated-es32b-lr2x.csv \
  --stdout-file results/finetuned_llm/logs/feb24calibrated-es32b-lr2x.log
