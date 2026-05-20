#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=2
#PJM -L elapse=8:00:00
#PJM -N ftllm_feb24calibrated-GLM-4-32B-Chat_folds2to3_cn
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_feb24calibrated-GLM-4-32B-Chat_folds2to3_cn.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

FOLDS=2-3 conda run --no-capture-output -n llm_newer \
  bash scripts/run_finetune_llm_cv_parallel.sh \
  -v \
  --config-name feb24calibrated-GLM-4-32B-Chat-cn \
  --model-name zai-org/GLM-4-32B-0414 \
  --languages cn \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr-scheduler constant \
  --learning-rate 2e-4 \
  --loss-type ce_prob \
  --predict prob \
  --calibrate \
  --trust-remote-code \
  --results-path results/finetuned_llm/feb24calibrated-GLM-4-32B-Chat-cn.csv \
  --stdout-file results/finetuned_llm/logs/feb24calibrated-GLM-4-32B-Chat-cn.log
