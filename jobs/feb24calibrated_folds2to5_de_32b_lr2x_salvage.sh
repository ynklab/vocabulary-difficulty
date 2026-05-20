#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=4
#PJM -L elapse=4:30:00
#PJM -N ftllm_feb24calibrated_folds2to5_de_32b_lr2x_salvage
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_feb24calibrated_folds2to5_de_32b_lr2x_salvage.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

FOLDS=2-5 conda run --no-capture-output -n llm_newer \
  bash scripts/run_finetune_llm_cv_parallel.sh \
  -v \
  --config-name feb24calibrated-de32b-lr2x \
  --model-name zai-org/GLM-4-32B-Base-0414 \
  --languages de \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr-scheduler constant \
  --learning-rate 2e-4 \
  --loss-type ce_prob \
  --predict prob \
  --calibrate \
  --trust-remote-code \
  --results-path results/finetuned_llm/feb24calibrated-de32b-lr2x.csv \
  --stdout-file results/finetuned_llm/logs/feb24calibrated-de32b-lr2x.log
