#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=6:00:00
#PJM -N ftllm_feb24calibrated_fold1_cn_32b_lr2x
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_feb24calibrated_fold1_cn_32b_lr2x.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_newer python -u scripts/finetune_llm.py \
  -v \
  --config-name feb24calibrated-cn32b-lr2x \
  --model-name zai-org/GLM-4-32B-Base-0414 \
  --cv-mode first \
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
  --results-path results/finetuned_llm/feb24calibrated-cn32b-lr2x.csv \
  --stdout-file results/finetuned_llm/logs/feb24calibrated-cn32b-lr2x.log
