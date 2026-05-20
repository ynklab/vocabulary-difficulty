#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=30:00:00
#PJM -N mar20c-Qwen27B-lr1p5x_full
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_mar20calibrated-Qwen3.5-27B-allinone-lr1p5x_full.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --config-name mar20calibrated-Qwen3.5-27B-allinone-lr1p5x-full \
  --model-name Qwen/Qwen3.5-27B \
  --all-in-one \
  --final-data \
  --languages cn es de \
  --epochs 4 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr-scheduler constant \
  --learning-rate 1.5e-4 \
  --loss-type ce_prob \
  --predict prob \
  --calibrate \
  --trust-remote-code \
  --results-path results/finetuned_llm/mar20calibrated-Qwen3.5-27B-allinone-lr1p5x-full.csv \
  --stdout-file results/finetuned_llm/logs/mar20calibrated-Qwen3.5-27B-allinone-lr1p5x-full.log
