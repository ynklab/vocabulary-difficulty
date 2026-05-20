#!/bin/bash
#PJM -g gh35
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -N apr19-mmbert-abl-reg
#PJM -j
#PJM -o results/finetuned_llm/logs/apr19-mmbert-ep16-cnesde-abl-reg.out

set -euo pipefail

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --head-type regression \
  --regression-input-style finetune_components \
  --config-name apr19-mmbert-ep16-cnesde-abl-reg \
  --model-name jhu-clsp/mmBERT-base \
  --final-data \
  --languages cn es de \
  --epochs 16 \
  --batch-size 16 \
  --grad-accum 1 \
  --lr-scheduler constant \
  --learning-rate 3e-5 \
  --weight-decay 0.1 \
  --warmup-ratio 0.1 \
  --results-path results/finetuned_llm/apr19-mmbert-ep16-cnesde-abl-reg.csv \
  --stdout-file results/finetuned_llm/logs/apr19-mmbert-ep16-cnesde-abl-reg.log

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --head-type regression \
  --regression-input-style finetune_components \
  --all-in-one \
  --config-name apr19-mmbert-ep16-cnesde-allinone-abl-reg \
  --model-name jhu-clsp/mmBERT-base \
  --final-data \
  --languages cn es de \
  --epochs 16 \
  --batch-size 16 \
  --grad-accum 1 \
  --lr-scheduler constant \
  --learning-rate 3e-5 \
  --weight-decay 0.1 \
  --warmup-ratio 0.1 \
  --results-path results/finetuned_llm/apr19-mmbert-ep16-cnesde-allinone-abl-reg.csv \
  --stdout-file results/finetuned_llm/logs/apr19-mmbert-ep16-cnesde-allinone-abl-reg.log
