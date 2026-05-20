#!/bin/bash
#PJM -g gh35
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -N apr27-xlmr-mmbert-ep5-full
#PJM -j
#PJM -o results/finetuned_llm/logs/apr27-xlmr-mmbertsetup-ep5-allinone_full.out

set -euo pipefail

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --mlm \
  --all-in-one \
  --config-name apr27-xlmr-base-mmbertsetup-ep5-allinone \
  --model-name xlm-roberta-base \
  --final-data \
  --languages cn es de \
  --epochs 5 \
  --batch-size 32 \
  --grad-accum 1 \
  --lr-scheduler linear \
  --learning-rate 3e-5 \
  --weight-decay 0.1 \
  --warmup-ratio 0.1 \
  --loss-type ce_prob \
  --predict prob \
  --token-form bare \
  --calibrate \
  --results-path results/finetuned_llm/apr27-xlmr-base-mmbertsetup-ep5-allinone.csv \
  --stdout-file results/finetuned_llm/logs/apr27-xlmr-base-mmbertsetup-ep5-allinone.log

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --mlm \
  --all-in-one \
  --config-name apr27-xlmr-large-mmbertsetup-ep5-allinone \
  --model-name xlm-roberta-large \
  --final-data \
  --languages cn es de \
  --epochs 5 \
  --batch-size 32 \
  --grad-accum 1 \
  --lr-scheduler linear \
  --learning-rate 3e-5 \
  --weight-decay 0.1 \
  --warmup-ratio 0.1 \
  --loss-type ce_prob \
  --predict prob \
  --token-form bare \
  --calibrate \
  --results-path results/finetuned_llm/apr27-xlmr-large-mmbertsetup-ep5-allinone.csv \
  --stdout-file results/finetuned_llm/logs/apr27-xlmr-large-mmbertsetup-ep5-allinone.log
