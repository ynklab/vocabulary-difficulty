#!/bin/bash
#PJM -g gh35
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -N apr18c-sweep-mmbert
#PJM -j
#PJM -o results/finetuned_llm/logs/apr18checkpointpred-sweep_mmbert_base.out

set -euo pipefail

cd "$HOME/bea2026st"
module load miniconda/py39_4.9.2

bash scripts/run_finetune_llm_epoch_calibration_sweep.sh \
  --config-name mar26-mmbert-ep16-cnesde \
  --model-name jhu-clsp/mmBERT-base \
  --conda-env llm_plus \
  --epochs 16 \
  --languages 'cn es de' \
  --single-language \
  --mlm \
  --token-form bare \
  --loss-type ce_prob \
  --predict prob \
  --no-trust-remote-code

bash scripts/run_finetune_llm_epoch_calibration_sweep.sh \
  --config-name apr18-mmbert-ep16-cnesde-allinone \
  --model-name jhu-clsp/mmBERT-base \
  --conda-env llm_plus \
  --epochs 16 \
  --languages 'cn es de' \
  --all-in-one \
  --mlm \
  --token-form bare \
  --loss-type ce_prob \
  --predict prob \
  --no-trust-remote-code
