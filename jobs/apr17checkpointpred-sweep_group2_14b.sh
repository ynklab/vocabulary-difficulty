#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -N apr17c-sweep-g2-14b
#PJM -j
#PJM -o results/finetuned_llm/logs/apr17checkpointpred-sweep_group2_14b.out

set -euo pipefail

cd "$HOME/bea2026st"
module load miniconda/py39_4.9.2

bash scripts/run_finetune_llm_epoch_calibration_sweep.sh \
  --config-name apr16calibrated-Ministral-3-14B-abl-raft \
  --model-name mistralai/Ministral-3-14B-Base-2512 \
  --conda-env llm_plus \
  --epochs 4

bash scripts/run_finetune_llm_epoch_calibration_sweep.sh \
  --config-name mar20calibrated-Ministral-3-14B-Base-allinone-lr1x-full \
  --model-name mistralai/Ministral-3-14B-Base-2512 \
  --conda-env llm_plus \
  --epochs 4
