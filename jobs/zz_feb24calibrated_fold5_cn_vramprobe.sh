#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=1:00:00
#PJM -N zz_ftllm_feb24calibrated_fold5_cn_vramprobe
#PJM -j
#PJM -o results/finetuned_llm/logs/zz_pj_ftllm_feb24calibrated_fold5_cn_vramprobe.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

FOLDS=5 conda run --no-capture-output -n llm_env \
  bash scripts/run_finetune_llm_cv_parallel.sh \
  -v \
  --config-name zz-feb24calibrated-cn-vramprobe \
  --languages cn \
  --epochs 0.02 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr-scheduler constant \
  --learning-rate 1e-4 \
  --loss-type ce_prob \
  --predict prob \
  --calibrate \
  --trust-remote-code \
  --vram-stats \
  --results-path results/finetuned_llm/zz-feb24calibrated-cn-vramprobe.csv \
  --stdout-file results/finetuned_llm/logs/zz-feb24calibrated-cn-vramprobe.log
