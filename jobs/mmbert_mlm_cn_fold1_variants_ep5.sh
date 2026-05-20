#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=36:00:00
#PJM -N ftllm_mlm_mmbert_cn_f1_vars
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_mlm_mmbert_cn_fold1_variants_ep5.out

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

LOG_DIR='results/finetuned_llm/logs'
mkdir -p "$LOG_DIR"

run_variant() {
  local run_name="$1"
  shift
  local log_path="$LOG_DIR/${run_name}.log"
  echo "=== START ${run_name} ==="
  conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
    -v \
    --mlm \
    --config-name "$run_name" \
    --model-name jhu-clsp/mmBERT-base \
    --folds 1 \
    --languages cn \
    --epochs 5 \
    --lr-scheduler constant \
    --learning-rate 3e-5 \
    --weight-decay 0.1 \
    --warmup-ratio 0.1 \
    --loss-type ce_prob \
    --predict prob \
    --token-form bare \
    --calibrate \
    --results-path "results/finetuned_llm/${run_name}.csv" \
    --stdout-file "$log_path" \
    "$@"
  echo "=== END ${run_name} ==="
}

# var0: no additional changes from baseline.
run_variant mmbert-mlm-cn-fold1-var0-ep5-lr3e5-wd0p1-wu0p1 \
  --batch-size 8 \
  --grad-accum 2

# var1: batch size 32, no grad accumulation.
run_variant mmbert-mlm-cn-fold1-var1-ep5-lr3e5-bs32-wd0p1-wu0p1 \
  --batch-size 32 \
  --grad-accum 1

# var2: LR 2e-5.
run_variant mmbert-mlm-cn-fold1-var2-ep5-lr2e5-wd0p1-wu0p1 \
  --batch-size 8 \
  --grad-accum 2 \
  --learning-rate 2e-5

# var3: LR 1e-5.
run_variant mmbert-mlm-cn-fold1-var3-ep5-lr1e5-wd0p1-wu0p1 \
  --batch-size 8 \
  --grad-accum 2 \
  --learning-rate 1e-5

echo
echo '===== CONCATENATED VARIANT LOGS ====='
for log_file in \
  "$LOG_DIR"/mmbert-mlm-cn-fold1-var0-ep5-lr3e5-wd0p1-wu0p1.log \
  "$LOG_DIR"/mmbert-mlm-cn-fold1-var1-ep5-lr3e5-bs32-wd0p1-wu0p1.log \
  "$LOG_DIR"/mmbert-mlm-cn-fold1-var2-ep5-lr2e5-wd0p1-wu0p1.log \
  "$LOG_DIR"/mmbert-mlm-cn-fold1-var3-ep5-lr1e5-wd0p1-wu0p1.log
do
  echo
  echo "===== ${log_file} ====="
  cat "$log_file"
done
