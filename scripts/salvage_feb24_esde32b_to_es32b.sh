#!/usr/bin/env bash
set -euo pipefail

# Backup failed esde artifacts with a "_failed" suffix, then rename the
# current esde namespace to es for Spanish-only salvage prediction.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ESDE_CFG='feb24calibrated-esde32b-lr2x'
ES_CFG='feb24calibrated-es32b-lr2x'
MODEL_STEM='zai-org--GLM-4-32B-Base-0414'
ESDE_RUN="${ESDE_CFG}--${MODEL_STEM}"
ES_RUN="${ES_CFG}--${MODEL_STEM}"

copy_with_failed_suffix() {
  local src="$1"
  local dst
  if [[ ! -e "$src" ]]; then
    return 0
  fi
  if [[ -d "$src" ]]; then
    dst="${src}_failed"
    rm -rf "$dst"
    cp -a "$src" "$dst"
  else
    dst="${src%.*}_failed.${src##*.}"
    cp -a "$src" "$dst"
  fi
  echo "Backed up: $src -> $dst"
}

move_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "$src" ]]; then
    return 0
  fi
  if [[ -e "$dst" ]]; then
    echo "Refusing to overwrite existing target: $dst" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  mv "$src" "$dst"
  echo "Moved: $src -> $dst"
}

backup_and_move_file_pattern() {
  local src_glob="$1"
  local from_text="$2"
  local to_text="$3"
  local path dst

  shopt -s nullglob
  for path in $src_glob; do
    copy_with_failed_suffix "$path"
    dst="${path//$from_text/$to_text}"
    move_if_exists "$path" "$dst"
  done
  shopt -u nullglob
}

backup_and_move_file_pattern \
  "results/finetuned_llm/${ESDE_CFG}*.csv" \
  "$ESDE_CFG" \
  "$ES_CFG"

backup_and_move_file_pattern \
  "results/finetuned_llm/logs/${ESDE_CFG}*.log" \
  "$ESDE_CFG" \
  "$ES_CFG"

backup_and_move_file_pattern \
  "results/finetuned_llm/logs/pj_*${ESDE_CFG}*.out" \
  "$ESDE_CFG" \
  "$ES_CFG"

copy_with_failed_suffix "models/${ESDE_RUN}"
move_if_exists "models/${ESDE_RUN}" "models/${ES_RUN}"

shopt -s nullglob
for fold_dir in predictions/finetuned_llm/fold-*-of-*; do
  [[ -d "$fold_dir" ]] || continue
  src_pred="${fold_dir}/${ESDE_RUN}.csv"
  dst_pred="${fold_dir}/${ES_RUN}.csv"
  if [[ -e "$src_pred" ]]; then
    copy_with_failed_suffix "$src_pred"
    move_if_exists "$src_pred" "$dst_pred"
  fi
done
shopt -u nullglob

echo 'Done. esde artifacts were backed up with "_failed" suffix and renamed to es.'
