import argparse
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from merge_finetune_llm_predictions import (
    default_whole_output_path,
    discover_fold_files,
    normalize_run_filenames,
    read_and_merge_fold_predictions,
    run_merge_mode,
    validate_complete_folds,
    )
from kvl import ID_COL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run-names',
        nargs='+',
        required=True,
        help='Source run names (with or without .csv suffix).',
        )
    parser.add_argument(
        '--target-run-name',
        required=True,
        help='Target merged run name (with or without .csv suffix).',
        )
    parser.add_argument(
        '--models-root',
        default='models',
        help='Root directory containing per-run adapter folders.',
        )
    parser.add_argument(
        '--results-root',
        default='results/finetuned_llm',
        help='Root directory for per-run metrics CSVs.',
        )
    parser.add_argument(
        '--logs-root',
        default='results/finetuned_llm/logs',
        help='Root directory for per-run stdout logs.',
        )
    parser.add_argument(
        '--predictions-root',
        default='predictions/finetuned_llm',
        help='Root directory containing fold prediction files.',
        )
    parser.add_argument(
        '--space',
        choices=['logit', 'probability'],
        default='logit',
        help='Prediction space used when rebuilding merged whole outputs.',
        )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate inputs and print planned outputs without writing/moving.',
        )
    return parser.parse_args()


def _run_stem(run_name):
    return Path(run_name).stem


def _config_stem_from_run_name(run_name):
    return _run_stem(run_name).split('--', 1)[0]


def _resolve_existing_file(candidates, label):
    for path in candidates:
        if path.exists():
            return path
    tried = ', '.join(str(p) for p in candidates)
    raise FileNotFoundError(f'Missing {label}. Tried: {tried}')


def merge_metrics_csvs(run_names, target_run_name, results_root, dry_run=False):
    results_root = Path(results_root)
    frames = []
    for run_name in run_names:
        src = _resolve_existing_file(
            [
                results_root / f'{_run_stem(run_name)}.csv',
                results_root / f'{_config_stem_from_run_name(run_name)}.csv',
                ],
            label='results CSV',
            )
        df = pd.read_csv(src)
        df['_source_run'] = _run_stem(run_name)
        frames.append(df)
        print(f'Read metrics: {src} ({len(df)} row(s))')

    out = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ['fold', 'lang'] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    out.drop(columns=['_source_run'], inplace=True, errors='ignore')

    dst = results_root / f'{_run_stem(target_run_name)}.csv'
    if dry_run:
        print(f'[dry-run] Would write merged metrics: {dst} ({len(out)} row(s))')
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dst, index=False)
        print(f'Wrote merged metrics: {dst} ({len(out)} row(s))')
    return dst


def merge_logs(run_names, target_run_name, logs_root, dry_run=False):
    logs_root = Path(logs_root)
    dst = logs_root / f'{_run_stem(target_run_name)}.log'
    log_chunks = []
    for run_name in run_names:
        src = _resolve_existing_file(
            [
                logs_root / f'{_run_stem(run_name)}.log',
                logs_root / f'{_config_stem_from_run_name(run_name)}.log',
                ],
            label='stdout log',
            )
        text = src.read_text(encoding='utf-8')
        log_chunks.append((src, text))
        print(f'Read log: {src}')

    if dry_run:
        print(f'[dry-run] Would write merged log: {dst}')
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open('w', encoding='utf-8') as out_f:
            for src, text in log_chunks:
                out_f.write(f'===== run {src.stem} =====\n')
                out_f.write(text)
                out_f.write('\n')
        print(f'Wrote merged log: {dst}')
    return dst


def merge_models_move_scope_dirs(
    run_names,
    target_run_name,
    models_root,
    dry_run=False,
    ):
    models_root = Path(models_root)
    target_root = models_root / _run_stem(target_run_name)
    moved = 0

    for run_name in run_names:
        src_root = models_root / _run_stem(run_name)
        if not src_root.exists():
            raise FileNotFoundError(f'Missing model run directory: {src_root}')
        print(f'Merging model directories from: {src_root}')

        for fold_dir in sorted(src_root.glob('fold_*')):
            if not fold_dir.is_dir():
                continue
            for scope_dir in sorted(fold_dir.iterdir()):
                if not scope_dir.is_dir():
                    continue
                target_fold_dir = target_root / fold_dir.name
                dst_scope_dir = target_fold_dir / scope_dir.name
                if dst_scope_dir.exists():
                    raise FileExistsError(
                        f'Target adapter scope dir already exists: {dst_scope_dir}'
                        )
                if dry_run:
                    print(
                        f'[dry-run] Would move model dir: '
                        f'{scope_dir} -> {dst_scope_dir}'
                        )
                else:
                    target_fold_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(scope_dir), str(dst_scope_dir))
                    print(f'Moved model dir: {scope_dir} -> {dst_scope_dir}')
                moved += 1
    action = 'Would move' if dry_run else 'Moved'
    print(f'{action} {moved} model adapter scope directories into {target_root}')
    return target_root


def _merge_two_prediction_frames(left_df, right_df, right_path):
    left_cols = [c for c in left_df.columns if c.endswith('_ftllm_output')]
    right_cols = [c for c in right_df.columns if c.endswith('_ftllm_output')]
    overlap = sorted(set(left_cols) & set(right_cols))
    if overlap:
        raise ValueError(
            f'Duplicate prediction column(s) across runs in {right_path}: {overlap}'
            )
    return left_df.merge(
        right_df[[ID_COL, *right_cols]],
        on=ID_COL,
        how='inner',
        validate='one_to_one',
        )


def merge_fold_prediction_files(
    run_names,
    target_run_name,
    predictions_root,
    dry_run=False,
    ):
    predictions_root = Path(predictions_root)
    source_files = {}
    total_folds_by_run = {}

    for run_name in run_names:
        run_filename = f'{_run_stem(run_name)}.csv'
        matches = discover_fold_files(predictions_root, run_filename)
        total_folds_by_run[run_filename] = validate_complete_folds(matches)
        source_files[run_filename] = {
            fold_i: pred_path
            for fold_i, _, pred_path in matches
            }

    total_fold_values = sorted(set(total_folds_by_run.values()))
    if len(total_fold_values) != 1:
        raise ValueError(
            f'Inconsistent total folds across prediction runs: {total_folds_by_run}'
            )
    total_folds = total_fold_values[0]
    expected_folds = list(range(1, total_folds + 1))

    target_filename = f'{_run_stem(target_run_name)}.csv'
    merged_fold_frames = []
    for fold_i in expected_folds:
        fold_frames = []
        out_path = None
        for idx, run_filename in enumerate(source_files):
            pred_path = source_files[run_filename][fold_i]
            df = pd.read_csv(pred_path)
            if ID_COL not in df.columns:
                raise ValueError(f'Missing {ID_COL} in {pred_path}')
            pred_cols = [c for c in df.columns if c.endswith('_ftllm_output')]
            if not pred_cols:
                raise ValueError(f'No *_ftllm_output columns found in {pred_path}')
            if df[ID_COL].duplicated().any():
                raise ValueError(f'Duplicate {ID_COL} in {pred_path}')
            df = df[[ID_COL, *sorted(pred_cols)]].copy()
            fold_frames.append((pred_path, df))
            if idx == 0:
                out_path = pred_path.parent / target_filename

        merged_df = fold_frames[0][1]
        for pred_path, df in fold_frames[1:]:
            merged_df = _merge_two_prediction_frames(merged_df, df, pred_path)

        merged_df.sort_values(ID_COL, inplace=True)
        merged_fold_frames.append(merged_df.copy())
        if dry_run:
            print(
                f'[dry-run] Would write merged fold predictions: '
                f'{out_path} ({len(merged_df)} row(s))'
                )
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_csv(out_path, index=False)
            print(
                f'Wrote merged fold predictions: {out_path} '
                f'({len(merged_df)} row(s))'
                )

    merged_preview = pd.concat(merged_fold_frames, ignore_index=True)
    return target_filename, total_folds, merged_preview


def rebuild_prediction_whole_output(
    target_run_name,
    predictions_root,
    space,
    dry_run=False,
    merged_preds_override=None,
    total_folds_override=None,
    ):
    run_filename = f'{_run_stem(target_run_name)}.csv'
    if merged_preds_override is not None:
        merged_preds = merged_preds_override
        total_folds = total_folds_override
    else:
        matches = discover_fold_files(predictions_root, run_filename)
        total_folds = validate_complete_folds(matches)
        merged_preds = read_and_merge_fold_predictions(matches, space=space)
    if dry_run:
        pred_cols = [c for c in merged_preds.columns if c.endswith('_ftllm_output')]
        print(
            f'[dry-run] Validated merged fold predictions for {run_filename} '
            f'({len(merged_preds)} row(s), cols={pred_cols}, folds={total_folds})'
            )
        out_path = default_whole_output_path(predictions_root, run_filename)
        print(f'[dry-run] Would write whole predictions: {out_path}')
        return
    args = SimpleNamespace(
        output_path=None,
        predictions_root=str(predictions_root),
        )
    run_merge_mode(args, run_filename, merged_preds, total_folds)


def main():
    args = parse_args()
    run_names = normalize_run_filenames(args.run_names)
    target_run_name = normalize_run_filenames([args.target_run_name])[0]

    if target_run_name in set(run_names):
        raise ValueError('--target-run-name must differ from source --run-names')

    merge_metrics_csvs(
        run_names,
        target_run_name,
        args.results_root,
        dry_run=args.dry_run,
        )
    merge_logs(run_names, target_run_name, args.logs_root, dry_run=args.dry_run)
    _, merged_total_folds, merged_preds_preview = merge_fold_prediction_files(
        run_names,
        target_run_name,
        args.predictions_root,
        dry_run=args.dry_run,
        )
    rebuild_prediction_whole_output(
        target_run_name=target_run_name,
        predictions_root=args.predictions_root,
        space=args.space,
        dry_run=args.dry_run,
        merged_preds_override=merged_preds_preview if args.dry_run else None,
        total_folds_override=merged_total_folds if args.dry_run else None,
        )
    merge_models_move_scope_dirs(
        run_names,
        target_run_name,
        args.models_root,
        dry_run=args.dry_run,
        )


if __name__ == '__main__':
    main()
