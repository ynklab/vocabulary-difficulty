import argparse
import re
from pathlib import Path

import pandas as pd

from finetune_llm_spaces import scores_to_active_space
from kvl import ID_COL, L1_CODES, SUBSETS, read_subset


FOLD_DIR_RE = re.compile(r'^fold-(\d+)-of-(\d+)$')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run-names',
        nargs='+',
        required=True,
        help=(
            'One or more finetuned LLM prediction filenames (with or without '
            '.csv), e.g. my-config--my-model or my-config--my-model.csv'
            ),
        )
    parser.add_argument(
        '--collate',
        action='store_true',
        help=(
            'Collate predictions with gold data for one L1, using whatever fold '
            'files are available.'
            ),
        )
    parser.add_argument(
        '--lang',
        choices=L1_CODES,
        help='L1 language for --collate mode (cn, de, es).',
        )
    parser.add_argument(
        '--space',
        choices=['logit', 'probability'],
        default='logit',
        help=(
            'Output prediction space. `probability` applies the same expit '
            'transform used by the finetuning helpers.'
            ),
        )
    parser.add_argument(
        '--predictions-root',
        default='predictions/finetuned_llm',
        help='Root folder containing fold-* directories and merged outputs.',
        )
    parser.add_argument(
        '--output-path',
        help=(
            'Optional output CSV path. In default merge mode this overrides the '
            'default whole-dataset output path.'
            ),
        )
    return parser.parse_args()


def normalize_run_filename(run_name):
    return run_name if run_name.endswith('.csv') else f'{run_name}.csv'


def normalize_run_filenames(run_names):
    out = [normalize_run_filename(run_name) for run_name in run_names]
    if len(set(out)) != len(out):
        raise ValueError(f'Duplicate run names in --run-names: {run_names}')
    return out


def discover_fold_files(predictions_root, run_filename):
    predictions_root = Path(predictions_root)
    matches = []

    for fold_dir in sorted(predictions_root.glob('fold-*-of-*')):
        if not fold_dir.is_dir():
            continue
        m = FOLD_DIR_RE.match(fold_dir.name)
        if m is None:
            continue
        fold_i = int(m.group(1))
        total_folds = int(m.group(2))
        pred_path = fold_dir / run_filename
        if pred_path.exists():
            matches.append((fold_i, total_folds, pred_path))

    if not matches:
        raise FileNotFoundError(
            'No fold prediction files found for '
            f'{run_filename} under {predictions_root}'
            )

    return sorted(matches, key=lambda x: x[0])


def validate_complete_folds(matches):
    total_fold_values = sorted({total_folds for _, total_folds, _ in matches})
    if len(total_fold_values) != 1:
        raise ValueError(
            f'Inconsistent total-fold counts across directories: {total_fold_values}'
            )

    total_folds = total_fold_values[0]
    found_folds = sorted({fold_i for fold_i, _, _ in matches})
    expected_folds = list(range(1, total_folds + 1))
    if found_folds != expected_folds:
        missing = sorted(set(expected_folds) - set(found_folds))
        extra = sorted(set(found_folds) - set(expected_folds))
        raise ValueError(
            'Not all folds are available for merge mode. '
            f'expected={expected_folds}, found={found_folds}, '
            f'missing={missing}, extra={extra}'
            )

    return total_folds


def _apply_space_transform(df, pred_cols, space):
    if space == 'logit':
        return df
    out = df.copy()
    for col in pred_cols:
        out[col] = scores_to_active_space(out[col].to_numpy(), space)
    return out


def read_and_merge_fold_predictions(matches, space):
    frames = []
    common_pred_cols = None

    for fold_i, total_folds, pred_path in matches:
        df = pd.read_csv(pred_path)
        if ID_COL not in df.columns:
            raise ValueError(f'Missing {ID_COL} in {pred_path}')
        pred_cols = [c for c in df.columns if c.endswith('_ftllm_output')]
        if not pred_cols:
            raise ValueError(f'No *_ftllm_output columns found in {pred_path}')

        pred_cols = sorted(pred_cols)
        if common_pred_cols is None:
            common_pred_cols = pred_cols
        elif pred_cols != common_pred_cols:
            raise ValueError(
                f'Prediction columns differ in {pred_path}. '
                f'Expected {common_pred_cols}, found {pred_cols}'
                )

        out = df[[ID_COL, *pred_cols]].copy()
        out = _apply_space_transform(out, pred_cols, space)
        out['fold'] = fold_i
        out['total_folds'] = total_folds
        frames.append(out)
        print(f'Read: {pred_path} ({len(out)} row(s))')

    merged = pd.concat(frames, ignore_index=True)
    if merged[ID_COL].duplicated().any():
        dup_ids = merged.loc[merged[ID_COL].duplicated(), ID_COL].tolist()[:10]
        raise ValueError(
            'Duplicate item_id values across fold files. '
            f'Examples: {dup_ids}'
            )

    return merged


def read_and_merge_runs(run_filenames, predictions_root, space):
    merged_by_run = {}
    total_folds_by_run = {}

    for run_filename in run_filenames:
        matches = discover_fold_files(predictions_root, run_filename)
        total_folds_by_run[run_filename] = validate_complete_folds(matches)
        merged_by_run[run_filename] = read_and_merge_fold_predictions(matches, space)

    total_fold_values = sorted(set(total_folds_by_run.values()))
    if len(total_fold_values) != 1:
        raise ValueError(
            f'Inconsistent total folds across runs: {total_folds_by_run}'
            )

    merged_out = None
    seen_pred_cols = set()
    for run_filename in run_filenames:
        run_df = merged_by_run[run_filename].copy()
        pred_cols = [c for c in run_df.columns if c.endswith('_ftllm_output')]
        overlap = sorted(set(pred_cols) & seen_pred_cols)
        if overlap:
            raise ValueError(
                f'Duplicate prediction column(s) across runs: {overlap}. '
                f'Conflict found in {run_filename}'
                )
        seen_pred_cols.update(pred_cols)

        keep_cols = [ID_COL, 'fold', 'total_folds', *pred_cols]
        run_df = run_df[keep_cols]

        if merged_out is None:
            merged_out = run_df
            continue

        left_meta = merged_out[[ID_COL, 'fold', 'total_folds']].copy()
        right_meta = run_df[[ID_COL, 'fold', 'total_folds']].copy()
        if not left_meta.equals(right_meta):
            left_sorted = left_meta.sort_values([ID_COL]).reset_index(drop=True)
            right_sorted = right_meta.sort_values([ID_COL]).reset_index(drop=True)
            if not left_sorted.equals(right_sorted):
                raise ValueError(
                    f'Fold/item alignment differs across runs (e.g. {run_filename})'
                    )

        merged_out = merged_out.merge(
            run_df[[ID_COL, *pred_cols]],
            on=ID_COL,
            how='inner',
            validate='one_to_one',
            )

    if merged_out is None:
        raise ValueError('No runs were provided')

    return merged_out, total_fold_values[0]


def read_gold_all():
    return pd.concat([read_subset('train'), read_subset('dev')], ignore_index=True)


def read_original_subsets():
    subset2df = {subset: read_subset(subset) for subset in SUBSETS}
    subset2ids = {
        subset: set(df[ID_COL].tolist())
        for subset, df in subset2df.items()
        }
    return subset2df, subset2ids


def default_collate_output_path(predictions_root, run_filename, lang):
    stem = Path(run_filename).stem
    return Path(predictions_root) / 'collated' / f'{stem}--{lang}.csv'


def default_whole_output_path(predictions_root, run_filename):
    return Path(predictions_root) / 'whole' / run_filename


def run_collate_mode(args, run_filename, merged_preds):
    if not args.lang:
        raise ValueError('--lang is required with --collate')

    pred_col = f'{args.lang}_ftllm_output'
    if pred_col not in merged_preds.columns:
        available = [
            c for c in merged_preds.columns if c.endswith('_ftllm_output')
            ]
        raise ValueError(
            f'Language column {pred_col} not found in merged predictions. '
            f'Available columns: {available}'
            )

    pred_df = merged_preds[[ID_COL, pred_col]].copy()
    pred_df.rename(columns={pred_col: 'predicted_value'}, inplace=True)

    gold = read_gold_all()
    gold_col = f'{args.lang}_GLMM_score'
    gold_df = gold[[ID_COL, 'en_target_word', gold_col]].copy()
    gold_df.rename(columns={gold_col: 'target_value'}, inplace=True)

    out = pred_df.merge(gold_df, on=ID_COL, how='left')
    if out['target_value'].isna().any():
        missing = out.loc[out['target_value'].isna(), ID_COL].tolist()[:10]
        raise ValueError(
            'Prediction item_id(s) missing from gold data. '
            f'Examples: {missing}'
            )

    out = out[[ID_COL, 'en_target_word', 'target_value', 'predicted_value']]
    out.sort_values(ID_COL, inplace=True)
    out.reset_index(drop=True, inplace=True)

    out_path = (
        Path(args.output_path) if args.output_path
        else default_collate_output_path(args.predictions_root, run_filename, args.lang)
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f'Wrote collated predictions: {out_path} ({len(out)} row(s))')


def run_merge_mode(args, run_filename, merged_preds, total_folds):
    pred_cols = [c for c in merged_preds.columns if c.endswith('_ftllm_output')]

    subset2df, subset2ids = read_original_subsets()
    all_gold_ids = subset2ids['train'].union(subset2ids['dev'])
    pred_ids = set(merged_preds[ID_COL].tolist())

    missing_ids = sorted(all_gold_ids - pred_ids)
    extra_ids = sorted(pred_ids - all_gold_ids)
    if missing_ids or extra_ids:
        raise ValueError(
            'Merged predictions do not match original train+dev item ids. '
            f'missing={missing_ids[:10]} extra={extra_ids[:10]}'
            )

    base_df = merged_preds[[ID_COL, *pred_cols]].copy()
    out = base_df.sort_values(ID_COL).set_index(ID_COL)
    out_path = (
        Path(args.output_path) if args.output_path
        else default_whole_output_path(args.predictions_root, run_filename)
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path)
    print(
        f'Wrote merged predictions: {out_path} '
        f'({len(out)} row(s), complete {total_folds} folds)'
        )


def main():
    args = parse_args()
    run_filenames = normalize_run_filenames(args.run_names)
    merged_preds, total_folds = read_and_merge_runs(
        run_filenames=run_filenames,
        predictions_root=args.predictions_root,
        space=args.space,
        )
    run_filename = run_filenames[0] if len(run_filenames) == 1 else (
        f'combined--{"__".join(Path(name).stem for name in run_filenames)}.csv'
        )

    if args.collate:
        run_collate_mode(args, run_filename, merged_preds)
        return

    if args.lang is not None:
        raise ValueError('--lang is only valid with --collate')
    run_merge_mode(args, run_filename, merged_preds, total_folds)


if __name__ == '__main__':
    main()
