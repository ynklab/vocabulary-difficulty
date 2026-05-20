import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from kvl import ID_COL, L1_CODES
from merge_finetune_llm_predictions import (
    discover_fold_files,
    read_and_merge_fold_predictions,
    run_merge_mode,
    validate_complete_folds,
    )


FOLD_DIR_RE = re.compile(r'^fold_(\d+)$')
DEFAULT_LANG_ORDER = ['cn', 'es', 'de']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run-name',
        required=True,
        help='Run name used under models/ (with or without .csv suffix).',
        )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help=(
            'Checkpoint selector: exact tag (e.g. epoch_001p000) '
            'or 1-based lexical index (e.g. 1).'
            ),
        )
    parser.add_argument(
        '--scope',
        choices=['all', 'lang'],
        default='all',
        help='Adapter scope used during finetuning.',
        )
    parser.add_argument(
        '--languages',
        nargs='+',
        choices=L1_CODES,
        default=DEFAULT_LANG_ORDER,
        help='Languages to include in merged prediction columns.',
        )
    parser.add_argument(
        '--models-root',
        default='models',
        help='Root containing per-run model artifacts.',
        )
    parser.add_argument(
        '--predictions-root',
        default='predictions/finetuned_llm',
        help='Root for fold-wise and merged prediction CSVs.',
        )
    parser.add_argument(
        '--merge',
        action='store_true',
        help='Also run standard cross-fold merge into whole output.',
        )
    parser.add_argument(
        '--space',
        choices=['logit', 'probability'],
        default='logit',
        help='Output prediction space for --merge mode.',
        )
    parser.add_argument(
        '--output-path',
        help=(
            'Optional single merged output path for --merge mode. '
            'If omitted, writes to predictions-root/{train,dev}/...'
            ),
        )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite fold prediction files if they already exist.',
        )
    return parser.parse_args()


def normalize_run_stem(run_name):
    return Path(run_name).stem


def resolve_fold_dirs(run_root):
    folds = []
    for path in sorted(run_root.glob('fold_*')):
        if not path.is_dir():
            continue
        match = FOLD_DIR_RE.match(path.name)
        if not match:
            continue
        folds.append((int(match.group(1)), path))
    if not folds:
        raise FileNotFoundError(f'No fold_* directories found under {run_root}')
    return folds


def list_checkpoint_tags(scope_dir):
    checkpoint_root = scope_dir / 'epoch_checkpoints'
    if not checkpoint_root.exists():
        return []
    return sorted(
        p.name for p in checkpoint_root.iterdir() if p.is_dir()
        )


def resolve_checkpoint_tag(scope_dir, checkpoint_selector):
    tags = list_checkpoint_tags(scope_dir)
    if not tags:
        raise FileNotFoundError(
            f'No checkpoints found under {scope_dir / "epoch_checkpoints"}'
            )
    selector = str(checkpoint_selector)
    if selector.isdigit():
        idx = int(selector)
        if idx <= 0 or idx > len(tags):
            raise ValueError(
                '--checkpoint index out of range '
                f'({idx}); available 1..{len(tags)}: {tags}'
                )
        return tags[idx - 1]
    if selector not in tags:
        raise FileNotFoundError(
            f'Checkpoint tag not found: {selector} under {scope_dir}; '
            f'available={tags}'
            )
    return selector


def read_checkpoint_predictions_all(scope_dir, checkpoint_tag, languages):
    pred_path = (
        scope_dir /
        'epoch_checkpoints' /
        checkpoint_tag /
        'dev_predictions.csv'
        )
    if not pred_path.exists():
        raise FileNotFoundError(f'Missing checkpoint predictions: {pred_path}')
    df = pd.read_csv(pred_path)
    needed = {ID_COL, 'prediction', 'lang'}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f'Missing columns in {pred_path}: {missing}')

    out = df[[ID_COL, 'lang', 'prediction']].copy()
    out = out[out['lang'].isin(languages)].copy()
    if out.empty:
        raise ValueError(
            f'No rows for requested languages {languages} in {pred_path}'
            )
    if out[[ID_COL, 'lang']].duplicated().any():
        raise ValueError(f'Duplicate ({ID_COL}, lang) pairs in {pred_path}')

    pivoted = out.pivot(index=ID_COL, columns='lang', values='prediction')
    missing_langs = [lang for lang in languages if lang not in pivoted.columns]
    if missing_langs:
        raise ValueError(
            f'Missing language(s) {missing_langs} in {pred_path}; '
            f'available={sorted(pivoted.columns.tolist())}'
            )

    pivoted = pivoted[languages].copy()
    pivoted.columns = [f'{lang}_ftllm_output' for lang in languages]
    return pivoted.reset_index().sort_values(ID_COL).reset_index(drop=True)


def read_checkpoint_predictions_lang_scope(
    fold_dir,
    checkpoint_selector,
    languages,
    ):
    fold_df = None
    resolved_tag = None

    for lang in languages:
        scope_dir = fold_dir / lang
        if not scope_dir.exists():
            raise FileNotFoundError(f'Missing scope directory: {scope_dir}')

        lang_tag = resolve_checkpoint_tag(scope_dir, checkpoint_selector)
        if resolved_tag is None:
            resolved_tag = lang_tag
        elif lang_tag != resolved_tag:
            raise ValueError(
                f'Checkpoint selector resolves inconsistently in {fold_dir}: '
                f'{resolved_tag} vs {lang_tag}. '
                'Use an explicit --checkpoint tag.'
                )

        pred_path = (
            scope_dir /
            'epoch_checkpoints' /
            resolved_tag /
            'dev_predictions.csv'
            )
        if not pred_path.exists():
            raise FileNotFoundError(f'Missing checkpoint predictions: {pred_path}')

        df = pd.read_csv(pred_path)
        needed = {ID_COL, 'prediction'}
        missing = sorted(needed - set(df.columns))
        if missing:
            raise ValueError(f'Missing columns in {pred_path}: {missing}')
        if df[ID_COL].duplicated().any():
            raise ValueError(f'Duplicate {ID_COL} in {pred_path}')

        one = df[[ID_COL, 'prediction']].copy()
        one.rename(columns={'prediction': f'{lang}_ftllm_output'}, inplace=True)
        fold_df = one if fold_df is None else fold_df.merge(one, on=ID_COL, how='inner')

    return resolved_tag, fold_df.sort_values(ID_COL).reset_index(drop=True)


def write_fold_prediction_file(
    fold_df,
    predictions_root,
    run_filename,
    fold_i,
    total_folds,
    overwrite,
    ):
    out_path = (
        Path(predictions_root) /
        f'fold-{fold_i}-of-{total_folds}' /
        run_filename
        )
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            'Prediction file already exists '
            f'({out_path}). Pass --overwrite to replace it.'
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out_path, index=False)
    print(f'Wrote fold predictions: {out_path} ({len(fold_df)} row(s))')


def main():
    args = parse_args()
    run_stem = normalize_run_stem(args.run_name)
    run_root = Path(args.models_root) / run_stem
    if not run_root.exists():
        raise FileNotFoundError(f'Run directory not found: {run_root}')

    selected_langs = [
        lang for lang in DEFAULT_LANG_ORDER if lang in set(args.languages)
        ]
    if not selected_langs:
        raise ValueError('No languages selected')

    folds = resolve_fold_dirs(run_root)
    total_folds = len(folds)
    resolved_tags = set()

    for fold_i, fold_dir in folds:
        if args.scope == 'all':
            scope_dir = fold_dir / 'all'
            if not scope_dir.exists():
                raise FileNotFoundError(f'Missing scope directory: {scope_dir}')
            checkpoint_tag = resolve_checkpoint_tag(scope_dir, args.checkpoint)
            fold_df = read_checkpoint_predictions_all(
                scope_dir,
                checkpoint_tag,
                selected_langs,
                )
        else:
            checkpoint_tag, fold_df = read_checkpoint_predictions_lang_scope(
                fold_dir,
                args.checkpoint,
                selected_langs,
                )

        resolved_tags.add(checkpoint_tag)
        if len(resolved_tags) > 1:
            raise ValueError(
                f'Checkpoint selector resolved to different tags across folds: '
                f'{sorted(resolved_tags)}. Use an explicit --checkpoint tag.'
                )

        run_filename = f'{run_stem}--{checkpoint_tag}.csv'
        write_fold_prediction_file(
            fold_df=fold_df,
            predictions_root=args.predictions_root,
            run_filename=run_filename,
            fold_i=fold_i,
            total_folds=total_folds,
            overwrite=args.overwrite,
            )

    checkpoint_tag = sorted(resolved_tags)[0]
    run_filename = f'{run_stem}--{checkpoint_tag}.csv'
    print(f'Checkpoint run filename: {run_filename}')

    if not args.merge:
        return

    matches = discover_fold_files(args.predictions_root, run_filename)
    complete_folds = validate_complete_folds(matches)
    merged_preds = read_and_merge_fold_predictions(matches, args.space)
    merge_args = SimpleNamespace(
        predictions_root=args.predictions_root,
        output_path=args.output_path,
        )
    run_merge_mode(merge_args, run_filename, merged_preds, complete_folds)


if __name__ == '__main__':
    main()
