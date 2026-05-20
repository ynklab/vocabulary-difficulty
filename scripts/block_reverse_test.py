import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LANGS = ('cn', 'de', 'es')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Build block-reversed test predictions by language from full '
            'train+dev+test GLMM_score rankings.'
            )
        )
    parser.add_argument(
        '--block-size',
        '-b',
        type=int,
        default=100,
        help=(
            'Default block size for score reversal after sorting by GLMM_score. '
            'Used unless language-specific block size overrides are supplied.'
            ),
        )
    parser.add_argument(
        '--cn-block-size',
        '-cn-block-size',
        '--cnb',
        type=int,
        help='Chinese block size override.',
        )
    parser.add_argument(
        '--de-block-size',
        '-de-block-size',
        '--deb',
        type=int,
        help='German block size override.',
        )
    parser.add_argument(
        '--es-block-size',
        '-es-block-size',
        '--esb',
        type=int,
        help='Spanish block size override.',
        )
    parser.add_argument(
        '--output',
        '-o',
        required=True,
        help='Output CSV path for combined test predictions.',
        )
    parser.add_argument(
        '--worst-case', '-w',
        action='store_true',
        help=(
            'Use local worst-case assignment instead of block reversal. '
            'For each sorted item i, selects the value within index range '
            '[i - block_size, i + block_size] that maximizes absolute '
            'difference from the current value.'
            ),
        )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help=(
            'Print per-language rank-gap statistics for score delta = 1.0 '
            'on full train+dev+test data.'
            ),
        )
    return parser.parse_args()


def load_split_scores(lang, split):
    path = f'data/{split}/{lang}/kvl_shared_task_{lang}_{split}.csv'
    return pd.read_csv(path, usecols=['item_id', 'GLMM_score'])


def reverse_scores_in_blocks(df, block_size):
    sorted_df = df.sort_values('GLMM_score', ascending=True).reset_index(drop=True)
    reversed_scores = sorted_df['GLMM_score'].copy()

    for i in range(0, len(sorted_df), block_size):
        block = reversed_scores.iloc[i:i + block_size].to_numpy(copy=True)
        reversed_scores.iloc[i:i + block_size] = block[::-1]

    sorted_df = sorted_df[['item_id']].copy()
    sorted_df['reversed_GLMM_score'] = reversed_scores
    return sorted_df


def worst_case_scores_in_window(df, block_size):
    sorted_df = df.sort_values('GLMM_score', ascending=True).reset_index(drop=True)
    scores = sorted_df['GLMM_score'].to_numpy(copy=True)
    n = len(scores)
    worst_case_scores = np.empty(n, dtype=float)

    for i in range(n):
        lo = max(0, i - block_size)
        hi = min(n - 1, i + block_size)
        current = scores[i]
        lo_diff = abs(scores[lo] - current)
        hi_diff = abs(scores[hi] - current)
        worst_case_scores[i] = scores[lo] if lo_diff >= hi_diff else scores[hi]

    sorted_df = sorted_df[['item_id']].copy()
    sorted_df['reversed_GLMM_score'] = worst_case_scores
    return sorted_df


def rank_diff_stats_for_score_delta(df, delta=1.0):
    sorted_scores = np.sort(df['GLMM_score'].to_numpy(dtype=float))
    n = len(sorted_scores)
    rank_diffs = []

    for i, score in enumerate(sorted_scores):
        upper_idx = np.searchsorted(sorted_scores, score + delta, side='left')
        lower_idx = np.searchsorted(sorted_scores, score - delta, side='right') - 1

        gaps = []
        if upper_idx < n:
            gaps.append(float(abs(upper_idx - i)))
        if lower_idx >= 0:
            gaps.append(float(abs(i - lower_idx)))
        if gaps:
            rank_diffs.append(float(np.mean(gaps)))

    if not rank_diffs:
        return None

    rank_diffs_arr = np.asarray(rank_diffs, dtype=float)
    sd = float(rank_diffs_arr.std(ddof=1)) if len(rank_diffs_arr) > 1 else 0.0
    return {
        'min': float(rank_diffs_arr.min()),
        'max': float(rank_diffs_arr.max()),
        'median': float(np.median(rank_diffs_arr)),
        'mean': float(rank_diffs_arr.mean()),
        'sd': sd,
        }


def build_lang_predictions(lang, block_size, worst_case, verbose):
    train_df = load_split_scores(lang, 'train')
    dev_df = load_split_scores(lang, 'dev')
    test_df = load_split_scores(lang, 'test')

    full_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    if verbose:
        stats = rank_diff_stats_for_score_delta(full_df, delta=1.0)
        if stats is None:
            print(f'{lang} rank-diff(delta=1): no valid statistics')
        else:
            print(
                f'{lang} rank-diff(delta=1): '
                f'min={stats["min"]:.3f}, '
                f'max={stats["max"]:.3f}, '
                f'median={stats["median"]:.3f}, '
                f'mean={stats["mean"]:.3f}, '
                f'sd={stats["sd"]:.3f}'
                )
    reversed_df = (
        worst_case_scores_in_window(full_df, block_size)
        if worst_case
        else reverse_scores_in_blocks(full_df, block_size)
        )

    test_item_ids = test_df[['item_id']].copy()
    merged = test_item_ids.merge(
        reversed_df,
        on='item_id',
        how='left',
        validate='one_to_one'
        )

    if merged['reversed_GLMM_score'].isna().any():
        raise ValueError(f'Missing reversed score(s) for test items in {lang}')

    return merged.rename(columns={'reversed_GLMM_score': f'{lang}_ftllm_output'})


def main():
    args = parse_args()
    if args.block_size <= 0:
        raise ValueError('--block-size must be > 0')

    lang_block_sizes = {
        'cn': args.cn_block_size if args.cn_block_size is not None else args.block_size,
        'de': args.de_block_size if args.de_block_size is not None else args.block_size,
        'es': args.es_block_size if args.es_block_size is not None else args.block_size,
        }

    for lang, block_size in lang_block_sizes.items():
        if block_size <= 0:
            raise ValueError(f'Block size for {lang} must be > 0')

    cn_df = build_lang_predictions(
        'cn',
        lang_block_sizes['cn'],
        args.worst_case,
        args.verbose
        )
    de_df = build_lang_predictions(
        'de',
        lang_block_sizes['de'],
        args.worst_case,
        args.verbose
        )
    es_df = build_lang_predictions(
        'es',
        lang_block_sizes['es'],
        args.worst_case,
        args.verbose
        )

    out = cn_df.merge(de_df, on='item_id', how='inner', validate='one_to_one')
    out = out.merge(es_df, on='item_id', how='inner', validate='one_to_one')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f'Wrote block-reversed test predictions: {args.output}')


if __name__ == '__main__':
    main()
