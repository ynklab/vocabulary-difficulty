import argparse
from pathlib import Path

import numpy as np
import pandas as pd


L1_CODES = ('cn', 'de', 'es')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sort SHAP items by diversity and/or prediction accuracy.'
        )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Use shap_detailed.csv.'
        )
    mode.add_argument(
        '--grouped', '-g',
        action='store_true',
        help='Use shap_grouped.csv.'
        )
    parser.add_argument(
        'shap_dir',
        help='SHAP output directory containing shap_detailed.csv/shap_grouped.csv.'
        )
    parser.add_argument(
        '--diverse',
        action='store_true',
        help=('Rank items by cross-language SHAP diversity using pairwise '
              'absolute SHAP differences.')
        )
    parser.add_argument(
        '--accurate',
        nargs='+',
        metavar='L1',
        help=('Rank items by prediction accuracy for selected L1(s). '
              f'Choices: {", ".join(L1_CODES)}, all.')
        )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help=('Output CSV path. Default: <shap_dir>/sorted_items_'
              '<detailed|grouped>.csv')
        )
    return parser.parse_args()


def normalize_01(s):
    s = s.astype(float)
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index, dtype=float)
    lo = valid.min()
    hi = valid.max()
    if hi <= lo:
        out = pd.Series(np.nan, index=s.index, dtype=float)
        out.loc[valid.index] = 1.0
        return out
    return (s - lo) / (hi - lo)


def parse_accurate_l1s(raw_l1s):
    if raw_l1s is None:
        return None
    lowered = [x.lower() for x in raw_l1s]
    if 'all' in lowered:
        if len(lowered) != 1:
            raise ValueError(
                '--accurate all cannot be combined with explicit L1 codes.'
                )
        return list(L1_CODES)
    bad = sorted(set(x for x in lowered if x not in L1_CODES))
    if bad:
        raise ValueError(
            f'Unsupported --accurate L1 code(s): {", ".join(bad)}.'
            )
    # Keep order but unique
    return list(dict.fromkeys(lowered))


def item_text_meta(df):
    meta_cols = ['item_id', 'en_target_word', 'en_target_pos']
    available = [c for c in meta_cols if c in df.columns]
    if 'item_id' not in available:
        return pd.DataFrame(columns=meta_cols)
    meta = df[available].drop_duplicates(subset=['item_id']).copy()
    for c in meta_cols:
        if c not in meta:
            meta[c] = ''
    return meta[meta_cols]


def diverse_scores(df, key_col):
    wide = df.pivot_table(
        index=['item_id', key_col],
        columns='lang',
        values='shap_value',
        aggfunc='mean'
        )
    wide = wide.reindex(columns=list(L1_CODES))
    complete = wide.dropna()
    if complete.empty:
        return pd.Series(dtype=float, name='diverse_score')
    diffs = pd.DataFrame({
        'cn_de': (complete['cn'] - complete['de']).abs(),
        'cn_es': (complete['cn'] - complete['es']).abs(),
        'de_es': (complete['de'] - complete['es']).abs()
        })
    row_div = diffs.min(axis=1)
    score = row_div.groupby(level='item_id').mean()
    score.name = 'diverse_score'
    return score


def accurate_scores(df, selected_l1s):
    pred_df = df[['item_id', 'lang', 'prediction', 'target_score']].copy()
    pred_df = pred_df.drop_duplicates(subset=['item_id', 'lang'])
    pred_df = pred_df[pred_df['lang'].isin(selected_l1s)]
    pred_df = pred_df.dropna(subset=['prediction', 'target_score'])
    if pred_df.empty:
        return pd.Series(dtype=float, name='mean_abs_error')
    pred_df['abs_error'] = (pred_df['prediction'] - pred_df['target_score']).abs()
    item_err = pred_df.groupby('item_id')['abs_error'].mean()
    item_err.name = 'mean_abs_error'
    return item_err


def main(args):
    mode_label = 'detailed' if args.detailed else 'grouped'
    shap_dir = Path(args.shap_dir)
    in_path = shap_dir / f'shap_{mode_label}.csv'
    if not in_path.exists():
        raise FileNotFoundError(f'SHAP file not found: {in_path}')

    if (not args.diverse) and (args.accurate is None):
        raise ValueError('Specify at least one criterion: --diverse and/or --accurate.')

    selected_l1s = parse_accurate_l1s(args.accurate)
    key_col = 'feature' if args.detailed else 'feature_group'
    df = pd.read_csv(in_path)

    required = {'item_id', 'lang', key_col, 'shap_value', 'prediction', 'target_score'}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f'Missing required columns in {in_path}: {", ".join(missing)}'
            )

    base_items = pd.DataFrame({'item_id': sorted(df['item_id'].dropna().unique())})
    out = base_items.merge(item_text_meta(df), on='item_id', how='left')

    norm_cols = []
    if args.diverse:
        div = diverse_scores(df, key_col=key_col).rename('diverse_score')
        out = out.merge(div, on='item_id', how='left')
        out['diverse_norm'] = normalize_01(out['diverse_score'])
        norm_cols.append('diverse_norm')

    if selected_l1s is not None:
        err = accurate_scores(df, selected_l1s=selected_l1s)
        out = out.merge(err, on='item_id', how='left')
        out['accurate_score'] = -out['mean_abs_error']
        out['accurate_norm'] = normalize_01(out['accurate_score'])
        norm_cols.append('accurate_norm')

    if not norm_cols:
        raise ValueError('No active scoring criteria.')

    out['combined_score'] = out[norm_cols].mean(axis=1, skipna=True)
    out = out.sort_values(
        by=['combined_score', 'item_id'],
        ascending=[False, True],
        na_position='last'
        )

    if args.output is None:
        out_path = shap_dir / f'sorted_items_{mode_label}.csv'
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f'Wrote sorted items to {out_path}')


if __name__ == '__main__':
    try:
        main(parse_args())
    except Exception as exc:
        print(f'Error: {exc}')
        raise SystemExit(1)
