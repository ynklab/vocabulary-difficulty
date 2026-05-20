import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pred-files',
        nargs='+',
        required=True,
        help='Prediction CSV files to average.',
        )
    parser.add_argument(
        '--output-path',
        required=True,
        help='Output CSV path for mean predictions.',
        )
    return parser.parse_args()


def main():
    args = parse_args()
    pred_paths = [Path(p) for p in args.pred_files]

    dataframes = []
    for path in pred_paths:
        if not path.exists():
            raise FileNotFoundError(f'Prediction file not found: {path}')
        df = pd.read_csv(path)
        if 'item_id' not in df.columns:
            raise ValueError(f'Missing item_id column in {path}')
        dataframes.append(df)

    common_cols = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_cols &= set(df.columns)

    common_cols.discard('item_id')
    pred_cols = []
    for col in sorted(common_cols):
        if pd.api.types.is_numeric_dtype(dataframes[0][col]):
            pred_cols.append(col)

    if not pred_cols:
        raise ValueError('No shared numeric prediction columns found.')

    merged = dataframes[0][['item_id', *pred_cols]].copy()
    merged.rename(
        columns={col: f'{col}__m0' for col in pred_cols},
        inplace=True,
        )

    for i, df in enumerate(dataframes[1:], start=1):
        right = df[['item_id', *pred_cols]].copy()
        right.rename(
            columns={col: f'{col}__m{i}' for col in pred_cols},
            inplace=True,
            )
        merged = merged.merge(right, on='item_id', how='inner')

    out = merged[['item_id']].copy()
    for col in pred_cols:
        value_cols = [f'{col}__m{i}' for i in range(len(dataframes))]
        out[col] = merged[value_cols].mean(axis=1)

    out.sort_values('item_id', inplace=True)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f'Saved mean predictions: {output_path}')


if __name__ == '__main__':
    main()
