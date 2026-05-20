import argparse
from pathlib import Path

import pandas as pd


def path_with_suffix_inserted(path, suffix):
    path = Path(path)
    return path.with_name(f'{path.stem}{suffix}{path.suffix}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results-path',
        required=True,
        help='Base (unsuffixed) results CSV path to write merged output to.',
        )
    parser.add_argument(
        '--folds',
        type=int,
        nargs='+',
        required=True,
        help='1-based fold ids to merge.',
        )
    parser.add_argument(
        '--stdout-file',
        help='Base (unsuffixed) stdout log path to merge from per-fold logs.',
        )
    return parser.parse_args()


def main():
    args = parse_args()

    if any(fold_i <= 0 for fold_i in args.folds):
        raise ValueError('--folds must be positive (1-based)')
    if len(set(args.folds)) != len(args.folds):
        raise ValueError('--folds must not contain duplicates')

    base_path = Path(args.results_path)
    frames = []
    for fold_i in sorted(args.folds):
        fold_path = path_with_suffix_inserted(base_path, f'_fold{fold_i}')
        if not fold_path.exists():
            raise FileNotFoundError(f'Missing per-fold results file: {fold_path}')
        fold_df = pd.read_csv(fold_path)
        frames.append(fold_df)
        print(f'Read: {fold_path} ({len(fold_df)} row(s))')

    merged = pd.concat(frames, ignore_index=True)
    sort_cols = [col for col in ['fold', 'lang'] if col in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    base_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(base_path, index=False)
    print(f'Wrote merged results: {base_path} ({len(merged)} row(s))')

    if args.stdout_file:
        stdout_base = Path(args.stdout_file)
        stdout_base.parent.mkdir(parents=True, exist_ok=True)
        with stdout_base.open('w', encoding='utf-8') as out_f:
            for fold_i in sorted(args.folds):
                fold_log = path_with_suffix_inserted(stdout_base, f'_fold{fold_i}')
                if not fold_log.exists():
                    raise FileNotFoundError(f'Missing per-fold log file: {fold_log}')
                out_f.write(f'===== fold {fold_i} =====\n')
                out_f.write(fold_log.read_text(encoding='utf-8'))
                out_f.write('\n')
        print(f'Wrote merged stdout log: {stdout_base}')


if __name__ == '__main__':
    main()
