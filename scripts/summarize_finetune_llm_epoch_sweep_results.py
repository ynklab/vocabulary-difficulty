import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CAL_TAGS = ['cal', 'no_cal']
CAL_LABELS = {'cal': 'yes', 'no_cal': 'no'}
METRICS = [('RMSE', 'rmse'), ('PCC', 'pearson')]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run-name',
        required=True,
        help='Safe run name used in result file paths.',
        )
    parser.add_argument(
        '--results-root',
        default='results/finetuned_llm',
        help='Base directory containing per-run result CSVs.',
        )
    parser.add_argument(
        '--epochs',
        type=int,
        default=4,
        help='Maximum epoch to include (rows are 0..epochs).',
        )
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['cn', 'es', 'de'],
        help='Languages to include before adding mean column.',
        )
    parser.add_argument(
        '--output-path',
        help=(
            'Optional output path. Default: '
            '<results-root>/<run-name>/<run-name>--epoch_eval_summary.csv'
            ),
        )
    return parser.parse_args()


def validate_args(args):
    if args.epochs < 1:
        raise ValueError(f'--epochs must be >= 1, got {args.epochs}')
    if not args.languages:
        raise ValueError('--languages must be non-empty')
    if len(set(args.languages)) != len(args.languages):
        raise ValueError(
            f'--languages must be unique; got {args.languages}'
            )


def default_output_path(results_root, run_name):
    return (
        Path(results_root) /
        run_name /
        f'{run_name}--epoch_eval_summary.csv'
        )


def case_result_path(results_root, run_name, epoch, cal_tag):
    return (
        Path(results_root) /
        run_name /
        f'{run_name}--epoch_{epoch}--{cal_tag}.csv'
        )


def discover_required_paths(results_root, run_name, epochs):
    paths = {}
    missing = []
    for epoch in range(0, epochs + 1):
        for cal_tag in CAL_TAGS:
            path = case_result_path(results_root, run_name, epoch, cal_tag)
            paths[(epoch, cal_tag)] = path
            if not path.exists():
                missing.append(path)
    if missing:
        msg = '\n'.join(str(p) for p in missing)
        raise FileNotFoundError(
            'Missing required sweep result file(s):\n'
            f'{msg}'
            )
    return paths


def extract_metric_by_lang(df, languages):
    if 'row_type' not in df.columns or 'lang' not in df.columns:
        return pd.DataFrame(index=languages, columns=['rmse', 'pearson'], dtype=float)

    final_df = df[df['row_type'] == 'final'].copy()
    final_df = final_df[final_df['lang'].isin(languages)].copy()
    if final_df.empty:
        return pd.DataFrame(index=languages, columns=['rmse', 'pearson'], dtype=float)

    cols = ['lang', 'rmse', 'pearson']
    missing_cols = [c for c in cols if c not in final_df.columns]
    if missing_cols:
        keep_cols = ['lang']
        metric_df = final_df[keep_cols].drop_duplicates().set_index('lang')
        for col in ['rmse', 'pearson']:
            metric_df[col] = np.nan
    else:
        metric_df = (
            final_df[cols]
            .drop_duplicates(subset=['lang'], keep='last')
            .set_index('lang')
            )

    metric_df = metric_df.reindex(languages)
    return metric_df[['rmse', 'pearson']].astype(float)


def nanmean_or_nan(values):
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return np.nan
    return float(np.nanmean(arr))


def build_summary_df(paths, epochs, languages):
    language_scopes = list(languages) + ['mean']
    column_index = pd.MultiIndex.from_product(
        [language_scopes, ['RMSE', 'PCC'], ['yes', 'no']],
        names=['language_scope', 'metric', 'calibration'],
        )
    index = pd.Index(range(0, epochs + 1), name='epoch')
    summary_df = pd.DataFrame(index=index, columns=column_index, dtype=float)

    for epoch in range(0, epochs + 1):
        for cal_tag in CAL_TAGS:
            cal_label = CAL_LABELS[cal_tag]
            case_df = pd.read_csv(paths[(epoch, cal_tag)])
            by_lang = extract_metric_by_lang(case_df, languages)

            for pretty_metric, raw_metric in METRICS:
                for lang in languages:
                    summary_df.loc[epoch, (lang, pretty_metric, cal_label)] = (
                        by_lang.loc[lang, raw_metric]
                        if lang in by_lang.index
                        else np.nan
                        )

                mean_value = nanmean_or_nan(
                    by_lang[raw_metric].to_numpy(dtype=float)
                    )
                summary_df.loc[epoch, ('mean', pretty_metric, cal_label)] = (
                    mean_value
                    )

    return summary_df


def main():
    args = parse_args()
    validate_args(args)

    paths = discover_required_paths(
        results_root=args.results_root,
        run_name=args.run_name,
        epochs=args.epochs,
        )
    out_path = (
        Path(args.output_path)
        if args.output_path
        else default_output_path(args.results_root, args.run_name)
        )

    summary_df = build_summary_df(
        paths=paths,
        epochs=args.epochs,
        languages=args.languages,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path)
    print(f'Saved summary: {out_path}')


if __name__ == '__main__':
    main()
