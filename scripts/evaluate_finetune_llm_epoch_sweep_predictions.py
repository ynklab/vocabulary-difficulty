import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CAL_TAGS = ['cal', 'no_cal']
CAL_LABELS = {'cal': 'yes', 'no_cal': 'no'}
METRICS = [('RMSE', 'rmse'), ('PCC', 'pearson')]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-name',
        required=True,
        help='Run config name used to resolve sweep prediction file paths.',
        )
    parser.add_argument(
        '--model-name',
        help=(
            'Model name accepted for interface parity with the sweep script. '
            'Used for logging only.'
            ),
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
        help='Languages to score and include before adding mean column.',
        )
    parser.add_argument(
        '--predictions-root',
        default='predictions/finetuned_llm',
        help='Base directory containing per-run sweep prediction CSVs.',
        )
    parser.add_argument(
        '--results-root',
        default='results/finetuned_llm',
        help='Base directory for summary outputs.',
        )
    parser.add_argument(
        '--output-path',
        help=(
            'Optional summary output path. Default: '
            '<results-root>/<safe_config>/<safe_config>--epoch_eval_summary.csv'
            ),
        )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite summary file if it already exists.',
        )
    return parser.parse_args()


def validate_args(args):
    if args.epochs < 0:
        raise ValueError(f'--epochs must be >= 0, got {args.epochs}')
    if not args.languages:
        raise ValueError('--languages must be non-empty')
    if len(set(args.languages)) != len(args.languages):
        raise ValueError(
            f'--languages must be unique; got {args.languages}'
            )


def safe_name(value):
    value = re.sub(r'[^A-Za-z0-9._-]+', '--', value)
    value = re.sub(r'^-+|-+$', '', value)
    return value


def default_output_path(results_root, safe_config):
    return (
        Path(results_root) /
        safe_config /
        f'{safe_config}--epoch_eval_summary.csv'
        )


def case_prediction_path(predictions_root, safe_config, epoch, cal_tag):
    return (
        Path(predictions_root) /
        safe_config /
        f'{safe_config}--epoch_{epoch}--{cal_tag}.csv'
        )


def build_label_frames(languages):
    labels_by_lang = {}
    for lang in languages:
        path = Path('data') / 'test' / lang / f'kvl_shared_task_{lang}_test.csv'
        if not path.exists():
            raise FileNotFoundError(f'Missing gold label file: {path}')
        df = pd.read_csv(path, usecols=['item_id', 'GLMM_score'])
        labels_by_lang[lang] = df.rename(columns={'GLMM_score': 'gold'})
    return labels_by_lang


def rmse(y_true, y_pred):
    arr_true = np.asarray(y_true, dtype=float)
    arr_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((arr_true - arr_pred) ** 2)))


def pearson_or_nan(y_true, y_pred):
    series_true = pd.Series(y_true, dtype=float)
    series_pred = pd.Series(y_pred, dtype=float)
    if len(series_true) < 2:
        return np.nan
    return float(series_true.corr(series_pred, method='pearson'))


def nanmean_or_nan(values):
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return np.nan
    return float(np.nanmean(arr))


def evaluate_case(pred_path, labels_by_lang, languages):
    pred_df = pd.read_csv(pred_path)
    if 'item_id' not in pred_df.columns:
        raise ValueError(f'Missing item_id column in {pred_path}')

    metric_by_lang = {}
    for lang in languages:
        col = f'{lang}_ftllm_output'
        if col not in pred_df.columns:
            metric_by_lang[lang] = {'rmse': np.nan, 'pearson': np.nan}
            continue
        merged = labels_by_lang[lang].merge(
            pred_df[['item_id', col]].rename(columns={col: 'prediction'}),
            on='item_id',
            how='inner',
            )
        if merged.empty:
            metric_by_lang[lang] = {'rmse': np.nan, 'pearson': np.nan}
            continue

        valid = merged[['gold', 'prediction']].dropna()
        if valid.empty:
            metric_by_lang[lang] = {'rmse': np.nan, 'pearson': np.nan}
            continue

        metric_by_lang[lang] = {
            'rmse': rmse(valid['gold'].to_numpy(), valid['prediction'].to_numpy()),
            'pearson': pearson_or_nan(
                valid['gold'].to_numpy(),
                valid['prediction'].to_numpy(),
                ),
            }
    return metric_by_lang


def build_summary_df(args, safe_config, labels_by_lang):
    language_scopes = list(args.languages) + ['mean']
    column_index = pd.MultiIndex.from_product(
        [language_scopes, ['RMSE', 'PCC'], ['yes', 'no']],
        names=['language_scope', 'metric', 'calibration'],
        )
    index = pd.Index(range(0, args.epochs + 1), name='epoch')
    summary_df = pd.DataFrame(index=index, columns=column_index, dtype=float)

    missing_cases = []
    for epoch in range(0, args.epochs + 1):
        for cal_tag in CAL_TAGS:
            path = case_prediction_path(
                args.predictions_root,
                safe_config,
                epoch,
                cal_tag,
                )
            if not path.exists():
                missing_cases.append((epoch, cal_tag, path))
                continue

            metric_by_lang = evaluate_case(path, labels_by_lang, args.languages)
            cal_label = CAL_LABELS[cal_tag]
            for pretty_metric, raw_metric in METRICS:
                mean_values = []
                for lang in args.languages:
                    value = metric_by_lang[lang][raw_metric]
                    summary_df.loc[epoch, (lang, pretty_metric, cal_label)] = value
                    mean_values.append(value)
                summary_df.loc[epoch, ('mean', pretty_metric, cal_label)] = (
                    nanmean_or_nan(mean_values)
                    )

    return summary_df, missing_cases


def main():
    args = parse_args()
    validate_args(args)

    safe_config = safe_name(args.config_name)
    out_path = (
        Path(args.output_path)
        if args.output_path
        else default_output_path(args.results_root, safe_config)
        )
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            'Summary file already exists '
            f'({out_path}). Pass --overwrite to replace it.'
            )

    if args.model_name:
        print(f'Model (for logging): {args.model_name}')
    print(f'Config: {args.config_name} (safe={safe_config})')
    print(f'Epoch range: 0..{args.epochs}')
    print(f'Languages: {" ".join(args.languages)}')

    labels_by_lang = build_label_frames(args.languages)
    summary_df, missing_cases = build_summary_df(args, safe_config, labels_by_lang)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path)
    print(f'Saved summary: {out_path}')

    if missing_cases:
        print(
            '\nWARNING: Missing prediction case file(s); '
            'summary contains NA for those cells.',
            file=sys.stderr,
            )
        for epoch, cal_tag, path in missing_cases:
            print(
                f'  epoch={epoch} cal={cal_tag} path={path}',
                file=sys.stderr,
                )


if __name__ == '__main__':
    main()
