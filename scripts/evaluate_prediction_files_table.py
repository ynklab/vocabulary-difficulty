import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
import re

import numpy as np
import pandas as pd


DEFAULT_LANGUAGES = ['cn', 'de', 'es']
LANG2NAME = {
    'cn': 'Chinese',
    'de': 'German',
    'es': 'Spanish',
    }
BASELINES = {
    'closed': {
        'es': {'RMSE': 1.257, 'PCC': 0.765},
        'de': {'RMSE': 1.258, 'PCC': 0.773},
        'cn': {'RMSE': 1.140, 'PCC': 0.753},
        },
    'open': {
        'es': {'RMSE': 1.198, 'PCC': 0.783},
        'de': {'RMSE': 1.166, 'PCC': 0.786},
        'cn': {'RMSE': 1.034, 'PCC': 0.804},
        },
    }


def published_langs(s):
    return s.split(' ')


PUBLISHED2FILTER = {
    'yes': (lambda x: bool(x)),
    'no': (lambda x: not x),
    'all': (lambda x: len(published_langs(x)) == 3),
    'partial': (lambda x: 0 < len(published_langs(x)) < 3),
    'de': (lambda x: 'de' in published_langs(x)),
    'es': (lambda x: 'es' in published_langs(x)),
    'cn': (lambda x: 'cn' in published_langs(x))
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pred-files',
        nargs='+',
        required=True,
        help=(
            'Prediction sources. Each entry becomes one table row. '
            'Supported forms: '
            '(1) a CSV path, or '
            '(1b) logit:<path> to apply logit transform before scoring, or '
            '(2) open:<name> / closed:<name> / close:<name> '
            'for <submission-directory>/<track>/<lang>/<name>.csv files, or '
            '(3) <submission_directory>:open:<name> / '
            '<submission_directory>:close:<name> to override '
            '--submission-directory for one source.'
            ),
        )
    parser.add_argument(
        '--row-labels',
        nargs='+',
        help=(
            'Optional row labels for prediction files. '
            'Use "/" between labels to insert \\midrule between rows in LaTeX. '
            'Defaults to filenames/specs from --pred-files.'
            ),
        )
    parser.add_argument(
        '--first-label', default='System',
        help=(
            'Optional first column label. Default: system'
            ),
        )
    parser.add_argument(
        '--published', choices=PUBLISHED2FILTER,
        nargs='+',
        help='Filter items based on whether published previously.'
        )
    parser.add_argument(
        '--languages',
        nargs='+',
        default=DEFAULT_LANGUAGES,
        help='Languages to evaluate and display. Default: cn es de.',
        )
    parser.add_argument(
        '--labels-root',
        default='data/test',
        help='Root directory containing test labels by language.',
        )
    parser.add_argument(
        '--submission-directory',
        default='submission',
        help='Root directory for open:/closed: submission files.',
        )
    parser.add_argument(
        '--output-csv',
        help='Output CSV path for combined RMSE/PCC table.',
        )
    parser.add_argument(
        '--output-tex',
        help='Output .tex path with RMSE and PCC tabulars.',
        )
    parser.add_argument(
        '--append-tex', '-a', action='store_true',
        help='Append TeX instead of overwriting the file.',
        )
    parser.add_argument(
        '--tabularx', '-x', action='store_true',
        help='Use column-wide tabularx.',
        )
    parser.add_argument(
        '--label-id',
        help='ID for table labels.',
        )
    parser.add_argument(
        '--float-format',
        default='%.3f',
        help='Float format for LaTeX output. Default: %%.3f.',
        )
    parser.add_argument(
        '--baseline', '-b',
        choices=sorted(BASELINES),
        help='Optionally append a Baseline row (closed or open).',
        )
    parser.add_argument(
        '--block-reversed',
        metavar='X',
        help=(
            'Optionally append a final row for '
            'predictions/finetuned_llm/test/block-reversed.csv, '
            'using X as the row label.'
            ),
        )
    return parser.parse_args()


def validate_args(args):
    if len(set(args.languages)) != len(args.languages):
        raise ValueError(
            f'--languages must be unique; got {args.languages}'
            )


def parse_row_labels(row_labels, pred_sources, marker='/'):
    if row_labels is None:
        return [source['label'] for source in pred_sources], set()

    labels = []
    midrule_after_rows = set()
    for token in row_labels:
        if token == marker:
            if not labels:
                raise ValueError(
                    '--row-labels marker "/" cannot appear before first label.'
                    )
            midrule_after_rows.add(len(labels) - 1)
            continue
        labels.append(token)

    if len(labels) != len(pred_sources):
        raise ValueError(
            '--row-labels must contain exactly one non-marker label per '
            f'--pred-files entry ({len(labels)} vs {len(pred_sources)}).'
            )
    return labels, midrule_after_rows


def parse_pred_source(spec):
    if spec.lower().startswith('logit:'):
        nested_spec = spec[len('logit:'):].strip()
        if not nested_spec:
            raise ValueError(
                f'Invalid prediction source (missing path after logit:): {spec}'
                )
        nested_source = parse_pred_source(nested_spec)
        if nested_source['kind'] != 'path':
            raise ValueError(
                f'Invalid prediction source {spec}: logit: is only supported '
                'for file paths, not submission prefixes.'
                )
        nested_source['transform'] = 'logit'
        nested_source['label'] = spec
        return nested_source

    if ':' not in spec:
        path = (
            Path('predictions') / 'finetuned_llm' / 'test' / spec
            if ('/' not in spec)
            else Path(spec)
            )
        return {
            'kind': 'path',
            'path': path,
            'transform': 'none',
            'label': path.name,
            }

    parts = spec.split(':', 2)
    if len(parts) == 3 and parts[1].strip().lower() in {'open', 'closed', 'close'}:
        submission_directory = parts[0].strip()
        prefix = parts[1].strip().lower()
        name = parts[2].strip()
    else:
        prefix, name = spec.split(':', 1)
        prefix = prefix.strip().lower()
        name = name.strip()
        submission_directory = None

    if not name:
        raise ValueError(f'Invalid prediction source (missing name): {spec}')
    if submission_directory == '':
        raise ValueError(
            f'Invalid prediction source (missing submission directory): {spec}'
            )
    if prefix not in {'open', 'closed', 'close'}:
        raise ValueError(
            f'Invalid prediction source prefix in {spec}; '
            'expected open, close, or closed (optionally prefixed by '
            '<submission_directory>:).'
            )
    track = 'closed' if prefix in {'closed', 'close'} else 'open'
    normalized_name = (
        name if name.startswith('predictions_')
        else f'predictions_{name}'
        )
    filename = (
        normalized_name
        if normalized_name.endswith('.csv')
        else f'{normalized_name}.csv'
        )
    return {
        'kind': 'submission',
        'track': track,
        'name': normalized_name,
        'filename': filename,
        'submission_directory': submission_directory,
        'transform': 'none',
        'label': spec,
        }


def apply_prediction_transform(values, transform):
    arr = np.asarray(values, dtype=float)
    if transform == 'none':
        return arr
    if transform == 'logit':
        eps = 1e-6
        clipped = np.clip(arr, eps, 1.0 - eps)
        return np.log(clipped / (1.0 - clipped))
    raise ValueError(f'Unsupported prediction transform: {transform}')


def load_gold_labels(languages, labels_root):
    labels_by_lang = {}
    root = Path(labels_root)
    for lang in languages:
        path = root / lang / f'kvl_shared_task_{lang}_test.csv'
        if not path.exists():
            raise FileNotFoundError(f'Missing gold label file: {path}')
        labels_by_lang[lang] = pd.read_csv(
            path,
            usecols=['item_id', 'GLMM_score']
            ).rename(columns={'GLMM_score': 'gold'})
    return labels_by_lang


def rmse(y_true, y_pred):
    arr_true = np.asarray(y_true, dtype=float)
    arr_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((arr_true - arr_pred) ** 2)))


def pearson_or_nan(y_true, y_pred):
    true_s = pd.Series(y_true, dtype=float)
    pred_s = pd.Series(y_pred, dtype=float)
    if len(true_s) < 2:
        return np.nan
    return float(true_s.corr(pred_s, method='pearson'))


def extract_prediction_column(df, source_label):
    if 'item_id' not in df.columns:
        raise ValueError(f'Missing item_id column in {source_label}')
    if 'prediction' in df.columns:
        return 'prediction'

    candidate_cols = [
        col for col in df.columns
        if (
            (col != 'item_id') and
            pd.api.types.is_numeric_dtype(df[col])
            )
        ]
    if len(candidate_cols) == 1:
        return candidate_cols[0]
    raise ValueError(
        f'Could not uniquely identify prediction column in {source_label}; '
        'expected a "prediction" column or exactly one numeric non-item_id column.'
        )


def evaluate_wide_file(
    pred_path,
    languages,
    labels_by_lang,
    published_ids=None,
    prediction_transform='none'
    ):
    df = pd.read_csv(pred_path)
    if 'item_id' not in df.columns:
        raise ValueError(f'Missing item_id column in {pred_path}')

    row_rmse = {}
    row_pcc = {}
    for lang in languages:
        pred_col = f'{lang}_ftllm_output'
        if pred_col not in df.columns:
            print(
                f'Warning: missing column {pred_col} in {pred_path}; using NaN',
                file=sys.stderr,
                )
            row_rmse[lang] = np.nan
            row_pcc[lang] = np.nan
            continue

        merged = labels_by_lang[lang].merge(
            df[['item_id', pred_col]].rename(columns={pred_col: 'prediction'}),
            on='item_id',
            how='inner',
            )
        valid = merged[['gold', 'prediction']].dropna()
        if valid.empty:
            print(
                f'Warning: no evaluable rows for {lang} in {pred_path}; using NaN',
                file=sys.stderr,
                )
            row_rmse[lang] = np.nan
            row_pcc[lang] = np.nan
            continue

        prediction = apply_prediction_transform(
            valid['prediction'].to_numpy(),
            prediction_transform
            )
        row_rmse[lang] = rmse(
            valid['gold'].to_numpy(),
            prediction
            )
        row_pcc[lang] = pearson_or_nan(
            valid['gold'].to_numpy(),
            prediction
            )
    return row_rmse, row_pcc


def evaluate_submission_track(
    track,
    filename,
    languages,
    labels_by_lang,
    published_ids=None,
    submission_directory='submission'
    ):
    row_rmse = {}
    row_pcc = {}
    submission_root = Path(submission_directory)
    for lang in languages:
        pred_path = submission_root / track / lang / filename
        if not pred_path.exists():
            print(
                f'Warning: missing file {pred_path}; using NaN for {lang}',
                file=sys.stderr,
                )
            row_rmse[lang] = np.nan
            row_pcc[lang] = np.nan
            continue

        df = pd.read_csv(pred_path)
        pred_col = extract_prediction_column(df, str(pred_path))
        merged = labels_by_lang[lang].merge(
            df[['item_id', pred_col]].rename(columns={pred_col: 'prediction'}),
            on='item_id',
            how='inner',
            )
        if published_ids is not None:
            merged = merged[merged['item_id'].isin(published_ids)]
        valid = merged[['gold', 'prediction']].dropna()
        if valid.empty:
            print(
                f'Warning: no evaluable rows for {lang} in {pred_path}; using NaN',
                file=sys.stderr,
                )
            row_rmse[lang] = np.nan
            row_pcc[lang] = np.nan
            continue

        row_rmse[lang] = rmse(
            valid['gold'].to_numpy(),
            valid['prediction'].to_numpy()
            )
        row_pcc[lang] = pearson_or_nan(
            valid['gold'].to_numpy(),
            valid['prediction'].to_numpy()
            )
    return row_rmse, row_pcc


def finalize_means(row_rmse, row_pcc):
    row_rmse['Mean'] = (
        np.nan if np.isnan(np.asarray(list(row_rmse.values()))).all()
        else float(np.nanmean(np.asarray(list(row_rmse.values()), dtype=float)))
        )
    row_pcc['Mean'] = (
        np.nan if np.isnan(np.asarray(list(row_pcc.values()))).all()
        else float(np.nanmean(np.asarray(list(row_pcc.values()), dtype=float)))
        )
    return row_rmse, row_pcc


def insert_midrules_in_tabular(tabular, midrule_after_rows):
    if not midrule_after_rows:
        return tabular

    lines = tabular.splitlines()
    out_lines = []
    in_data_rows = False
    data_row_i = -1
    for line in lines:
        stripped = line.strip()
        out_lines.append(line)
        if stripped == r'\midrule':
            in_data_rows = True
            continue
        if stripped == r'\bottomrule':
            in_data_rows = False
            continue
        if in_data_rows and stripped.endswith(r'\\'):
            data_row_i += 1
            if data_row_i in midrule_after_rows:
                out_lines.append(r'\midrule')
    return '\n'.join(out_lines).removesuffix('\n')


def postprocess_tabular(
    tabular, first_label, metric, label_id, midrule_after_rows, tabularx
    ):
    if not label_id:
        label_id = ''
    if tabularx:
        tabular = re.sub(
            r'\\end{tabular}$',
            r'\\end{tabularx}',
            re.sub(
                r'^\\begin{tabular}{l',
                r'\\begin{tabularx}{\\linewidth}{X',
                tabular
                )
            )
    tabular = re.sub(
        r'^ \& ',
        f'{first_label} & ',
        re.sub(r'^row_label .*\n', '', tabular, flags=re.MULTILINE),
        flags=re.MULTILINE
        )
    tabular = insert_midrules_in_tabular(tabular, midrule_after_rows)
    tabular = tabular.rstrip('\n')
    return (
        '\\begin{table}[t]\n'
        '\\setlength{\\tabcolsep}{3.1pt}\n'
        '\\footnotesize\n'
        '\\centering\n' +
        tabular +
        f'\n\\caption{{{metric} {label_id}}}\n' +
        (f'\\label{{tab:{metric.lower()}-{label_id}}}\n' if label_id else '') +
        '\\end{table}'
        )


def latex_with_bold_best(
    df, float_format, minimize, first_label, metric, label_id, midrule_after_rows,
    tabularx, exclude_bold_rows=None
    ):
    exclude_bold_rows = set(exclude_bold_rows or [])
    df_for_best = df.drop(index=list(exclude_bold_rows), errors='ignore')

    best_by_col = {}
    for col in df.columns:
        valid = df_for_best[col].dropna()
        if valid.empty:
            best_by_col[col] = None
        elif minimize:
            best_by_col[col] = float(valid.min())
        else:
            best_by_col[col] = float(valid.max())

    formatted_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)
    for row_label in df.index:
        for col in df.columns:
            value = df.at[row_label, col]
            if pd.isna(value):
                formatted_df.at[row_label, col] = ''
                continue
            formatted = float_format % float(value)
            best_value = best_by_col[col]
            if (
                (row_label not in exclude_bold_rows) and
                (best_value is not None) and
                np.isclose(float(value), best_value)
                ):
                formatted_df.at[row_label, col] = f'\\textbf{{{formatted}}}'
            else:
                formatted_df.at[row_label, col] = formatted

    column_format = 'l' + ('c' * len(df.columns))
    return postprocess_tabular(
        formatted_df.to_latex(escape=False, column_format=column_format),
        first_label, metric, label_id, midrule_after_rows, tabularx
        )


def main():
    args = parse_args()
    validate_args(args)

    pred_sources = [parse_pred_source(spec) for spec in args.pred_files]
    for source in pred_sources:
        if source['kind'] == 'path' and not source['path'].exists():
            raise FileNotFoundError(
                f'Prediction file not found: {source["path"]}'
                )

    row_labels, midrule_after_rows = parse_row_labels(
        args.row_labels,
        pred_sources
        )

    published_filters = (
        [PUBLISHED2FILTER[p] for p in args.published]
        if args.published is not None else [None for _ in pred_sources]
        )

    if args.published:
        published = pd.read_csv(Path('data/published_test_data.csv'), na_filter=False)

    labels_by_lang = load_gold_labels(args.languages, args.labels_root)

    rmse_rows = []
    pcc_rows = []
    for label, source, published_filter in zip(
        row_labels, pred_sources, published_filters
        ):
        published_ids = published['item_id'].loc[
            published['published_langs'].map(published_filter)
            ] if published_filter else None
        if source['kind'] == 'path':
            row_rmse, row_pcc = evaluate_wide_file(
                source['path'],
                args.languages,
                labels_by_lang,
                published_ids=published_ids,
                prediction_transform=source.get('transform', 'none')
                )
        else:
            row_rmse, row_pcc = evaluate_submission_track(
                source['track'],
                source['filename'],
                args.languages,
                labels_by_lang,
                published_ids=published_ids,
                submission_directory=(
                    source.get('submission_directory') or
                    args.submission_directory
                    )
                )
        row_rmse, row_pcc = finalize_means(row_rmse, row_pcc)
        row_rmse['row_label'] = label
        row_pcc['row_label'] = label
        rmse_rows.append(row_rmse)
        pcc_rows.append(row_pcc)

    if args.baseline:
        baseline_rmse = {
            lang: BASELINES[args.baseline].get(lang, {}).get('RMSE', np.nan)
            for lang in args.languages
            }
        baseline_pcc = {
            lang: BASELINES[args.baseline].get(lang, {}).get('PCC', np.nan)
            for lang in args.languages
            }
        baseline_rmse, baseline_pcc = finalize_means(
            baseline_rmse,
            baseline_pcc
            )
        baseline_label = (
            'Open-Track Baseline'
            if args.baseline == 'open'
            else 'Closed-Track Baseline'
            )
        baseline_rmse['row_label'] = baseline_label
        baseline_pcc['row_label'] = baseline_label
        rmse_rows.append(baseline_rmse)
        pcc_rows.append(baseline_pcc)

    if args.block_reversed:
        block_reversed_path = (
            Path('predictions') /
            'finetuned_llm' /
            'test' /
            'block-reversed.csv'
            )
        if not block_reversed_path.exists():
            raise FileNotFoundError(
                f'Prediction file not found: {block_reversed_path}'
                )
        row_rmse, row_pcc = evaluate_wide_file(
            block_reversed_path,
            args.languages,
            labels_by_lang,
            published_ids=None
            )
        row_rmse, row_pcc = finalize_means(row_rmse, row_pcc)
        row_rmse['row_label'] = args.block_reversed
        row_pcc['row_label'] = args.block_reversed
        rmse_rows.append(row_rmse)
        pcc_rows.append(row_pcc)

    ordered_cols = [*args.languages, 'Mean']
    rmse_df = (
        pd.DataFrame(rmse_rows)
        .set_index('row_label')[ordered_cols]
        .rename(columns=LANG2NAME)
        )
    pcc_df = (
        pd.DataFrame(pcc_rows)
        .set_index('row_label')[ordered_cols]
        .rename(columns=LANG2NAME)
        )
    exclude_bold_rows = [args.block_reversed] if args.block_reversed else []

    combined = pd.concat({'RMSE': rmse_df, 'PCC': pcc_df}, axis=1)
    if args.output_csv is not None:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_csv)
        print(f'Saved CSV: {out_csv}')
    else:
        combined.to_csv(sys.stdout)
        print()

    if args.output_tex is not None:
        out_tex = Path(args.output_tex)
        out_tex.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_tex = None
    with (
        out_tex.open('a' if args.append_tex else 'w') if (out_tex is not None) else
        nullcontext(sys.stdout)
        ) as f:
        f.write(f'% RMSE {args.label_id}\n')
        f.write(
            latex_with_bold_best(
                rmse_df,
                float_format=args.float_format,
                minimize=True,
                first_label=args.first_label,
                metric='RMSE',
                label_id=args.label_id,
                midrule_after_rows=midrule_after_rows,
                tabularx=args.tabularx,
                exclude_bold_rows=exclude_bold_rows
                )
            )
        f.write(f'\n\n% PCC {args.label_id}\n')
        f.write(
            latex_with_bold_best(
                pcc_df,
                float_format=args.float_format,
                minimize=False,
                first_label=args.first_label,
                metric='PCC',
                label_id=args.label_id,
                midrule_after_rows=midrule_after_rows,
                tabularx=args.tabularx,
                exclude_bold_rows=exclude_bold_rows
                )
            )
        f.write('\n\n')
    if out_tex is not None:
        print(f'Saved LaTeX: {out_tex}')


if __name__ == '__main__':
    main()
