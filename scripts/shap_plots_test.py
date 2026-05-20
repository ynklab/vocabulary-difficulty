import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
from matplotlib.transforms import blended_transform_factory

from kvl import LANG2NAME

REQUIRED_SUMMARY_BASE_COLS = {'lang', 'mean_abs_shap', 'mean_shap', 'n_items'}
BASE_FONT_SIZE = 13
BAR_COLOR = '#ff0051'
NEG_BAR_COLOR = '#3c78d8'
INSIDE_LABEL_THRESHOLD = 0.15
X_AXIS_ANNOTATION_Y = -0.075
ITEM_LABEL_X = -0.01
ITEM_VALUE_X = -0.13


def configure_fonts():
    # Prefer CJK-capable fonts so Chinese L1 words render in titles.
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = [
        'Arial Unicode MS',
        'Apple SD Gothic Neo',
        'Hiragino Sans',
        'Songti SC',
        'DejaVu Sans'
        ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'shap_dir',
        help='Path to SHAP output directory containing SHAP CSVs.'
        )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Use detailed SHAP files.'
        )
    mode.add_argument(
        '--grouped', '-g',
        action='store_true',
        help='Use grouped SHAP files.'
        )
    parser.add_argument(
        '--items', '-i',
        action='store_true',
        help='Generate item-level SHAP plots instead of summaries.'
        )
    parser.add_argument(
        '--show-base',
        action='store_true',
        help='For item plots, show E[f(X)] baseline marker and label.'
        )
    parser.add_argument(
        '--ids',
        nargs='+',
        default=None,
        help='Item IDs to plot (only with --items). Default: all item IDs.'
        )
    parser.add_argument(
        '--format', '-f',
        choices=['pdf', 'png'],
        default='pdf',
        help='Output format for plots. Default: pdf.'
        )
    parser.add_argument(
        '--print-titles',
        action='store_true',
        help='Do not render plot titles; print them to stdout instead.'
        )
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='Number of bars per plot. Default: 20.'
        )
    parser.add_argument(
        '--outdir',
        default=None,
        help='Output directory for plots. Default: <shap_dir>/plots.'
        )
    parser.add_argument(
        '--dpi',
        type=int,
        default=180,
        help='Output figure DPI. Default: 180.'
        )
    parser.add_argument(
        '--x-max', '--xm',
        type=float,
        default=None,
        help='Maximum x-axis magnitude. Default: auto.'
        )
    parser.add_argument(
        '--x-axis-decimals', '--xd',
        type=int,
        default=2,
        help='Number of decimals for x-axis tick labels. Default: 2.'
        )
    parser.add_argument(
        '--label-decimals', '--ld',
        type=int,
        default=2,
        help='Decimals for item feature values in labels. Default: 2.'
        )
    parser.add_argument(
        '--figure-scale',
        type=float,
        default=1.0,
        help='Scale factor for figure width/height. Default: 1.0.'
        )
    parser.add_argument(
        '--left-margin', '-L',
        type=float,
        default=0.30,
        help='Left subplot margin. Default: 0.30.'
        )
    parser.add_argument(
        '--right-margin', '-R',
        type=float,
        default=None,
        help='Right subplot margin. Default: matplotlib default.'
        )
    parser.add_argument(
        '--top-margin', '-T',
        type=float,
        default=None,
        help='Top subplot margin. Default: matplotlib default.'
        )
    parser.add_argument(
        '--bottom-margin', '-bottom', '-B',
        type=float,
        default=None,
        help='Bottom subplot margin. Default: matplotlib default.'
        )
    return parser.parse_args()


def validate_args(args):
    if args.top_k <= 0:
        raise ValueError('--top-k must be > 0.')
    if args.dpi <= 0:
        raise ValueError('--dpi must be > 0.')
    if (args.x_max is not None) and (args.x_max <= 0):
        raise ValueError('--x-max must be > 0.')
    if args.x_axis_decimals < 0:
        raise ValueError('--x-axis-decimals must be >= 0.')
    if args.label_decimals < 0:
        raise ValueError('--label-decimals must be >= 0.')
    if args.figure_scale <= 0:
        raise ValueError('--figure-scale must be > 0.')
    if args.ids and (not args.items):
        raise ValueError('--ids can only be used with --items.')
    for name, value in (
        ('--left-margin', args.left_margin),
        ('--right-margin', args.right_margin),
        ('--top-margin', args.top_margin),
        ('--bottom-margin', args.bottom_margin)
        ):
        if (value is not None) and not (0 <= value < 1):
            raise ValueError(f'{name} must be in [0, 1).')
    if (args.right_margin is not None) and (args.left_margin >= args.right_margin):
        raise ValueError('--left-margin must be smaller than --right-margin.')
    if (
        (args.top_margin is not None) and
        (args.bottom_margin is not None) and
        (args.bottom_margin >= args.top_margin)
        ):
        raise ValueError('--bottom-margin must be smaller than --top-margin.')


def mode_label(args):
    return 'detailed' if args.detailed else 'grouped'


def key_col_for_mode(m_label):
    return 'feature' if m_label == 'detailed' else 'feature_group'


def summary_file_for_mode(m_label):
    return f'shap_{m_label}_summary.csv'


def items_file_for_mode(m_label):
    return f'shap_{m_label}.csv'


def validate_summary_df(df, key_col, summary_path):
    required_cols = REQUIRED_SUMMARY_BASE_COLS | {key_col}
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        missing_txt = ', '.join(missing)
        raise ValueError(
            f'Missing required columns in {summary_path}: {missing_txt}'
            )
    if df.empty:
        raise ValueError(f'No rows found in {summary_path}.')


def validate_items_df(df, key_col, items_path):
    required_cols = {'lang', 'item_id', key_col, 'shap_value', 'base_value'}
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        missing_txt = ', '.join(missing)
        raise ValueError(
            f'Missing required columns in {items_path}: {missing_txt}'
            )
    if df.empty:
        raise ValueError(f'No rows found in {items_path}.')


def compute_summary_x_limits(df, x_max=None):
    min_x = 0.0
    if x_max is not None:
        return min_x, x_max
    max_x = max(float(df['mean_abs_shap'].max()), 0.0)
    span = max_x - min_x
    right_pad = 0.12 * span if span > 0 else 0.1
    return min_x, max_x + right_pad


def compute_item_x_limits(df, x_max=None):
    if x_max is not None:
        return -x_max, x_max
    max_abs = float(df['shap_value'].abs().max())
    if max_abs <= 0:
        return -0.1, 0.1
    pad = 0.12 * max_abs
    lim = max_abs + pad
    return -lim, lim


def apply_common_axes_style(ax, labels, x_axis_decimals, x_label):
    y_pos = list(range(len(labels)))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=BASE_FONT_SIZE)
    ax.tick_params(axis='y', length=0, labelsize=BASE_FONT_SIZE)
    ax.tick_params(axis='x', labelsize=BASE_FONT_SIZE)
    ax.tick_params(axis='x', which='minor', length=3)
    ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{x_axis_decimals}f'))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.invert_yaxis()
    ax.set_xlabel(x_label, fontsize=BASE_FONT_SIZE)
    ax.set_ylabel('')
    ax.grid(axis='y', linestyle=':', linewidth=0.8, color='#d0d0d0')
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def apply_margins(fig, args):
    adjust_kwargs = {'left': args.left_margin}
    if args.right_margin is not None:
        adjust_kwargs['right'] = args.right_margin
    if args.top_margin is not None:
        adjust_kwargs['top'] = args.top_margin
    if args.bottom_margin is not None:
        adjust_kwargs['bottom'] = args.bottom_margin
    fig.subplots_adjust(**adjust_kwargs)


def set_or_print_title(ax, title, args):
    if args.print_titles:
        print(title)
    else:
        ax.set_title(title)


def add_difficulty_axis_hints(ax, x_left, x_right, scale):
    txt_trans = blended_transform_factory(ax.transData, ax.transAxes)
    # Keep these hints aligned with the x-axis label baseline.
    y = X_AXIS_ANNOTATION_Y * scale
    ax.xaxis.set_label_coords(0.5, y)
    ax.text(
        x_left,
        y,
        'Difficult \u2190',
        transform=txt_trans,
        ha='left',
        va='top',
        color='#666666',
        fontsize=BASE_FONT_SIZE,
        clip_on=False
        )
    ax.text(
        x_right,
        y,
        '\u2192 Easy',
        transform=txt_trans,
        ha='right',
        va='top',
        color='#666666',
        fontsize=BASE_FONT_SIZE,
        clip_on=False
        )


def annotate_summary_values(ax, y_pos, bar_vals, x_left, x_right):
    x_offset = 0.01 * (x_right - x_left)
    for y, value in zip(y_pos, bar_vals):
        is_zero = abs(float(value)) < 1e-12
        show_inside = (value > INSIDE_LABEL_THRESHOLD) and (not is_zero)
        x_pos = (value - x_offset) if show_inside else (value + x_offset)
        ax.text(
            x_pos,
            y,
            'N/A' if is_zero else f'{value:.2f}',
            ha=('right' if show_inside else 'left'),
            va='center',
            color=('white' if show_inside else
                   ('#9a9a9a' if is_zero else BAR_COLOR)),
            fontsize=BASE_FONT_SIZE
            )


def annotate_item_values(ax, y_pos, bar_vals, x_left, x_right, x_max=None):
    x_offset = 0.01 * (x_right - x_left)
    for y, value in zip(y_pos, bar_vals):
        is_zero = abs(float(value)) < 1e-12
        is_clipped = (x_max is not None) and (abs(float(value)) > x_max)
        if is_clipped:
            if value > 0:
                x_pos = x_right - x_offset
                ha = 'right'
                label = f'{value:.2f} 》'
            else:
                x_pos = x_left + x_offset
                ha = 'left'
                label = f'《 {value:.2f}'
            color = 'white'
            ax.text(
                x_pos,
                y,
                label,
                ha=ha,
                va='center',
                color=color,
                fontsize=BASE_FONT_SIZE
                )
            continue
        if is_zero:
            x_pos = x_offset
            ha = 'left'
            color = '#9a9a9a'
            label = 'N/A'
        else:
            abs_val = abs(float(value))
            show_inside = abs_val > INSIDE_LABEL_THRESHOLD
            if value > 0:
                x_pos = (value - x_offset) if show_inside else (value + x_offset)
                ha = 'right' if show_inside else 'left'
                outside_color = BAR_COLOR
            else:
                x_pos = (value + x_offset) if show_inside else (value - x_offset)
                ha = 'left' if show_inside else 'right'
                outside_color = NEG_BAR_COLOR
            color = 'white' if show_inside else outside_color
            label = f'{value:.2f}'
        ax.text(
            x_pos,
            y,
            label,
            ha=ha,
            va='center',
            color=color,
            fontsize=BASE_FONT_SIZE
            )


def item_feature_value_col(item_df, key_col):
    candidates = []
    if key_col == 'feature':
        candidates.append('feature_value')
    elif key_col == 'feature_group':
        candidates.extend(('feature_group_value', 'feature_value'))
    if 'feature_value' not in candidates:
        candidates.append('feature_value')
    for col in candidates:
        if col in item_df.columns:
            return col
    return None


def format_item_feature_value(value, decimals):
    if pd.isna(value):
        return 'N/A'
    try:
        return f'{float(value):.{decimals}f}'
    except (TypeError, ValueError):
        return str(value)


def draw_item_y_labels(ax, y_pos, labels, value_labels=None):
    txt_trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.set_yticklabels([])
    for idx, y in enumerate(y_pos):
        if value_labels is not None:
            ax.text(
                ITEM_VALUE_X,
                y,
                f'{value_labels[idx]} =',
                transform=txt_trans,
                ha='right',
                va='center',
                color='#7a7a7a',
                fontsize=BASE_FONT_SIZE,
                clip_on=False
                )
        ax.text(
            ITEM_LABEL_X,
            y,
            labels[idx],
            transform=txt_trans,
            ha='right',
            va='center',
            color='black',
            fontsize=BASE_FONT_SIZE,
            clip_on=False
            )


def figure_size(n_rows, figure_scale, has_title=True):
    extra_h = 1.8 if has_title else 1.3
    fig_h = max(4.0, 0.5 * n_rows + extra_h) * figure_scale
    fig_w = 11 * figure_scale
    return fig_w, fig_h


def plot_language_summary(lang_df, lang, m_label, key_col, out_path, args):
    labels = lang_df[key_col].astype(str).tolist()
    y_pos = list(range(len(lang_df)))
    fig_w, fig_h = figure_size(
        len(lang_df), args.figure_scale, has_title=not args.print_titles
        )
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bar_vals = lang_df['mean_abs_shap'].astype(float)
    ax.barh(y_pos, bar_vals, color=BAR_COLOR, alpha=0.98, height=0.68)

    x_left, x_right = compute_summary_x_limits(lang_df, x_max=args.x_max)
    ax.set_xlim(x_left, x_right)
    ax.axvline(0.0, color='black', linewidth=1.2, alpha=0.95, zorder=4)

    apply_common_axes_style(
        ax=ax,
        labels=labels,
        x_axis_decimals=args.x_axis_decimals,
        x_label='mean(|SHAP value|)'
        )

    lang_name = LANG2NAME.get(lang, lang)
    if m_label == 'grouped':
        title_prefix = 'SHAP Summary by Feature Group'
    else:
        title_prefix = 'SHAP summary'
    set_or_print_title(ax, f'{title_prefix} (L1={lang_name})', args=args)

    annotate_summary_values(ax, y_pos, bar_vals, x_left, x_right)
    apply_margins(fig, args)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)


def normalize_item_ids(series):
    return series.astype(str)


def select_requested_ids(item_df, requested_ids):
    if not requested_ids:
        return item_df
    ids_str = [str(i) for i in requested_ids]
    item_id_str = normalize_item_ids(item_df['item_id'])
    available = set(item_id_str.unique())
    missing = sorted(set(ids_str).difference(available))
    if missing:
        raise ValueError(
            'Requested --ids not found in item SHAP file: ' + ', '.join(missing)
            )
    return item_df[item_id_str.isin(ids_str)].copy()


def plot_item_explanations(item_df, m_label, key_col, items_out_dir, ext, args):
    def first_non_null_or_na(series):
        non_null = series.dropna()
        return str(non_null.iloc[0]) if len(non_null) else 'N/A'

    def first_non_null_fmt(series, decimals=2):
        non_null = series.dropna()
        if len(non_null) == 0:
            return 'N/A'
        try:
            return f'{float(non_null.iloc[0]):.{decimals}f}'
        except (TypeError, ValueError):
            return 'N/A'

    def first_non_null_float(series):
        non_null = series.dropna()
        if len(non_null) == 0:
            return None
        try:
            return float(non_null.iloc[0])
        except (TypeError, ValueError):
            return None

    n_written = 0
    value_col = item_feature_value_col(item_df, key_col=key_col)
    if value_col is None:
        print(
            f'No item feature-value column found for {m_label} mode; '
            'rendering labels without prefixed values.'
            )
    grouped = item_df.groupby(['lang', 'item_id'], sort=True)
    for (lang, item_id), gdf in grouped:
        gdf = gdf.copy()
        gdf['_abs_shap'] = gdf['shap_value'].abs()
        gdf = gdf.sort_values('_abs_shap', ascending=False).head(args.top_k)

        labels = gdf[key_col].astype(str).tolist()
        value_labels = None
        if value_col is not None:
            value_labels = [
                format_item_feature_value(v, decimals=args.label_decimals)
                for v in gdf[value_col]
                ]
        y_pos = list(range(len(gdf)))
        fig_w, fig_h = figure_size(
            len(gdf), args.figure_scale, has_title=not args.print_titles
            )
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        bar_vals = gdf['shap_value'].astype(float)
        colors = [BAR_COLOR if v >= 0 else NEG_BAR_COLOR for v in bar_vals]
        ax.barh(y_pos, bar_vals, color=colors, alpha=0.98, height=0.68)

        x_left, x_right = compute_item_x_limits(gdf, x_max=args.x_max)
        ax.set_xlim(x_left, x_right)
        ax.axvline(0.0, color='black', linewidth=1.2, alpha=0.95, zorder=4)

        apply_common_axes_style(
            ax=ax,
            labels=labels,
            x_axis_decimals=args.x_axis_decimals,
            x_label='SHAP value'
            )
        draw_item_y_labels(
            ax=ax,
            y_pos=y_pos,
            labels=labels,
            value_labels=value_labels
            )
        add_difficulty_axis_hints(
            ax, x_left=x_left, x_right=x_right, scale=1 / args.figure_scale
            )

        lang_name = LANG2NAME.get(lang, lang)
        en_word = first_non_null_or_na(gdf['en_target_word'])
        en_pos = first_non_null_or_na(gdf['en_target_pos'])
        l1_word = first_non_null_or_na(gdf['l1_source_word'])
        pred_value = first_non_null_fmt(gdf['prediction'], decimals=2)
        target_value = first_non_null_fmt(gdf['target_score'], decimals=2)
        set_or_print_title(
            ax,
            (f'L1={lang_name}, En={en_word}, POS={en_pos}, L1={l1_word}, '
             f'Pred={pred_value}, Target={target_value}'),
            args=args
            )

        # Add SHAP baseline marker on the x-axis: E[f(X)] = base_value.
        base_value = first_non_null_float(gdf['base_value'])
        if args.show_base and (base_value is not None):
            txt_trans = blended_transform_factory(ax.transData, ax.transAxes)
            ax.plot(
                base_value,
                0.0,
                marker='o',
                markersize=6,
                markerfacecolor='#666666',
                markeredgecolor='#666666',
                transform=txt_trans,
                clip_on=False,
                alpha=0.95,
                zorder=5
                )
            ax.text(
                base_value,
                0.008,
                f'E[f(X)] = {base_value:.2f}',
                transform=txt_trans,
                ha='center',
                va='bottom',
                color='#666666',
                fontsize=BASE_FONT_SIZE - 2
                )

        annotate_item_values(
            ax,
            y_pos,
            bar_vals,
            x_left,
            x_right,
            x_max=args.x_max
            )
        apply_margins(fig, args)

        out_path = items_out_dir / f'shap_{m_label}_{item_id}_{lang}.{ext}'
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f'Wrote {out_path}')
        n_written += 1

    if n_written == 0:
        raise ValueError('No item plots were generated.')
    print(f'Generated {n_written} item plot(s) in {items_out_dir}.')


def main(args):
    validate_args(args)
    configure_fonts()
    m_label = mode_label(args)
    key_col = key_col_for_mode(m_label)

    shap_dir = Path(args.shap_dir)
    out_dir = Path(args.outdir) if args.outdir else (shap_dir / 'plots')
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.items:
        items_path = shap_dir / items_file_for_mode(m_label)
        if not items_path.exists():
            raise FileNotFoundError(f'Item SHAP file not found: {items_path}')
        item_df = pd.read_csv(items_path)
        validate_items_df(item_df, key_col=key_col, items_path=items_path)
        item_df = select_requested_ids(item_df, requested_ids=args.ids)
        items_out_dir = out_dir / f'items_{m_label}'
        items_out_dir.mkdir(parents=True, exist_ok=True)
        plot_item_explanations(
            item_df=item_df,
            m_label=m_label,
            key_col=key_col,
            items_out_dir=items_out_dir,
            ext=args.format,
            args=args
            )
        return

    summary_path = shap_dir / summary_file_for_mode(m_label)
    if not summary_path.exists():
        raise FileNotFoundError(f'Summary file not found: {summary_path}')

    df = pd.read_csv(summary_path)
    validate_summary_df(df, key_col=key_col, summary_path=summary_path)

    n_written = 0
    for lang in sorted(df['lang'].dropna().unique()):
        lang_df = df[df['lang'] == lang].copy()
        if lang_df.empty:
            continue
        lang_df = lang_df.sort_values('mean_abs_shap', ascending=False).head(args.top_k)
        out_path = out_dir / f'shap_summary_{m_label}_{lang}.{args.format}'
        plot_language_summary(
            lang_df=lang_df,
            lang=lang,
            m_label=m_label,
            key_col=key_col,
            out_path=out_path,
            args=args
            )
        n_written += 1
        print(f'Wrote {out_path}')

    if n_written == 0:
        raise ValueError(
            f'No plots generated from {summary_path}; no language rows found.'
            )
    print(f'Generated {n_written} summary plot(s) in {out_dir}.')


if __name__ == '__main__':
    try:
        main(parse_args())
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(1)
