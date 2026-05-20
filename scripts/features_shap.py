from pathlib import Path
import sys

import numpy as np
import pandas as pd


def opt_sample_df(df, n_rows, random_state):
    if (n_rows is None) or (n_rows <= 0) or (len(df) <= n_rows):
        return df
    return df.sample(n=n_rows, random_state=random_state)


def _make_shap_explainer(fitted_model, model_name, x_background, shap):
    est = (
        fitted_model.estimator_ if hasattr(fitted_model, 'estimator_')
        else fitted_model
        )
    if model_name == 'lr':
        return shap.LinearExplainer(est, x_background)
    if model_name in {'gbr', 'xgbr'}:
        return shap.TreeExplainer(est)
    return shap.Explainer(est, x_background)


def compute_language_shap(
    lang,
    item_id_col,
    model_name,
    model_cls,
    features,
    target_col,
    orig_train,
    orig_dev,
    seed,
    shap_sample_size,
    shap_background_size,
    bin_predictions,
    clip_predictions,
    make_model_fn,
    fitted_model=None
    ):
    import shap

    if not features:
        print(
            f'Warning: Skipping SHAP for {lang}; no features selected.',
            file=sys.stderr
            )
        return None, None

    if fitted_model is None:
        shap_model = make_model_fn(
            model_cls,
            bin_predictions=bin_predictions,
            clip_predictions=clip_predictions
            )
        shap_model.fit(orig_train[features], orig_train[target_col])
    else:
        shap_model = fitted_model
    x_background = opt_sample_df(
        orig_train[features], shap_background_size, random_state=seed
        )
    x_explain = opt_sample_df(
        orig_dev[features], shap_sample_size, random_state=seed
        )
    explain_idx = x_explain.index
    item_ids = orig_dev.loc[explain_idx, item_id_col].to_numpy()
    en_target_words = orig_dev.loc[explain_idx, 'en_target_word'].to_numpy()
    en_target_pos = orig_dev.loc[explain_idx, 'en_target_pos'].to_numpy()
    l1_word_col = f'{lang}_L1_source_word'
    l1_context_col = f'{lang}_L1_context'
    l1_source_words = (
        orig_dev.loc[explain_idx, l1_word_col].to_numpy()
        if l1_word_col in orig_dev else np.repeat('', len(explain_idx))
        )
    l1_contexts = (
        orig_dev.loc[explain_idx, l1_context_col].to_numpy()
        if l1_context_col in orig_dev else np.repeat('', len(explain_idx))
        )
    if target_col in orig_dev:
        target_scores = orig_dev.loc[explain_idx, target_col].to_numpy()
    else:
        target_scores = np.full(len(explain_idx), np.nan)
    explainer = _make_shap_explainer(shap_model, model_name, x_background, shap)
    shap_values = explainer(x_explain)

    shap_matrix = np.asarray(shap_values.values)
    if shap_matrix.ndim == 1:
        shap_matrix = shap_matrix.reshape(-1, 1)

    base_values = np.asarray(shap_values.base_values)
    if base_values.ndim == 0:
        base_values = np.full(len(x_explain), float(base_values))
    elif base_values.ndim > 1:
        base_values = base_values.reshape(len(x_explain), -1)[:, 0]

    predictions = np.asarray(shap_model.predict(x_explain), dtype=float)
    x_vals = x_explain.to_numpy()
    n_examples, n_features = shap_matrix.shape
    fts = list(x_explain.columns)

    detailed_df = pd.DataFrame({
        'lang': np.repeat([lang], n_examples * n_features),
        item_id_col: np.repeat(item_ids, n_features),
        'en_target_word': np.repeat(en_target_words, n_features),
        'en_target_pos': np.repeat(en_target_pos, n_features),
        'l1_source_word': np.repeat(l1_source_words, n_features),
        'l1_context': np.repeat(l1_contexts, n_features),
        'target_score': np.repeat(target_scores, n_features),
        'feature': np.tile(fts, n_examples),
        'feature_value': x_vals.reshape(-1),
        'shap_value': shap_matrix.reshape(-1),
        'base_value': np.repeat(base_values, n_features),
        'prediction': np.repeat(predictions, n_features)
        })
    summary_df = pd.DataFrame({
        'lang': lang,
        'feature': fts,
        'mean_abs_shap': np.abs(shap_matrix).mean(axis=0),
        'mean_shap': shap_matrix.mean(axis=0),
        'n_items': n_examples
        }).sort_values('mean_abs_shap', ascending=False)
    return detailed_df, summary_df


def write_shap_outputs(
    output_dir,
    detailed_rows,
    summary_rows,
    decimals,
    item_id_col='item_id',
    print_shap_summary='detailed',
    shap_format='full'
    ):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = out_dir / 'shap_detailed.csv'
    detailed_summary_path = out_dir / 'shap_detailed_summary.csv'
    grouped_path = out_dir / 'shap_grouped.csv'
    grouped_summary_path = out_dir / 'shap_grouped_summary.csv'

    if detailed_rows:
        detailed_df = pd.concat(detailed_rows, axis=0, ignore_index=True)
    else:
        detailed_df = pd.DataFrame(columns=[
            'lang', item_id_col, 'en_target_word', 'en_target_pos',
            'l1_source_word', 'l1_context', 'target_score',
            'feature', 'feature_value', 'shap_value', 'base_value', 'prediction'
            ])
        print('Warning: No SHAP detailed rows were produced.', file=sys.stderr)
    if shap_format == 'simple':
        detailed_out = _simple_shap_table(
            detailed_df,
            key_col='feature',
            item_id_col=item_id_col
            )
    else:
        detailed_out = detailed_df
    detailed_out.to_csv(detailed_path, index=False)

    if summary_rows:
        detailed_summary_df = pd.concat(summary_rows, axis=0, ignore_index=True)
    else:
        detailed_summary_df = pd.DataFrame(columns=[
            'lang', 'feature', 'mean_abs_shap', 'mean_shap', 'n_items'
            ])
        print('Warning: No SHAP detailed summary rows were produced.',
              file=sys.stderr)
    detailed_summary_df.to_csv(detailed_summary_path, index=False)

    grouped_detailed_df, grouped_summary_df = build_grouped_shap_outputs(
        detailed_df, item_id_col=item_id_col
        )
    if shap_format == 'simple':
        grouped_out = _simple_shap_table(
            grouped_detailed_df,
            key_col='feature_group',
            item_id_col=item_id_col
            )
    else:
        grouped_out = grouped_detailed_df
    grouped_out.to_csv(grouped_path, index=False)
    grouped_summary_df.to_csv(grouped_summary_path, index=False)
    print(
        f'Wrote SHAP outputs to {out_dir}.',
        file=sys.stderr
        )

    if print_shap_summary == 'detailed':
        print(
            '\nSHAP detailed summary (mean |SHAP| and signed mean SHAP)\n'
            '========================================================'
            )
        print(
            detailed_summary_df.to_string(
                index=False,
                float_format=lambda x: f'{x:.{decimals}f}'
                )
            )
    elif print_shap_summary == 'grouped':
        print(
            '\nGrouped SHAP summary (mean |SHAP| and signed mean SHAP)\n'
            '======================================================='
            )
        print(
            grouped_summary_df.to_string(
                index=False,
                float_format=lambda x: f'{x:.{decimals}f}'
                )
            )
    else:
        assert print_shap_summary == 'no'


def _feature_to_group(feature):
    if feature in {'evp_level', 'cefrj_level', 'gse_level'}:
        return 'CEFR Level'
    if feature.endswith('_similarity'):
        return 'L1 Similarity'
    if feature.startswith('glasgow_'):
        return 'Glasgow Norms'
    if feature.endswith('_log_frequency') or feature.endswith('_log_range'):
        if feature.startswith('lang_8_'):
            return 'Prod. Frequency'
        return 'Rec. Frequency'
    if feature.endswith('_ftllm_output'):
        return 'Finetuned LLM'
    if feature.endswith('_transliteration'):
        return 'Transliteration'
    if feature.endswith('_difficulty'):
        return 'Difficulty'
    if feature.endswith('_trickiness'):
        return 'Trickiness'
    if feature.endswith('_calque'):
#         if feature.startswith('morphtran--'):
#             return 'Morph. Translation'
        return 'L1 Calque'
    if feature.endswith('_lexical_ambiguity'):
        return 'Lexical Ambiguity'
    if feature.endswith('_spelling_diff_with_l1_word'):
        return 'Spelling Difficulty'
    if feature == 'word_length':
        return 'Word Length'
    return feature


def build_grouped_shap_outputs(shap_detailed_df, item_id_col='item_id'):
    grouped = shap_detailed_df.copy()
    grouped['feature_group'] = grouped['feature'].map(_feature_to_group)

    grouped_detailed_df = grouped.groupby(
        [
            'lang', item_id_col, 'en_target_word', 'en_target_pos',
            'l1_source_word', 'l1_context', 'target_score', 'feature_group'
            ],
        as_index=False,
        dropna=False
        ).agg({
            'shap_value': 'sum',
            'base_value': 'first',
            'prediction': 'first'
            })

    grouped_summary_df = grouped_detailed_df.groupby(
        ['lang', 'feature_group'],
        as_index=False
        ).agg(
            mean_abs_shap=('shap_value', lambda x: np.abs(x).mean()),
            mean_shap=('shap_value', 'mean'),
            n_items=(item_id_col, 'nunique')
            ).sort_values(['lang', 'mean_abs_shap'], ascending=[True, False])

    return grouped_detailed_df, grouped_summary_df


def _simple_shap_table(shap_df, key_col, item_id_col='item_id'):
    if shap_df.empty:
        return pd.DataFrame(columns=[
            'lang', item_id_col, 'en_target_word', 'en_target_pos',
            'l1_source_word', 'l1_context', 'target_score',
            'base_value', 'prediction'
            ])
    key_order = list(shap_df[key_col].drop_duplicates())
    wide = shap_df.pivot_table(
        index=[
            'lang', item_id_col, 'en_target_word',
            'en_target_pos', 'l1_source_word', 'l1_context',
            'target_score', 'base_value', 'prediction'
            ],
        columns=key_col,
        values='shap_value',
        aggfunc='sum',
        fill_value=0
        ).reset_index()
    wide = wide.rename_axis(None, axis=1)
    base_cols = [
        'lang', item_id_col, 'en_target_word',
        'en_target_pos', 'l1_source_word', 'l1_context',
        'target_score', 'base_value', 'prediction'
        ]
    key_cols = [k for k in key_order if k in wide.columns]
    return wide[base_cols + key_cols]
