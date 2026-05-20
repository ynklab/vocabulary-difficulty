import argparse
import json
import pandas as pd
import numpy as np
import csv
import sys
from itertools import starmap
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor  # RandomForestRegressor,
from xgboost import XGBRegressor
# from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR  # LinearSVR
from scipy.stats import pearsonr
from pathlib import Path
import matplotlib.pyplot as plt
# from wordfreq import word_frequency, zipf_frequency
from bnc import BNC
from kvl import (
    read_data_cv, read_subset, SUBSETS, L1_CODES, LANG2NAME, ID_COL, spaced_clue
    )
from util import (
    int_or_none, str_float_pair, OptionalLogit,
    download_if_necessary, similarity_to_l1, SIMILARITY_TRANSFORMS,
    METRICS, METRIC_IS_SCORE, metric2scorer,
    cli_command, final_predict_command,
    ClippedRegressor, BinnedRegressor, PassthroughRegressor
    )
from zipfile import ZipFile
from joblib import dump, load

from prompt_util import (
    clean_prompting_output, prompting_logprobs2prob, prompting_logprobs2num,
    prompts_and_models, split_rate_compare_soft_probs, concat_feature_columns,
    prompting_logprobs2nums_split
    )
from features_shap import (
    compute_language_shap,
    write_shap_outputs
    )
from frequencies import COCA_COL2TOTAL, get_frequencies_and_missing

LANG_8_LANGUAGES = ['any', 'cn', 'es']

SDIFF_LANGUAGES = ['cn', 'es', 'de']

CEFR2NUM = {
    'A1': -1,
    'A2': -2,
    'B1': -3,
    'B2': -4,
    'C1': -5,
    'C2': -6
    }
CEFR_C2_NUM = -6

EVP_POS2KVL_POS = {
    'modal verb': 'verb',
    'ordinal number': 'number',
    'auxiliary verb': 'verb',
    'adv': 'adverb',
    'NUMBER': 'number',
    'adverb; preposition': 'adverb'
    }
CEFRJ_POS2KVL_POS = {
    'modal auxiliary': 'verb'   # We do not map 'be-verb' etc. (verb forms, not lemmas)
    }

FINETUNED_CONFIG_SHORT_NAMES = {
    'ministral-3-8b':
        'feb24calibrated-allinone--mistralai--Ministral-3-8B-Base-2512',
    # Best Ministral:
    'ministral-3-14b':
        'mar20calibrated-allinone--mistralai--Ministral-3-14B-Base-2512-lr1x-ep4',
    'glm-4-32b-per-language':
        'feb24calibrated-zai-org--GLM-4-32B-Base-0414',
    'glm-4-32b-lr1p5':  # 3rd epoch would actually be better than 4th
        'mar20calibrated-GLM-4-32B-Base-allinone-lr1p5x--zai-org--GLM-4-32B-Base-0414',
    # Best GLM:
    'glm-4-32b':
        'mar20calibrated-GLM-4-32B-Base-allinone-lr1x--zai-org--GLM-4-32B-Base-0414',
    # Qwen 2.5 outpefromed 3, but the current 'whole' data are actually a mixture
    # of 2.5 and 3:
    'qwen2.5-32b':
        'mar20calibrated-Qwen2.5-32B-allinone-lr1p5x--Qwen--Qwen2.5-32B',
    'mmbert-base-closed':
        'mar26-mmbert-ep16-cnesde--jhu-clsp--mmBERT-base'
    }


def read_cefr_levels(path, cefrj=False, ignore_pos=False):
    df = pd.read_csv(path)
    if cefrj:
        # Some headwords contain variants, e.g. "afterward/afterwards", split:
        df['word'] = df['headword'].str.split('/')
        df = df.explode('word', ignore_index=True)
        df['word'] = df['word'].str.strip()  # just in case
        df.rename(columns={'CEFR': 'level'}, inplace=True)
    df['word'] = df['word'].str.lower()
    df['level'] = df['level'].map(CEFR2NUM.get)
    use_pos = ('pos' in df.columns) and not ignore_pos
    if use_pos:
        pos2kvl_pos = CEFRJ_POS2KVL_POS if cefrj else EVP_POS2KVL_POS
        df['pos'] = df['pos'].map(lambda pos: pos2kvl_pos.get(pos, pos))
    # Aggregate duplicates by taking minimum level:
    df = df.groupby(['word', 'pos'] if use_pos else ['word']).min()
    return df


def get_cefrj_levels(ignore_pos=False):
    path = 'data/downloads/cefrj-vocabulary-profile-1.5.csv'
    download_if_necessary(
        url=('https://github.com/openlanguageprofiles/olp-en-cefrj/raw/'
             'c5c6a64303a9fc2d3da22a06dd9827e471dc244c/'
             'cefrj-vocabulary-profile-1.5.csv'),
        path=path
        )
    return read_cefr_levels(path, cefrj=True, ignore_pos=ignore_pos)


def get_vxgl():
    path = 'data/downloads/VXGL.v1.4.csv'
    download_if_necessary(
        url=('https://github.com/maafiah/VXGL/raw/'
             'e2c4af8762c98efe10a5bbbc089e0bbe22444ccd/VXGL.v1.4.csv'),
        path=path
        )
    df = pd.read_csv(path, comment='#', header=None, names=['word', 'level'])
    df['word'] = df['word'].str.lower()
    df = df.groupby(['word']).min()
    return df


MAX_VXGL = 16   # Grade 16 is maximum


def make_model(model_cls, bin_predictions=None, clip_predictions=False):
    m = model_cls()
    if bin_predictions:
        m = BinnedRegressor(m, bin_predictions)
    elif clip_predictions:
        m = ClippedRegressor(m)
    return m


def fitted_lr_coefs(m, fts):
    est = m
    if hasattr(est, 'estimator_'):
        est = est.estimator_
    if not hasattr(est, 'coef_'):
        return None
    coefs = np.asarray(est.coef_).reshape(-1)
    if len(coefs) != len(fts):
        raise ValueError(
            f'Coef count ({len(coefs)}) does not match '
            f'#features ({len(fts)}).'
            )
    coefs_series = pd.Series(coefs, index=fts, dtype=float)
    if hasattr(est, 'intercept_'):
        intercept = np.asarray(est.intercept_).reshape(-1)
        coefs_series = pd.concat((
            pd.Series({'intercept': float(intercept[0])}),
            coefs_series
            ))
    return coefs_series


def fit_coefs_table(
    lang,
    target_col,
    fts,
    data,
    cv,
    orig_train,
    model_cls,
    use_cv=False,
    predict_binned_target=None,
    predict_feature_mean=False,
    bin_predictions=None,
    clip_predictions=False
    ):
    if predict_binned_target:
        return None
    if predict_feature_mean:
        return None
    if not fts:
        return None
    coefs_by_col = {}
    if use_cv:
        for split_no, split in enumerate(cv, start=1):
            m = make_model(
                model_cls,
                bin_predictions=bin_predictions,
                clip_predictions=clip_predictions
                )
            m.fit(
                data.iloc[split.train][fts],
                data.iloc[split.train][target_col]
                )
            c = fitted_lr_coefs(m, fts)
            if c is None:
                return None
            coefs_by_col[(f'split_{split_no}', lang)] = c
    else:
        m = make_model(
            model_cls,
            bin_predictions=bin_predictions,
            clip_predictions=clip_predictions
            )
        m.fit(orig_train[fts], orig_train[target_col])
        c = fitted_lr_coefs(m, fts)
        if c is None:
            return None
        coefs_by_col[lang] = c
    return pd.DataFrame(coefs_by_col)


def print_coefs_table(
    coef_tables,
    coefs_supported,
    use_cv=False,
    cv_suffix='',
    dev_suffix='',
    decimals=3
    ):
    if not coefs_supported:
        print(
            'Coef output is only supported for models exposing '
            'fitted `coef_` (e.g. LR) and non-binned targets.'
            )
        return
    if not coef_tables:
        return
    coef_df = pd.concat(coef_tables, axis=1)
    if use_cv:
        coef_df.columns = pd.MultiIndex.from_tuples(
            coef_df.columns, names=['split', 'language']
            )
        coef_df = coef_df.sort_index(axis=1, level=[0, 1])
        header = f'Learned coefs{cv_suffix} (features model)'
    else:
        header = f'Learned coefs{dev_suffix} (features model)'
    underline = '=' * len(header)
    print(f'\n{header}\n{underline}')
    print(coef_df.to_string(float_format=lambda x: f'{x:.{decimals}f}'))


KNOWN_MODELS = {'gpt-5.2', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'deepseek-v3.2'}


def temperatures2map(temp_args: list[tuple[str, float]]) -> dict[str, float]:
    '''
    If all=... is present, it will become the fallback value, else None is used.
    '''
    all_t = None
    model2t = {}
    for m, t in temp_args:
        if t <= 0:
            raise ValueError(f'Temperature {m}={t} <= 0')
        if m.lower() == 'all':
            if all_t is not None:
                raise ValueError('--temperatures all=... specified multiple times')
            all_t = t
        else:
            if m not in KNOWN_MODELS:
                raise ValueError(f'Unknown model {m} in --temperatures')
            if m in model2t:
                raise ValueError(f'--temperatures {m}=... specified multiple times')
            model2t[m] = t
    return {m: model2t.get(m, all_t) for m in KNOWN_MODELS}


class SimpleStats:
    def __init__(self, x):
        self.sd = x.std()
        self.mean = x.mean()
        self.deltas = np.abs(x - self.mean)
        self.within_1sd = (self.deltas < self.sd).mean()
        self.max_sds = self.deltas.max() / self.sd
        self.min = x.min()
        self.max = x.max()

    def __str__(self):
        return (
            f'μ = {self.mean:.2f}, σ = {self.sd:.2f}, '
            f'{self.within_1sd:.0%} data within ±σ, '
            f'all within ±{self.max_sds:.2f}σ, '
            f'⬇ {self.min:.2f} ⬆ {self.max:.2f}'
            )


def main(args: argparse.Namespace):
    np.random.seed(args.seed)
    ridge_alpha = None
    ridgecv_alphas = None
    if args.model == 'ridge':
        if args.alpha is None:
            ridge_alpha = 1.0
        elif len(args.alpha) != 1:
            raise ValueError('--model ridge expects exactly one --alpha value.')
        else:
            ridge_alpha = args.alpha[0]
    elif args.model == 'ridgecv':
        if not args.cv:
            raise ValueError('--model ridgecv can only be used with --cv.')
        ridgecv_alphas = tuple(args.alpha) if (args.alpha is not None) else (
            0.1, 1.0, 10.0
            )
    if args.final_train_folds is not None and args.final_train is None:
        raise ValueError('--final-train-folds requires --final-train.')
    no_default_features = args.no_default_features
    no_prompting = args.no_prompting or no_default_features
    no_evp = args.no_evp or no_default_features
    no_frequency = args.no_frequency or no_default_features
    no_similarity = args.no_similarity or no_default_features
    no_length = args.no_length or no_default_features
    trickiness = None if (no_prompting or args.trickiness == 'no') else args.trickiness
    calque              = not (no_prompting or args.no_calque)
    lexical_ambiguity   = not (no_prompting or args.no_lexical_ambiguity)
    spelling_diff_without_l1_word = args.spelling_diff_without_l1_word
    spelling_diff_with_l1_word = args.spelling_diff_with_l1_word
    rate_compare_feature_form = args.spelling_diff_feature_form
    prompting_preds     = []
    opt_logit           = OptionalLogit(args.logit)
    temperatures        = temperatures2map(args.temperatures)

    if args.verbose:
        def log(s):
            print(s, file=sys.stderr)
    else:
        def log(_):
            pass

    def models2temps(models):
        return [temperatures[m] for m in models]

    def split_rate_compare_outputs(outputs):
        split_vals = outputs.fillna('').astype(str).str.split(',', expand=True)
        split_vals = split_vals.reindex(columns=range(3), fill_value='')
        split_vals = split_vals.apply(
            lambda col: pd.to_numeric(col.astype(str).str.strip(), errors='coerce')
            ).fillna(3.0)
        return {'cn': split_vals[0], 'es': split_vals[1], 'de': split_vals[2]}

    if (trickiness or args.output_tricky):
        prompting_preds.extend(
            prompts_and_models(args.trickiness_prompts, args.trickiness_models)
            )
    if args.trickiness_temperatures is None:
        args.trickiness_temperatures = models2temps(args.trickiness_models)
    elif len(args.trickiness_temperatures) != len(args.trickiness_models):
        raise Exception(
            'Number of --trickiness-temperatures does not match number of '
            '--trickiness-models.'
            )
    if calque:
        prompting_preds.extend(
            prompts_and_models(args.calque_prompts, args.calque_models)
            # [f'calque--{m}' for m in args.calque_models]
            )
    if lexical_ambiguity:
        prompting_preds.extend(
            [f'lexical_ambiguity--{m}' for m in args.lexical_ambiguity_models]
            )
    if args.transliteration:
        prompting_preds.extend(
            [f'transliteration--{m}' for m in args.transliteration_models]
            )
    if args.difficulty:
        prompting_preds.extend(
            prompts_and_models(args.difficulty_prompts, args.difficulty_models)
            )
    if args.difficulty_temperatures is None:
        args.difficulty_temperatures = models2temps(args.difficulty_models)
    elif len(args.difficulty_temperatures) != len(args.difficulty_models):
        raise Exception(
            'Number of --difficulty-temperatures does not match number of '
            '--difficulty-models.'
            )

    if spelling_diff_without_l1_word:
        prompting_preds.extend(
            [f'spelling_diff_without_l1_word--{m}'
             for m in args.spelling_diff_without_l1_word_models]
            )
    if spelling_diff_with_l1_word:
        prompting_preds.extend(
            [f'spelling_diff_with_l1_word--{m}'
             for m in args.spelling_diff_with_l1_word_models]
            )

    prompting_preds = prompting_preds if prompting_preds else None

    if args.output_tricky:
        train, dev = [
            read_subset(
                s,
                baseline_preds='both',
                prompting_preds=(
                    prompts_and_models(args.trickiness_prompts, args.trickiness_models)
                    if (trickiness or args.output_tricky) else None
                    ),
                provider=args.provider
                )
            for s in SUBSETS
            ]
        assert len(train) > len(dev)
        for subset, data in zip(SUBSETS, (train, dev)):
            for lang in args.l1s:
                for p in args.trickiness_prompts:
                    for m, t in zip(args.trickiness_models,
                                    args.trickiness_temperatures):
                        pm = f'{p}--{m}'
                        if f'{pm}_{lang}_prompting_output' not in data:
                            continue
                        outputs = clean_prompting_output(
                            data[f'{pm}_{lang}_prompting_output'], prompt=p
                            )
                        tricky = outputs != data['en_target_word']
                        tricky_p = data[f'{pm}_{lang}_trickiness'] = (
                            1 - prompting_logprobs2prob(
                                data[f'{pm}_{lang}_prompting_logprobs'],
                                data['en_target_word'], prompt=p, temperature=t
                                )
                            )
                        tricky_data = data.loc[tricky, [
                            'item_id', 'en_target_word', f'{lang}_L1_source_word',
                            f'{lang}_L1_context'
                            ]].copy()
                        tricky_data[f'{pm}_prompting_output'] = outputs[tricky]
                        tricky_data[f'{pm}_prompting_trickiness'] = tricky_p[tricky]
                        tricky_data[f'{pm}_same_length'] = (
                            tricky_data[f'{pm}_prompting_output'].str.len() ==
                            tricky_data['en_target_word'].str.len()
                            )
                        tricky_path = Path('results/tricky') / subset / f'{lang}.csv'
                        tricky_path.parent.mkdir(parents=True, exist_ok=True)
                        tricky_data.to_csv(tricky_path, index=False)
        return

    finetuned_llm_preds = args.finetuned_configs if args.finetuned else None

    if args.no_save and (args.final_predict is None):
        raise ValueError('--no-save can only be used with --final-predict.')

    if (
        (args.final_train is not None) or
        (args.final_predict is not None) or
        (args.final_eval is not None)
        ):
        if (args.train is not None) or (args.eval is not None):
            raise ValueError('Cannot specify --train/--eval with '
                             '--final-(train|predict|eval).')
        if args.final_train is not None:
            args.train = 'full'
            args.eval = 'full'
        else:
            assert (args.final_predict is not None) or (args.final_eval is not None)
            if (
                (args.final_predict is not None) and
                (args.track is None) and
                (not args.no_save)
                ):
                raise ValueError(
                    '--final-predict requires either --track or --no-save.'
                    )
            args.train = 'full'  # Not used actually
            args.eval = 'test'

    log('Reading data...')
    need_test_labels = (
        (args.final_eval is not None) or
        (
            (args.final_predict is not None) and
            (args.output_shap is not None)
            )
        )
    data_cv = read_data_cv(
        args.splits if args.cv else None,
        baseline_preds='both' if (
            (args.final_predict is None) and (args.final_eval is None)
            ) else None,
        prompting_preds=prompting_preds,
        finetuned_llm_preds=finetuned_llm_preds,
        strict_missing=(
            (args.final_train is not None) or (args.final_predict is not None) or
            (args.final_eval is not None)   # TODO more lenient if using predictions?
            ),
        train=args.train,
        eval=args.eval,
        finetuned_llm_short_names=FINETUNED_CONFIG_SHORT_NAMES,
        provider=args.provider,
        # For --final-predict with --output-shap, also load test labels so
        # SHAP outputs include target values.
        test_labels=need_test_labels
        )
    data = data_cv.data
    cv = data_cv.cv
    if args.cv_mode == 'first':
        cv = cv[:1]
    elif args.cv_mode == 'remaining':
        cv = cv[1:]
    else:
        assert args.cv_mode == 'whole'

    if not cv:
        raise ValueError(
            f'No splits selected for cv-mode={args.cv_mode}. '
            f'Need at least 2 folds for remaining mode.'
            )

    final_train_idx = None
    selected_final_train_folds = None
    if args.final_train_folds is not None:
        with open(args.splits, encoding='utf-8') as f:
            split_ids = json.load(f)
        n_splits = len(split_ids)
        selected_final_train_folds = sorted(set(args.final_train_folds))
        invalid_folds = [
            fold for fold in selected_final_train_folds
            if (fold < 1) or (fold > n_splits)
            ]
        if invalid_folds:
            invalid_text = ', '.join(str(f) for f in invalid_folds)
            raise ValueError(
                f'--final-train-folds contains invalid fold(s): {invalid_text}. '
                f'Expected 1..{n_splits}.'
                )

        selected_dev_ids = set()
        for fold_no in selected_final_train_folds:
            __, dev_ids = split_ids[fold_no - 1]
            selected_dev_ids.update(dev_ids)

        id2idx = {
            item_id: idx for idx, item_id in enumerate(data[ID_COL].tolist())
            }
        unknown_ids = sorted(set(selected_dev_ids).difference(id2idx))
        if unknown_ids:
            raise ValueError(
                '--final-train-folds selected IDs not present in loaded data. '
                f'Example missing IDs: {unknown_ids[:5]}.'
                )
        final_train_idx = sorted(id2idx[item_id] for item_id in selected_dev_ids)
        if not final_train_idx:
            raise ValueError(
                '--final-train-folds selected no training rows after union.'
                )
    if args.cv:
        cv_suffix = ' (CV)'
        dev_suffix = ' (train/dev)'
    else:
        cv_suffix = ''
        dev_suffix = ''

    metric = METRICS[args.metric]

    # BNC
    log('Reading BNC...')
    bnc = args.bnc
    if 'no' in bnc:
        assert len(bnc) == 1, bnc
        bnc = []
    if 'spoken' in bnc:
        bncs = BNC(spoken=True, upos=True)
    if 'written' in bnc:
        bncw = BNC(written=True, upos=True)
    if 'total' in bnc:
        bnct = BNC(spoken=True, written=True, upos=True)

    # TUBELEX
    log('Reading TUBELEX...')
    tubelex = args.tubelex
    if 'no' in tubelex:
        assert len(tubelex) == 1, tubelex
        tubelex = []
    tubelex_freq_missing = {
        freq_or_range: get_frequencies_and_missing(
            tubelex='en',
            log_smooth=True,
            column='channels' if (freq_or_range == 'range') else 'count'
            )
        for freq_or_range in tubelex
        }

    # SubIMDB
    log('Reading SubIMDB...')
    subimdb = args.subimdb
    if 'no' in subimdb:
        assert len(subimdb) == 1, subimdb
        subimdb = []
    subimdb_freq_missing = {
        freq_or_range: get_frequencies_and_missing(
            # See procedure here: https://github.com/naist-nlp/tubelex
            path='data/subimdb.tsv',
            log_smooth=True,
            column='videos' if (freq_or_range == 'range') else 'count'
            )
        for freq_or_range in subimdb
        }

    # OpenSubtitles
    log('Reading OpenSubtitles...')
    if args.opensubtitles:
        opensub_log_freq, opensub_missing_log_freq = get_frequencies_and_missing(
            opensubtitles='en',
            log_smooth=True
            )

    # COCA
    if args.coca:
        coca_log_freq, coca_missing_log_freq = get_frequencies_and_missing(
            coca=True, column=args.coca,
            log_smooth=True
            )

    # Lang-8
    log('Reading Lang-8...')
    lang_8_l1s = args.lang_8_l1s
    if 'no' in lang_8_l1s:
        assert len(lang_8_l1s) == 1, lang_8_l1s
        lang_8_l1s = []
    lang_8_lang2freq_missing = {
        lang: get_frequencies_and_missing(
            # We cannot distribute the lang-8 frequency lists:(
            path=f'data/lang-8-en-{lang}.tsv', log_smooth=True
            )
        for lang in lang_8_l1s
        }

    # CEFR
    log('Reading CEFR data...')
    evp = read_cefr_levels('data/evp_pos_combined_levels.csv',
                           ignore_pos=args.no_cefr_pos)
    if args.gse:
        gse = read_cefr_levels('data/gse_levels.csv')
    if args.cefrj:
        cefrj = get_cefrj_levels(ignore_pos=args.no_cefr_pos)
    if args.vxgl:
        vxgl = get_vxgl()

    lang2cognates = {}
    if args.cognet:
        path = 'data/downloads/CogNet-v2.0.zip'
        download_if_necessary(
            url=('https://github.com/kbatsuren/CogNet/raw/'
                 '819c3aeb373ec66621a7ea6b74ff082a91d8b03a/CogNet-v2.0.zip'),
            path=path
            )
        with ZipFile(path) as zf:
            with zf.open('CogNet-v2.0.tsv') as f:
                cog_df = pd.read_table(f, quoting=csv.QUOTE_NONE, na_filter=False)
        # English is always 'lang 1' in the DB:
        cog_dfe = cog_df[cog_df['lang 1'] == 'eng']
        # Lowercase everything:
        cog_dfe['word 1'] = cog_dfe['word 1'].str.lower()
        cog_dfe['word 2'] = cog_dfe['word 2'].str.lower()
        # CogNet's language codes, include Mandarin (cmn) for "Chinese":
        langs2cognet_langs = {
            'cn': {'zho', 'cmn'},
            'es': {'spa'},
            'de': {'deu'}
            }
        # Optional condition:
        condition = not args.cognate_must_differ or (
            cog_dfe['word 1'] != cog_dfe['word 2']
            )
        lang2cognates = {
            lang: set(cog_dfe.loc[
                cog_dfe['lang 2'].isin(cognet_langs) & condition,
                'word 1'
                ]) for lang, cognet_langs in langs2cognet_langs.items()
            }

    # Glasgow norms

    glasgow_norms = None
    glasgow_norms_train_means = None
    if args.glasgow_norms:
        log('Reading Glasgow norms...')
        glasgow_norms = pd.read_csv('data/en-glasgow.csv', header=[0, 1])
        # Drop the second level, we just need to axcess the Words and norm means:
        glasgow_norms.columns = glasgow_norms.columns.droplevel(1)
        # Keep only means of the norms we add as features:
        glasgow_norms = glasgow_norms.drop(
            [c for c in glasgow_norms.columns if (
                c != 'Words' and c.lower() not in args.glasgow_norms
                )],
            axis=1
            ).rename(
                columns=lambda c: 'word' if c == 'Words' else f'glasgow_{c.lower()}'
                )
        # Lowercase:
        glasgow_norms['word'] = glasgow_norms['word'].str.lower()
        # There are no N/A values.
        # Words contains multi-sense items such as:
        # 'shell', 'shell (military)', 'shell (sea)'
        # in which case we only keep the unspecified ones ('shell'):
        glasgow_norms = glasgow_norms[
            ~glasgow_norms['word'].str.contains('(', regex=False)
            ].set_index('word')

    log('Merging data as features...')
    for freq_or_range, (t_values, t_missing) in tubelex_freq_missing.items():
        data[f'tubelex_log_{freq_or_range}'] = t_values.reindex(
            data['en_target_word'], fill_value=t_missing
            ).reset_index(drop=True)
    for freq_or_range, (s_values, s_missing) in subimdb_freq_missing.items():
        data[f'subimdb_log_{freq_or_range}'] = (
            s_values    # contains duplicate rows, TODO: why? ==> we groupby/sum
            ).groupby(level=0).sum().reindex(
                data['en_target_word'], fill_value=s_missing
                ).reset_index(drop=True)
    if args.opensubtitles:
        data['opensub_log_frequency'] = opensub_log_freq.reindex(
            data['en_target_word'],
            fill_value=opensub_missing_log_freq
            ).reset_index(drop=True)
    if args.coca:
        data['coca_log_frequency'] = coca_log_freq.reindex(
            data['en_target_word'],
            fill_value=coca_missing_log_freq
            ).reset_index(drop=True)
    for l8lang, (l8freq, l8missing) in lang_8_lang2freq_missing.items():
        data[f'lang_8_{l8lang}_log_frequency'] = l8freq.reindex(
            data['en_target_word'], fill_value=l8missing
            ).reset_index(drop=True)
    for lang, cognates in lang2cognates.items():
        data[f'{lang}_has_cognate'] = data['en_target_word'].isin(cognates)
    if 'spoken' in bnc:
        data['bnc_spoken_log_frequency'] = bncs.df.reindex(
            data[['en_target_word', 'en_target_pos']],
            fill_value=bncs.smooth_zero
            ).reset_index(drop=True)
    if 'written' in bnc:
        data['bnc_written_log_frequency'] = bncw.df.reindex(
            data[['en_target_word', 'en_target_pos']],
            fill_value=bncw.smooth_zero
            ).reset_index(drop=True)
    if 'total' in bnc:
        data['bnc_total_log_frequency'] = bnct.df.reindex(
            data[['en_target_word', 'en_target_pos']],
            fill_value=bnct.smooth_zero
            ).reset_index(drop=True)
    data['evp_level'] = evp['level'].reindex(
        (data['en_target_word'] if args.no_cefr_pos else
         data[['en_target_word', 'en_target_pos']]),
        fill_value=CEFR_C2_NUM
        ).reset_index(drop=True)
    if args.cefrj:
        data['cefrj_level'] = cefrj['level'].reindex(
            (data['en_target_word'] if args.no_cefr_pos else
             data[['en_target_word', 'en_target_pos']]),
            fill_value=CEFR_C2_NUM
            ).reset_index(drop=True)
    if args.gse:
        data['gse_level'] = gse['level'].reindex(
            data['en_target_word'],
            fill_value=CEFR_C2_NUM
            ).reset_index(drop=True)
    if args.vxgl:
        data['vxgl'] = vxgl['level'].reindex(
            data['en_target_word'],
            fill_value=MAX_VXGL
            ).reset_index(drop=True)
    data['word_length'] = data['en_target_word'].str.len()  # works better than log

    if glasgow_norms is not None:
        data_gn = glasgow_norms.reindex(data['en_target_word'])
        if glasgow_norms_train_means is None:
            glasgow_norms_train_means = data_gn.mean()  # mean of each norm
        data_gn = data_gn.fillna(glasgow_norms_train_means).reset_index(drop=True)
        for col in data_gn:
            data[col] = data_gn[col]

    if spelling_diff_without_l1_word:
        for m in args.spelling_diff_without_l1_word_models:
            pm = f'spelling_diff_without_l1_word--{m}'
            out_col = f'{pm}_cn_prompting_output'
            lp_col = f'{pm}_cn_prompting_logprobs'
            if out_col not in data:
                print(
                    f'Warning: No prompting output found for {pm}.',
                    file=sys.stdout
                    )
                continue
            new_cols = {}
            if rate_compare_feature_form == 'prob':
                raise Exception(
                    'rate_compare_feature_form=prob not implemented '
                    'for spelling_diff_without_l1_word'
                    )   # TODO (not used anyway with_l1 word gives better results)
            if rate_compare_feature_form == 'soft':
                lp_vals = (
                    data[lp_col] if (lp_col in data) else
                    pd.Series(np.nan, index=data.index)
                    )
                lang2probs = split_rate_compare_soft_probs(
                    lp_vals,
                    data[out_col]
                    )
                for lang, probs in lang2probs.items():
                    for digit in range(1, 6):
                        new_cols[
                            f'{pm}_{lang}_spelling_diff_without_l1_word_p{digit}'
                            ] = probs[:, digit - 1]
            else:
                lang2vals = split_rate_compare_outputs(data[out_col])
                for lang, vals in lang2vals.items():
                    new_cols[f'{pm}_{lang}_spelling_diff_without_l1_word'] = vals
            data = concat_feature_columns(data, new_cols)

    if spelling_diff_with_l1_word:
        for m in args.spelling_diff_with_l1_word_models:
            pm = f'spelling_diff_with_l1_word--{m}'
            # `cn` naming matches run_prompting.py but stores all languages:
            out_col = f'{pm}_cn_prompting_output'
            lp_col = f'{pm}_cn_prompting_logprobs'
            if out_col not in data:
                print(
                    f'Warning: No prompting output found for {pm}.',
                    file=sys.stdout
                    )
                continue
            new_cols = {}
            if rate_compare_feature_form == 'prob':
                t = temperatures[m]
                lp_vals = (
                    data[lp_col] if (lp_col in data) else
                    pd.Series(np.nan, index=data.index)
                    )
                new_cols = prompting_logprobs2nums_split(
                    lp_vals, temperature=t, columns=[
                        f'{pm}_{lang}_spelling_diff_with_l1_word'
                        for lang in SDIFF_LANGUAGES
                        ])
            elif rate_compare_feature_form == 'soft':
                lp_vals = (
                    data[lp_col] if (lp_col in data) else
                    pd.Series(np.nan, index=data.index)
                    )
                lang2probs = split_rate_compare_soft_probs(
                    lp_vals,
                    data[out_col]
                    )
                for lang, probs in lang2probs.items():
                    for digit in range(1, 6):
                        new_cols[
                            f'{pm}_{lang}_spelling_diff_with_l1_word_p{digit}'
                            ] = probs[:, digit - 1]
            else:
                lang2vals = split_rate_compare_outputs(data[out_col])
                for lang, vals in lang2vals.items():
                    new_cols[f'{pm}_{lang}_spelling_diff_with_l1_word'] = vals
            data = concat_feature_columns(data, new_cols)

    # Feature engineering above adds many columns incrementally; copy once to
    # defragment before training/evaluation code starts slicing and appending.
    data = data.copy()

    for lang in args.l1s:
        # Negligible gain:
        # (Idea: is word with the same spelling in L1, e.g. "kickboxing"?)
        # TODO: Use LLMs?
        # data[f'{lang}_wordfreq'] = data['en_target_word'].map(
        #     lambda w: zipf_frequency(w, LANG2ISO[lang]) > 2
        #     ) if (lang != 'cn') else 0
        # Negligible:
        # if (lang == 'cn'):
        #     def drop_small(x):
        #         return x>=2
        #     data[f'{lang}_similarity'] = (pd.read_csv(
        #         'chinese-en.csv', index_col='word', na_filter=False
        #         )['freq'] >= 2).reindex(
        #             data['en_target_word'],
        #             fill_value=0
        #             ).reset_index(drop=True)
        # else:
        data[f'{lang}_similarity'] = 0 if (lang == 'cn') else list(starmap(
            lambda en_w, l1_w: similarity_to_l1(
                en_w, l1_w, transform=args.similarity_transform
                ),
            data[['en_target_word', f'{lang}_L1_source_word']].itertuples(
                index=False)
            ))
        if args.exact_match:
            data[f'{lang}_exact_match'] = list(starmap(
                lambda en_w, l1_w: similarity_to_l1(en_w, l1_w, exact=True),
                data[['en_target_word', f'{lang}_L1_source_word']].itertuples(
                    index=False)
                ))
        for p in args.trickiness_prompts:
            for m, t in zip(args.trickiness_models,
                            args.trickiness_temperatures):
                pm = f'{p}--{m}'
                if trickiness and (f'{pm}_{lang}_prompting_output' in data):
                    t_method = (
                        ('correct' if (p == 'translate') else 'prob')
                        if (trickiness == 'auto') else trickiness)
                    if t_method == 'correct':
                        data[f'{pm}_{lang}_trickiness'] = (
                            clean_prompting_output(
                                data[f'{pm}_{lang}_prompting_output'], prompt=p
                                ) != data['en_target_word']
                            )
                    else:
                        assert t_method == 'prob'
                        data[f'{pm}_{lang}_trickiness'] = opt_logit.trickiness(
                            1 - prompting_logprobs2prob(
                                data[f'{pm}_{lang}_prompting_logprobs'],
                                data['en_target_word'], prompt=p, temperature=t
                                )
                            )
        for p in args.difficulty_prompts:
            for m, t in zip(args.difficulty_models, args.difficulty_temperatures):
                pm = f'{p}--{m}'
                if args.difficulty and (f'{pm}_{lang}_prompting_output' in data):
                    data[f'{pm}_{lang}_difficulty'] = opt_logit.difficulty(
                        prompting_logprobs2num(
                            data[f'{pm}_{lang}_prompting_logprobs'],
                            temperature=t
                            )
                        )
        # --- Calque feature (probability of "1") ---
        if calque:
            for p in args.calque_prompts:
                for m in args.calque_models:
                    t = temperatures[m]
                    pm = f'{p}--{m}'
                    out_col = f'{pm}_{lang}_prompting_output'
                    lp_col  = f'{pm}_{lang}_prompting_logprobs'
                    if lp_col not in data:
                        print(f'Warning: {lp_col} missing.', file=sys.stdout)
                    if lp_col in data:
                        # probability that the model output is "1"
                        data[f'{pm}_{lang}_calque'] = opt_logit.calque(
                            prompting_logprobs2prob(
                                data[lp_col],
                                '1' if (p == 'calque') else 'yes',
                                prompt='calque', temperature=t
                                )
                            )
                    elif out_col in data:
                        # fallback: hard 0/1 from output
                        outs = clean_prompting_output(data[out_col], prompt='calque')
                        data[f'{pm}_{lang}_calque'] = (outs == '1')
                    # FOR DEBUGGING:
                    # print(f'Example calque feature values for {lang}:')
                    # print(data[[out_col, lp_col, f'{pm}_{lang}_calque']].head())
                    # breakpoint()

        # --- Transliteration feature (probability of "1") ---
        if args.transliteration:
            for m in args.transliteration_models:
                t = temperatures[m]
                pm = f'transliteration--{m}'
                out_col = f'{pm}_{lang}_prompting_output'
                lp_col  = f'{pm}_{lang}_prompting_logprobs'
                if lp_col not in data:
                    print(f'Warning: {lp_col} missing.', file=sys.stdout)
                if lp_col in data:
                    data[f'{pm}_{lang}_transliteration'] = opt_logit.transliteration(
                        prompting_logprobs2prob(
                            data[lp_col],
                            '1', prompt='transliteration', temperature=t
                            )
                        )
                elif out_col in data:
                    outs = clean_prompting_output(data[out_col],
                                                  prompt='transliteration')
                    data[f'{pm}_{lang}_transliteration'] = (outs == '1')

        # --- Lexical ambiguity feature (probability of "1") ---
        if lexical_ambiguity:
            for m in args.lexical_ambiguity_models:
                t = temperatures[m]
                pm = f'lexical_ambiguity--{m}'
                out_col = f'{pm}_{lang}_prompting_output'
                lp_col  = f'{pm}_{lang}_prompting_logprobs'
                if lp_col not in data:
                    print(f'Warning: {lp_col} missing.', file=sys.stdout)
                if lp_col in data:
                    data[f'{pm}_{lang}_lexical_ambiguity'] = (
                        opt_logit.lexical_ambiguity(prompting_logprobs2prob(
                            data[lp_col],
                            '1', prompt='lexical_ambiguity', temperature=t
                            ))
                        )
                elif out_col in data:
                    outs = clean_prompting_output(data[out_col],
                                                  prompt='lexical_ambiguity')
                    data[f'{pm}_{lang}_lexical_ambiguity'] = (outs == '1')

    freq_features = [
        *(f'bnc_{corpus}_log_frequency' for corpus in bnc),
        *(f'lang_8_{lang}_log_frequency' for lang in lang_8_l1s),
        *(f'tubelex_log_{freq_or_range}' for freq_or_range in tubelex),
        *(f'subimdb_log_{freq_or_range}' for freq_or_range in subimdb)
        ]
    if args.opensubtitles:
        freq_features.append('opensub_log_frequency')
    if args.coca:
        freq_features.append('coca_log_frequency')

    def get_features(lang):
        if args.predict_binned_target:
            return [f'<binned target, #bins: {args.predict_binned_target}>']
        features = []
        if not no_evp:
            features.append('evp_level')
        if not no_frequency:
            features.extend(freq_features)
        if not no_similarity:
            features.append(f'{lang}_similarity')
        if args.exact_match:
            features.append(f'{lang}_exact_match')
        if not no_length:
            features.append('word_length')
        if args.cognet:
            features.append(f'{lang}_has_cognate')
        if args.gse:
            features.append('gse_level')
        if args.cefrj:
            features.append('cefrj_level')
        if args.vxgl:
            features.append('vxgl')
        if glasgow_norms is not None:
            features.extend(glasgow_norms.columns)
        if args.baseline_as_feature:
            features.append(f'{lang}_baseline_{args.baseline_as_feature}_pred')
        if args.finetuned:
            for cfg in args.finetuned_configs:
                features.append(f'{cfg}_{lang}_ftllm_output')
        if trickiness:
            for pm in prompts_and_models(
                args.trickiness_prompts, args.trickiness_models
                ):
                features.append(f'{pm}_{lang}_trickiness')  # No check if present
        # New features from prompting outputs:
        if calque:
            for pm in prompts_and_models(
                args.calque_prompts, args.calque_models
                ):
                features.append(f'{pm}_{lang}_calque')
        if args.transliteration:
            for m in args.transliteration_models:
                pm = f'transliteration--{m}'
                features.append(f'{pm}_{lang}_transliteration')
        if lexical_ambiguity:
            for m in args.lexical_ambiguity_models:
                pm = f'lexical_ambiguity--{m}'
                features.append(f'{pm}_{lang}_lexical_ambiguity')
        if args.difficulty:
            for pm in prompts_and_models(
                args.difficulty_prompts, args.difficulty_models
                ):
                features.append(f'{pm}_{lang}_difficulty')

        if spelling_diff_without_l1_word:
            for m in args.spelling_diff_without_l1_word_models:
                pm = f'spelling_diff_without_l1_word--{m}'
                if rate_compare_feature_form == 'soft':
                    features.extend([
                        f'{pm}_{lang}_spelling_diff_without_l1_word_p{digit}'
                        for digit in range(1, 6)
                        ])
                else:
                    features.append(
                        f'{pm}_{lang}_spelling_diff_without_l1_word'
                        )
        if spelling_diff_with_l1_word:
            for m in args.spelling_diff_with_l1_word_models:
                pm = f'spelling_diff_with_l1_word--{m}'
                if rate_compare_feature_form == 'prob':
                    features.extend([
                        f'{pm}_{lang}_spelling_diff_with_l1_word'
                        for lang in SDIFF_LANGUAGES
                        ])
                elif rate_compare_feature_form == 'soft':
                    features.extend([
                        f'{pm}_{lang}_spelling_diff_with_l1_word_p{digit}'
                        for digit in range(1, 6)
                        ])
                else:
                    features.append(
                        f'{pm}_{lang}_spelling_diff_with_l1_word'
                        )
        return features

    def get_model_cls():
        if args.model == 'ridge':
            return lambda: Ridge(alpha=ridge_alpha)
        return MODELS[args.model]

    model_cls = get_model_cls()

    data_stats = []
    error_agreement = []

    log('And finally...')
    if not args.quiet:
        print(f'Features:           {", ".join(get_features("L1"))}')
        if (args.final_predict is None) and (args.final_eval is None):
            print(f'Frequency features: {", ".join(freq_features)}')
        if args.final_predict is not None:
            print(f'Model:              models/{args.final_predict}')
        elif args.final_eval is not None:
            if args.track:
                print(f'Model:              submission/{args.track}/{args.final_eval}')
            else:
                print(f'Model:              models/{args.final_eval}')
        else:
            model_label = (
                'binned target' if args.predict_binned_target else
                'feature mean' if args.predict_feature_mean else
                (
                    f'binned ridge(alpha={ridge_alpha})'
                    if (args.bin_predictions and args.model == 'ridge') else
                    f'binned ridgecv(alphas={ridgecv_alphas})'
                    if (args.bin_predictions and args.model == 'ridgecv') else
                    f'binned {model_cls}'
                    )
                if args.bin_predictions else
                (
                    f'clipped ridge(alpha={ridge_alpha})'
                    if (args.clip_predictions and args.model == 'ridge') else
                    f'clipped ridgecv(alphas={ridgecv_alphas})'
                    if (args.clip_predictions and args.model == 'ridgecv') else
                    f'clipped {model_cls}'
                    )
                if args.clip_predictions else
                (
                    f'ridge(alpha={ridge_alpha})'
                    if args.model == 'ridge' else
                    f'ridgecv(alphas={ridgecv_alphas})'
                    if args.model == 'ridgecv' else
                    f'{model_cls}'
                    )
                )
            print(f'Model:              {model_label}')
            print(f'Metric:             {args.metric}')

        if args.cv:
            mode_suffix = (
                '' if args.cv_mode == 'whole' else f', mode={args.cv_mode}'
                )
            print(f'Cross-validated on: {args.splits} '
                  f'({len(cv)} splits{mode_suffix}).')
        else:   # TODO
            print(f'Trained on:         {args.train}, n={len(cv[0].train)}')
            print(f'Evaluated on:       {args.eval}, n={len(cv[0].dev)}')
        if final_train_idx is not None:
            folds = ', '.join(str(fold) for fold in selected_final_train_folds)
            print(f'Final-train folds:  {folds} (union of dev subsets)')
            print(f'Final-train n:      {len(final_train_idx)}')
        print()
        if args.final_predict is not None:
            print(
                f'Predictions: {args.final_predict}\n'
                f'================================='
                )
        elif args.final_eval is not None:
            print(
                f'Results: {", ".join(args.eval_metrics)}\n'
                f'=========================='
                )
            eval_metric_names = '\t'.join(args.eval_metrics)
            print(f'L1\t{eval_metric_names}')
        else:
            print(
                f'Results: {args.metric}{cv_suffix}\n'
                f'================'
                )

    model = make_model(
        model_cls,
        bin_predictions=args.bin_predictions,
        clip_predictions=args.clip_predictions
        )
    freq_model = (
        model_cls()
        if args.model != 'ridgecv'
        else None
        )
    metric = METRICS[args.metric]
    scorer = metric2scorer(args.metric)

    def scores_rep(scores, decimals=args.decimals, sd=args.sd):
        if not METRIC_IS_SCORE[args.metric]:
            scores = -scores                # cross_validate returns neg. loss (RMSE)
        if len(scores) == 1:
            return f'{scores[0]:.{decimals}f}'
        return (
            f'{scores.mean():.{decimals}f}±{scores.std():.{decimals}f}' if sd else
            f'{scores.mean():.{decimals}f}'
            )

    orig_train = data.iloc[data_cv.original_split.train]
    orig_dev = data.iloc[data_cv.original_split.dev]
    eval_df = data.iloc[cv[0].dev]
    coef_tables = []
    coefs_supported = (args.model != 'ridgecv')
    shap_detailed_rows = []
    shap_summary_rows = []
    final_train_saved = []
    final_train_predict_cmd = None

    if args.final_train is not None:
        existing_paths = []
        for lang in args.l1s:
            model_out = Path(f'models/{args.final_train}_{lang}.model')
            info_out = model_out.with_name(model_out.stem + '_info.txt')
            for out_path in (model_out, info_out):
                if out_path.exists():
                    existing_paths.append(out_path)
        if existing_paths and (not args.overwrite_final_model):
            existing_text = ', '.join(str(p) for p in existing_paths)
            raise FileExistsError(
                'Refusing to overwrite existing final model output(s): '
                f'{existing_text}. Use --overwrite-final-model to overwrite.'
                )

    for lang in args.l1s:
        target_col = f'{lang}_GLMM_score'
        y = data[target_col]

        def model_path(model_name: str) -> Path:
            return Path(f'models/{model_name}_{lang}.model')

        def maybe_compute_shap(
            explain_df,
            background_df,
            fitted_model=None
            ):
            if not args.output_shap:
                return
            if args.predict_binned_target or args.predict_feature_mean:
                print(
                    f'Warning: Skipping SHAP for {lang}; unsupported prediction mode.',
                    file=sys.stderr
                    )
                return
            if args.model == 'ridgecv':
                print(
                    f'Warning: Skipping SHAP for {lang}; unsupported model ridgecv.',
                    file=sys.stderr
                    )
                return
            try:
                shap_detailed, shap_summary = compute_language_shap(
                    lang=lang,
                    item_id_col=ID_COL,
                    model_name=args.model,
                    model_cls=model_cls,
                    features=features,
                    target_col=target_col,
                    orig_train=background_df,
                    orig_dev=explain_df,
                    seed=args.seed,
                    shap_sample_size=args.shap_sample_size,
                    shap_background_size=args.shap_background_size,
                    bin_predictions=args.bin_predictions,
                    clip_predictions=args.clip_predictions,
                    make_model_fn=make_model,
                    fitted_model=fitted_model
                    )
                if shap_detailed is not None:
                    shap_detailed_rows.append(shap_detailed)
                if shap_summary is not None:
                    shap_summary_rows.append(shap_summary)
            except Exception as exc:
                raise exc
#                 print(
#                     f'Warning: Failed to compute SHAP for {lang}: {exc}',
#                     file=sys.stderr
#                     )

        def fit_models_and_score_in_cv(m, fts, print_selected_alphas=False):
            if args.model == 'ridgecv':
                def build_inner_cv(train_indices, skip_fold_idx):
                    train_indices = list(train_indices)
                    train_index_set = set(train_indices)
                    local_idx = {idx: i for i, idx in enumerate(train_indices)}
                    inner_cv = []
                    for inner_fold_idx, inner_split in enumerate(cv):
                        if inner_fold_idx == skip_fold_idx:
                            continue
                        inner_train = [
                            local_idx[idx]
                            for idx in inner_split.train
                            if idx in train_index_set
                            ]
                        inner_dev = [
                            local_idx[idx]
                            for idx in inner_split.dev
                            if idx in train_index_set
                            ]
                        if inner_train and inner_dev:
                            inner_cv.append((inner_train, inner_dev))
                    return inner_cv

                def extract_alpha(fitted_model):
                    est = (
                        fitted_model.estimator_
                        if hasattr(fitted_model, 'estimator_')
                        else fitted_model
                        )
                    return float(est.alpha_)

                fitted_models = []
                fold_scores = []
                selected_alphas = []
                for fold_idx, split in enumerate(cv):
                    inner_cv = build_inner_cv(
                        split.train,
                        skip_fold_idx=fold_idx
                        )
                    if not inner_cv:
                        raise ValueError(
                            'RidgeCV requires at least 2 selected CV splits '
                            'to build train-time inner CV splits.'
                            )
                    ridgecv_model = make_model(
                        lambda: RidgeCV(
                            alphas=ridgecv_alphas,
                            cv=inner_cv
                            ),
                        bin_predictions=args.bin_predictions,
                        clip_predictions=args.clip_predictions
                        )
                    train_x = data.iloc[split.train][fts]
                    train_y = y.iloc[split.train]
                    dev_x = data.iloc[split.dev][fts]
                    dev_y = y.iloc[split.dev]
                    fitted = ridgecv_model.fit(train_x, train_y)
                    dev_pred = fitted.predict(dev_x)
                    score = metric(dev_y, dev_pred)
                    if not METRIC_IS_SCORE[args.metric]:
                        score = -score
                    fitted_models.append(fitted)
                    fold_scores.append(score)
                    selected_alphas.append(extract_alpha(fitted))
                if print_selected_alphas:
                    alpha_list = ', '.join(f'{a:g}' for a in selected_alphas)
                    print(f'{lang}: selected alpha(s): {alpha_list}')
                return (
                    fitted_models,
                    np.array(fold_scores),
                    )
            r = cross_validate(
                m, data[fts], y,
                scoring=scorer, cv=cv, n_jobs=(args.jobs if args.cv else 1),
                return_estimator=True
                )
            return (
                r['estimator'],     # Fitted models (1 per split)
                r['test_score'],    # Array of scores (1 per split)
                )

        def score_baseline(baseline_col: str):
            # Array containing a single score (original split)
            s = metric(
                orig_dev[target_col],
                orig_dev[baseline_col]
                )
            if not METRIC_IS_SCORE[args.metric]:
                s = -s                              # like cross_validate
            return np.array([s])

        def predict_original_split(m, fts):
            # original split is subject to --eval
            if args.model == 'ridgecv':
                train_indices = list(data_cv.original_split.train)
                train_index_set = set(train_indices)
                local_idx = {idx: i for i, idx in enumerate(train_indices)}
                inner_cv = []
                for split in cv:
                    inner_train = [
                        local_idx[idx] for idx in split.train if idx in train_index_set
                        ]
                    inner_dev = [
                        local_idx[idx] for idx in split.dev if idx in train_index_set
                        ]
                    if inner_train and inner_dev:
                        inner_cv.append((inner_train, inner_dev))
                if not inner_cv:
                    raise ValueError(
                        'RidgeCV requires at least 2 selected CV splits '
                        'to build train-time inner CV splits.'
                        )
                ridgecv_model = make_model(
                    lambda: RidgeCV(
                        alphas=ridgecv_alphas,
                        cv=inner_cv
                        ),
                    bin_predictions=args.bin_predictions,
                    clip_predictions=args.clip_predictions
                    )
                return ridgecv_model.fit(
                    orig_train[fts],
                    orig_train[target_col]
                    ).predict(orig_dev[fts])
            return m.fit(
                orig_train[fts], orig_train[target_col]
                ).predict(orig_dev[fts])

        if args.predict_binned_target:
            features = [target_col]
            model = BinnedRegressor(PassthroughRegressor(), args.predict_binned_target)
        else:
            features = get_features(lang)
            missing_features = [f for f in features if (f not in data)]
            if missing_features:
                print(f'Warning: Features missing in data will be ignored: '
                      f'{", ".join(missing_features)}.'
                      )
            features = [f for f in features if (f in data)]
            if args.predict_feature_mean:
                model = PassthroughRegressor()

        if not features:
            raise Exception('No features. Cannot fit the model.')

        if args.final_predict is not None:
            path = model_path(args.final_predict)
            model = load(path)
            test_pred = model.predict(eval_df[features])
            maybe_compute_shap(
                explain_df=eval_df,
                background_df=orig_train,
                fitted_model=model
                )
            if args.no_save:
                print(f'Computed final predictions for {lang} (not saved).')
            else:
                out_path = (
                    Path(args.submission_directory) /
                    args.track /
                    lang /
                    f'predictions_{args.final_predict}.csv'
                    )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out = pd.DataFrame({
                    ID_COL: eval_df[ID_COL],
                    'prediction': test_pred
                    })
                out.to_csv(out_path, index=False)
                print(f'Wrote final predictions for {lang}: {out_path}')
            continue    # TODO?

        if args.final_eval is not None:
            if not args.track:
                path = model_path(args.final_eval)
                model = load(path)
                test_pred = model.predict(eval_df[features])
                maybe_compute_shap(
                    explain_df=eval_df,
                    background_df=orig_train,
                    fitted_model=model
                    )
            else:
                pred_path = (
                    Path(args.submission_directory) /
                    args.track /
                    lang /
                    f'predictions_{args.final_eval}.csv'
                    )
                test_pred = pd.read_csv(pred_path, index_col=ID_COL)['prediction']
                if args.output_shap:
                    print(
                        f'Warning: Skipping SHAP for {lang}; --track uses saved '
                        'predictions with no fitted model to explain.',
                        file=sys.stderr
                        )

            test_y = eval_df[target_col]

            fmt_eval_metrics = '\t'.join(
                f'{metric(test_y, test_pred):.{args.decimals}f}'
                for metric_name in args.eval_metrics
                for metric in (METRICS[metric_name],)
                )
            print(f'{lang}\t{fmt_eval_metrics}')
            continue

        models, scores = fit_models_and_score_in_cv(
            model,
            features,
            print_selected_alphas=(args.model == 'ridgecv')
            )
        __, freq_scores = fit_models_and_score_in_cv(freq_model, freq_features)

        final_model = None
        if args.final_train is not None:
            if final_train_idx is not None:
                final_model = make_model(
                    model_cls,
                    bin_predictions=args.bin_predictions,
                    clip_predictions=args.clip_predictions
                    )
                final_model.fit(
                    data.iloc[final_train_idx][features],
                    data.iloc[final_train_idx][target_col]
                    )
            else:
                assert len(models) == 1
                final_model = models[0]
            path = model_path(args.final_train)
            dump(final_model, path)
            info_path = path.with_name(path.stem + '_info.txt')
            train_cmd = cli_command(sys.argv)
            predict_cmd = final_predict_command(sys.argv, args.final_train)
            fold_info = (
                ''
                if selected_final_train_folds is None else
                f'Final-train folds (dev union): {selected_final_train_folds}\n\n'
                )
            info_path.write_text(
                f'{fold_info}'
                f'Final-train command:\n{train_cmd}\n\n'
                f'Final-predict command:\n{predict_cmd}\n',
                encoding='utf-8'
                )
            final_train_saved.append((lang, path, info_path))
            final_train_predict_cmd = predict_cmd

        if args.output_coefs and coefs_supported:
            if args.final_train is not None:
                # Use the exact fitted model that is saved by --final-train.
                coef_series = fitted_lr_coefs(final_model, features)
                coef_table = (
                    None
                    if coef_series is None
                    else pd.DataFrame({lang: coef_series})
                    )
            else:
                coef_table = fit_coefs_table(
                    lang,
                    target_col,
                    features,
                    data,
                    cv,
                    orig_train,
                    model_cls,
                    use_cv=args.cv,
                    predict_binned_target=args.predict_binned_target,
                    predict_feature_mean=args.predict_feature_mean,
                    bin_predictions=args.bin_predictions,
                    clip_predictions=args.clip_predictions
                    )
            if coef_table is None:
                coefs_supported = False
            else:
                coef_tables.append(coef_table)

        # The following does not use CV (original split only):
        output_str = (
            f'{lang}: features: {scores_rep(scores)}'
            f' frequency: {scores_rep(freq_scores)}'
            )
        if args.final_predict is None:
            # Transformer baselines:
            b_open_scores = score_baseline(f'{lang}_baseline_open_pred')
            b_closed_scores = score_baseline(f'{lang}_baseline_closed_pred')
            output_str += (
                f' / baselines: open: {scores_rep(b_open_scores)}'
                f' closed: {scores_rep(b_closed_scores)}'
                )
        print(output_str)

        target = orig_dev[target_col]
        pred = predict_original_split(model, features)
        freq_pred = predict_original_split(model, freq_features)

        maybe_compute_shap(
            explain_df=orig_dev,
            background_df=orig_train
            )

        t_stats = SimpleStats(target)
        t_sd = t_stats.sd

        # Errors:

        if args.final_predict is None:
            b_open = orig_dev[f'{lang}_baseline_open_pred']
            b_closed = orig_dev[f'{lang}_baseline_closed_pred']
            orig_dev[f'{lang}_feat_error'] = (pred - target) / t_sd
            orig_dev[f'{lang}_freq_error'] = (freq_pred - target) / t_sd
            orig_dev[f'{lang}_baseline_open_error'] = (b_open - target) / t_sd
            orig_dev[f'{lang}_baseline_closed_error'] = (b_closed - target) / t_sd

            for i, m in enumerate(('feat', 'freq')):
                pcc_err = pearsonr(
                    orig_dev[f'{lang}_{m}_error'],
                    orig_dev[f'{lang}_baseline_closed_error']
                    ).statistic
                if not args.quiet:
                    same_sign_err = (
                        (orig_dev[f'{lang}_{m}_error'] > 0) ==
                        (orig_dev[f'{lang}_baseline_closed_error'] > 0)
                        ).mean()
                    m_abs_err = orig_dev[f'{lang}_{m}_error'].abs()
                    b_abs_err = orig_dev[f'{lang}_baseline_closed_error'].abs()
                    dev_large_err = orig_dev[
                        (m_abs_err > m_abs_err.mean()) | (b_abs_err > b_abs_err.mean())
                        ]
                    same_sign_large_err = (
                        (dev_large_err[f'{lang}_{m}_error'] > 0) ==
                        (dev_large_err[f'{lang}_baseline_closed_error'] > 0)
                        ).mean()
                    error_agreement.append(
                        (f'{lang}: ' if i == 0 else '    ') +
                        f'{m}: PCC = {pcc_err:.2f}, {same_sign_err:.0%} same sign'
                        f' ({same_sign_large_err:.0%} of above-average errors)'
                        )
        if not args.quiet:
            data_stats.append(f'{lang} target: {t_stats}')
            if args.extended_statistics:
                p_stats = SimpleStats(pred)
                data_stats.append(f'{lang} pred:   {p_stats}')

    if not args.quiet:
        if error_agreement:
            print(
                f'\n'
                f'Error correlation/same-signedness vs. open baseline{dev_suffix}\n'
                f'===================================================\n' +
                '\n'.join(error_agreement)
                )

        if data_stats:
            print(
                f'\n'
                f'Statistics{dev_suffix}\n'
                f'===============\n' +
                '\n'.join(data_stats)
                )

    if args.output_coefs:
        print_coefs_table(
            coef_tables,
            coefs_supported,
            use_cv=args.cv,
            cv_suffix=cv_suffix,
            dev_suffix=dev_suffix,
            decimals=args.decimals
            )

    if args.final_train is not None and final_train_saved:
        saved_models = ', '.join(
            f'{lang}: {path}'
            for lang, path, __ in final_train_saved
            )
        saved_infos = ', '.join(
            f'{lang}: {info_path}'
            for lang, __, info_path in final_train_saved
            )
        print(
            f'\n'
            f'Final models\n'
            f'============\n'
            f'Saved models: {saved_models}.\n'
            f'Saved command info files: {saved_infos}.'
            )
        if final_train_predict_cmd is not None:
            print('\nUse the following command to run final prediction:\n')
            print(final_train_predict_cmd)

    if args.write:
        path = Path('results/features_errors.csv')
        print(f'Writing results to {path}.', file=sys.stderr)
        out = orig_dev.copy()
        obj_cols = out.select_dtypes(include=['object']).columns
        for c in obj_cols:
            out[c] = out[c].astype(str).str.replace(r'[\r\n]+', ' ', regex=True)
        out.to_csv(path)

    if args.output_shap:
        write_shap_outputs(
            output_dir=args.output_shap,
            detailed_rows=shap_detailed_rows,
            summary_rows=shap_summary_rows,
            decimals=args.decimals,
            item_id_col=ID_COL,
            print_shap_summary=args.print_shap_summary,
            shap_format=args.shap_format
            )

    if args.plots:
        errors_feat = [orig_dev[f'{lang}_feat_error'] for lang in args.l1s]
        errors_open_baseline = [
            orig_dev[f'{lang}_baseline_open_error'] for lang in args.l1s
            ]
        n_bins = 20
        colors = ['r', 'g', 'b']

        plt.figure(figsize=(9, 6))
        plt.hist(
            errors_feat,
            bins=n_bins,
            color=colors, histtype='step', linewidth=2,
            label=[LANG2NAME[lang] + '/feat.' for lang in args.l1s]
            )
        plt.hist(
            errors_open_baseline,
            bins=n_bins,
            color=colors, histtype='step', linewidth=1, alpha=0.5,
            label=[LANG2NAME[lang] + '/open' for lang in args.l1s]
            )
        plt.xlabel('Error/SD')
        plt.ylabel('Frequency')
        plt.title('Histogram of Model Errors (normalized by data SD)')
        plt.legend()
        plt.tight_layout()
        path = Path('results/features_errors_plot.pdf')
        print(f'Writing plot to {path}.', file=sys.stderr)
        plt.savefig(path)

    if args.output_large_errors is not None:
        if args.large_errors_l1 is None:
            raise Exception('--output-large-errors requires -large-errors-l1')
        if args.large_errors_model is None:
            raise Exception('--output-large-errors requires -large-errors-model')

        lang = args.large_errors_l1
        model = (f'baseline_{args.large_errors_model}'
                 if args.large_errors_model in {'open', 'closed'}
                 else args.large_errors_model
                 )

        m_abs_err = orig_dev[f'{lang}_{model}_error'].abs()
        dev_large_err = orig_dev[(m_abs_err > m_abs_err.mean())]

        df = dev_large_err[[
            ID_COL,
            'en_target_word', f'{lang}_L1_source_word', f'{lang}_L1_context',
            f'{lang}_{model}_error'
            ]].sort_values(by=f'{lang}_{model}_error')
        df['en_clue'] = df['en_target_word'].map(spaced_clue)
        print(f'Writing items with large errors for L1={args.large_errors_l1}, '
              f'model={model} to {args.output_large_errors}', file=sys.stderr)
        df.to_csv(args.output_large_errors, index=False)


MODELS = {
    # TODO: use model / hyper param search via CV
    'lr': LinearRegression,
    'ridge': Ridge,
    'ridgecv': RidgeCV,
    # max_depth=3, learning_rate=0.1 are defaults for gbr, seems optimal:
    # n_estimators defaults to 100, slightly more works better:
    'gbr': lambda: GradientBoostingRegressor(n_estimators=150),
    # xbr gives **slightly** better results:
    'xgbr': lambda: XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=200),
    'svr': SVR
    }

GLASGOW_NORMS = [
    # arou val dom cnc imag fam aoa size gend
    # cnc imag fam aoa together result in about 0.01 improvement,
    # the rest is negligible/noise
    'arou', 'val', 'dom', 'cnc', 'imag', 'fam', 'aoa', 'size', 'gend'
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', choices=MODELS, default='lr',
                        help=f'Model type: {MODELS}, default: lr.')
    parser.add_argument(
        '--alpha', type=float, nargs='+', default=None,
        help=(
            'Ridge/RidgeCV alpha value(s). For --model ridge, pass one value '
            '(default: 1.0). For --model ridgecv, pass one or more values '
            '(default: 0.1 1.0 10.0).'
            )
        )
    parser.add_argument('--temperatures',
                        default=[
                            ('gpt-5.2', 9),
                            ('gpt-4.1', 4),
                            ('gpt-4.1-mini', 4),
                            ('gpt-4.1-nano', 1.5),
                            ('deepseek-v3.2', 15),
                            ],
                        type=str_float_pair, nargs='*',
                        help=(
                            'Apply temperature scaling to prompt predictions. '
                            'Can be overriden by --trickiness-temperatures or '
                            '--difficulty-temperatures. '
                            'Default: gpt-5.2=9 gpt-4.1=4 gpt-4.1-mini=4 '
                            'gpt-4.1-nano=1.5 deepseek-v3.2=15.'
                            ))
    parser.add_argument('--l1s', choices=L1_CODES, nargs='+', default=L1_CODES)
    predict_special = parser.add_mutually_exclusive_group()
    predict_special.add_argument(
        '--predict-binned-target', '--bt', const=5, type=int, nargs='?',
        help=('Ignore features and predict binned target values. '
              'Default #bins=5')
        )
    predict_special.add_argument(
        '--predict-feature-mean', '--fm', action='store_true',
        help=('Ignore the model and predict the row-wise mean of the selected '
              'feature values.')
        )
    parser.add_argument('--clip-predictions', '--cp', action='store_true',
                        help=('Clip predictions based on training data range'))
    parser.add_argument('--bin-predictions', '--bp', const=5, type=int, nargs='?',
                        help=('Bin predictions. Default #bins=5.'))

    cv_group = parser.add_mutually_exclusive_group()
    cv_group.add_argument(
        '--final-train',
        help='Train on full data and save model under a name.',
        )
    cv_group.add_argument(
        '--final-predict',
        help=('Predict on test data using a saved model. '
              'Uses models finetuned on full data. Requires either --track '
              'or --no-save.'),
        )
    cv_group.add_argument(
        '--final-eval',
        help=('Evaluate using a model saved via --final-train (without --track), '
              'or using submission data saved via --final-predict (with --track).'),
        )
    final_predict_output = parser.add_mutually_exclusive_group()
    final_predict_output.add_argument(
        '--track', choices=['open', 'closed'],
        help='Track name for submission output path.'
        )
    final_predict_output.add_argument(
        '--no-save', action='store_true',
        help='With --final-predict, compute predictions but do not save files.'
        )
    parser.add_argument(
        '--overwrite-final-model', action='store_true',
        help=('Allow --final-train to overwrite existing *_lang.model and '
              '*_lang_info.txt files.')
        )
    parser.add_argument(
        '--final-train-folds', type=int, nargs='+',
        help=('Only with --final-train. Use the union of selected CV fold dev '
              'subsets as training data (1-based fold numbers).')
        )
    cv_group.add_argument(
        '--cv', action='store_true',
        help='Cross-validate using --splits and --jobs.'
        )
    parser.add_argument('--cv-mode', choices=['whole', 'first', 'remaining'],
                        default='whole',
                        help=('CV fold selection mode. Default: whole. '
                              'Use "first" to run only fold 1, or "remaining" '
                              'to skip fold 1.'))
    parser.add_argument('--sd', action='store_true',
                        help='Print standard deviation for CV scores.')
    parser.add_argument('--decimals', type=int, default=3,
                        help='Decimals to print. Default: 3.')
    parser.add_argument(
        '--splits', default='data/cv-split-ids-5.json',
        help='Splits for cross-validation (--cv). Default: data/cv-split-ids-5.json.')
    parser.add_argument(
        '--jobs', '-j', default=-1,
        help='Use n jobs for cross-validation. Default: -1 => #CPUs.')
    parser.add_argument('--train', '-t', choices=SUBSETS, default=None)
    parser.add_argument('--eval', '-e', choices=SUBSETS, default=None)
    parser.add_argument('--seed', type=int_or_none, default=42,
                        help='Integer or "no" to use entropy. Default: 42.'
                        )
    parser.add_argument('--metric', choices=METRICS, default='pearson')
    parser.add_argument('--eval-metrics',
                        choices=METRICS, nargs='+', default=['rmse', 'pearson', 'r2'],
                        help='Metrics for --final-eval.')
    parser.add_argument('--submission-directory', default='submission',
                        help='Submission directory name. Default: submission.')

    output = parser.add_argument_group('Output options')
    #     parser.add_argument('--write-merged', action='store_true',
    #                         help='Write merged train/dev data and exit.')
    output.add_argument('--quiet', '-q', action='store_true',
                        help='Print only results (correlations).')
    output.add_argument('--verbose', '-v', action='store_true',
                        help='Extra logging.')
    output.add_argument('--extended-statistics', '-x', action='store_true',
                        help='Print more stats.')
    output.add_argument('--write', action='store_true',
                        help='Write out prediction errors for analysis.')
    output.add_argument('--plots', action='store_true',
                        help='Write out plots.')
    output.add_argument('--output-large-errors',
                        help='Write out items with a large error to a file.')
    output.add_argument('--large-errors-l1', '-L', choices=L1_CODES)
    output.add_argument('--large-errors-model', '-M',
                        choices=['feat', 'freq', 'open', 'close'])
    output.add_argument(
        '--output-coefs', '--coefs', action='store_true',
        help=('Print learned coefs for the features model '
              '(includes intercept for LR).')
        )
    output.add_argument('--output-tricky', action='store_true',
                        help='Write tricky questions to files.')
    output.add_argument(
        '--output-shap',
        help=('Write SHAP explanations to this output directory '
              '(shap_detailed.csv, shap_detailed_summary.csv, '
              'shap_grouped.csv, shap_grouped_summary.csv).')
        )
    output.add_argument(
        '--shap-sample-size', type=int_or_none, default=None,
        help=('Max number of dev rows to explain per language for SHAP. '
              'Default: no limit.')
        )
    output.add_argument(
        '--shap-background-size', type=int_or_none, default=None,
        help=('Background sample size per language for SHAP explainers. '
              'Default: no limit.')
        )
    output.add_argument(
        '--print-shap-summary', '-S',
        choices=['detailed', 'grouped', 'no'],
        default='detailed',
        help=('When running SHAP, print detailed summary, grouped summary, '
              'or no SHAP summary to stdout. Default: detailed.')
        )
    output.add_argument(
        '--shap-format',
        choices=['full', 'simple'],
        default='full',
        help=('Output format for non-summary SHAP CSVs. '
              'full=long format, simple=wide format with one column per '
              'feature/group. Default: full.')
        )

    opt = parser.add_argument_group('Optional features',
                                    'These features are not active by default.')
    opt.add_argument('--baseline-as-feature', '-B', choices=['open', 'closed'],
                     help='Add the baseline model\'s prediction as a feature.')
    opt.add_argument('--finetuned', '-f', action='store_true',
                     help='Add finetuned LLM predictions as features.')

    fc_short = ', '.join(FINETUNED_CONFIG_SHORT_NAMES.keys())
    opt.add_argument(
        '--finetuned-configs', '--fc', nargs='+',
        default=['ministral-3-14b', 'glm-4-32b', 'qwen2.5-32b'],
        help=(f'Finetuned LLM prediction configs to use. Default: '
              f'ministral-3-14b glm-4-32b qwen2.5-32b. '
              f'Available short names: {fc_short}')
        )
    opt.add_argument('--gse', '-G', action='store_true',
                     help='Add GSE CEFR levels as a feature (default: only EVP).')
    # GSE: no improvement when added to EVP (also doesn't have POS)
    opt.add_argument('--cefrj', '-J', action='store_true',
                     help='Add CEFR-J CEFR levels as a feature (default: only EVP).')
    # CEFR-J: no improvement when added to EVP, but seems slightly better than GSE
    opt.add_argument('--vxgl', '-V', action='store_true',
                     help='Add VXGL grade levels as a feature.')
    # VXGL: no/negligible improvemnt, (--gn aoa works better)
    opt.add_argument('--opensubtitles', '--os', action='store_true',
                     help='Add English OpenSubtitles frequencies as a feature.')
    # OpenSubtitles: negligible improvement when added to BNC, TUBELEX, Lang-8
    opt.add_argument('--subimdb', choices=['frequency', 'range', 'no'], nargs='+',
                     default=[],
                     help='Add SubIMDB frequency/range (videos). Default: no.')
    # SubIMDB: negligible improvement when added to BNC, TUBELEX, Lang-8
    opt.add_argument('--coca', choices=COCA_COL2TOTAL,
                     help='Add COCA top 5000 frequency or range as a feature.')
    # TODO: SUBTLEX-UK? UK, has range, has POS
    # COCA: only top 5000 words are freely available, not great. TUBELEX is better.
    opt.add_argument('--glasgow-norms', '--gn', choices=GLASGOW_NORMS, nargs='+',
                     help='Add Glasgow norms as features.')
    # Glasgow norms cnc imag fam aoa together result in about 0.01 improvement
    opt.add_argument('--cognet', action='store_true',
                     help=('Add presence of a cognate pair with L1 in CogNet '
                           'as a feature.'))
    opt.add_argument('--exact-match', action='store_true',
                     help=('Add exact match with L1 word (except for accents) '
                           'as a feature.'))
    # CogNet has negligible effect. It only improves German by 0.01 *if* we do not use
    # similarity (--no-sim --cognet), and by 0.02 *if* we do not use similarity and
    # trickiness (--no-sim -Tno --cognet).
    opt.add_argument('--cognate-must-differ', action='store_true',
                     help=('The L1 cognate word must not be exactly the same as the '
                           'English word to count')
                     )
    # must differ: basicaly no effect
    opt.add_argument(
        '--transliteration', '--tl', action='store_true',
        help='Add presence of a transliteration pair with L1 as a feature.'
        )
    opt.add_argument('--transliteration-models', '--tlm',
                     choices=['gpt-5.2'], nargs='+',
                     default=['gpt-5.2'],  # Only tested gpt-5.2
                     help='Default: gpt-5.2.')
    # transliteration doesn't help
    opt.add_argument(
        '--difficulty', '-D', action='store_true',
        help='Add difficulty prompt output as a feature.'
        )

    opt.add_argument(
        '--spelling-diff-without-l1-word', '--sdiff', action='store_true',
        help=('Add language-specific spelling difficulty rate-compare feature '
              'parsed from cn_prompting_output as cn,es,de values.'
              'This version does not use the L1 translation word in the prompt. ')
        )
    opt.add_argument(
        '--spelling-diff-without-l1-word-models', '--sdiff-m',
        choices=['gpt-5.2', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano'],
        nargs='+', default=['gpt-5.2'],
        help='Default: gpt-5.2.'  # only tested with gpt-5.2
        )
    opt.add_argument(
        '--spelling-diff-with-l1-word', '--sdiff-l1', action='store_true',
        help=('Add language-specific spelling difficulty rate-compare feature '
              'parsed from cn_prompting_output as cn,es,de values. '
              'This version use L1 translation word in the prompt')
        )
    opt.add_argument(
        '--spelling-diff-with-l1-word-models', '--sdiff-l1-m',
        choices=['gpt-5.2', 'deepseek-v3.2'],
        nargs='+', default=['gpt-5.2'],     # TODO: deepseek: doesn't have full logprobs
        help='Default: gpt-5.2.'
        )
    opt.add_argument(
        '--spelling-diff-feature-form', '--sdff',
        choices=['hard', 'soft', 'prob'],
        default='prob',
        # prob => same G-Eval style weighting as we use for other features,
        # makes no difference in the overall model, and decreases total # features
        help=('Feature form for pdrc/pdrca. hard=use parsed top-1 rating; '
              'soft=use p1..p5 probability features from logprobs. '
              'Default: soft.')
        )

    opt.add_argument('--logit',
                     choices=OptionalLogit.choices, nargs='+', default=[],
                     help='Apply logit to the features.'
                     )
    default = parser.add_argument_group(
        'Default features',
        'These features can be adjusted/deactivated.'
        )
    default.add_argument('--lang-8-l1s', '--l8', choices=[*LANG_8_LANGUAGES, 'no'],
                         nargs='+', default=LANG_8_LANGUAGES,
                         help=(f'Add lang-8 frequencies for the specified L1s. '
                               f'Default: {" ".join(LANG_8_LANGUAGES)}.'))
    default.add_argument('--trickiness', '-T',
                         choices=['auto', 'correct', 'prob', 'no'], default='auto',
                         help=('Use results from prompting to assess trickiness. '
                               'Default: auto (uses prob or correct).'))
    default.add_argument('--trickiness-models', '--tm',
                         choices=['gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano'], nargs='+',
                         default=['gpt-4.1-mini', 'gpt-4.1-nano'],    # best two
                         help='Default: gpt-4.1-mini gpt-4.1-nano.'
                         )
    default.add_argument('--trickiness-prompts', '--tp',
                         choices=['solve', 'long-solve', 'short-solve', 'terse-solve',
                                  'translate'], nargs='+',
                         default=['solve', 'long-solve'],    # best two
                         help='Default: solve long-solve.'
                         )
    default.add_argument('--trickiness-temperatures', '--tt',
                         default=None, type=float, nargs='+',
                         help=('Apply custom temperature scaling to trickiness '
                               '(1 value per model). Default: no.'))
    # Trickiness (-Tauto) helps a lot

    default.add_argument('--difficulty-models', '--dm',
                         choices=['gpt-5.2', 'gpt-4.1'],
                         nargs='+', default=['gpt-5.2', 'gpt-4.1'],
                         help='Default: gpt-5.2 gpt-4.1.')
    default.add_argument('--difficulty-prompts', '--dp', choices=[
        '3s-difficulty', 'zs-difficulty'
        # Small-scale experiments with hardness. We only have predictions for:
        # --dm gpt-4.1 --cv-mode first --l1s es --train dev
        # '3s-difficulty+easy', '3s-difficulty+hard', '3s-difficulty+both',
        # '5s-difficulty+easy'  # best (but similar to 3s-difficulty+easy)
        ], nargs='+', default=['3s-difficulty'], help='Default: 3s-difficulty.')
    default.add_argument('--difficulty-temperatures', '--dt',
                         default=None, type=float, nargs='+',
                         help=('Apply custom temperature scaling to difficulty '
                               '(1 value per model). Default: no.'))

    # New features from prompting outputs:
    default.add_argument('--calque-models', '--cqm',
                         choices=['gpt-5.2', 'deepseek-v3.2'], nargs='+',
                         default=['gpt-5.2'],  # deepseek-v3.2: only morphtran, not good
                         help='Default: gpt-5.2.')
    default.add_argument('--calque-prompts', '--cqp',
                         choices=['calque', 'morphtran'], nargs='+',
                         default=['calque'],
                         help='Default: calque.')
    default.add_argument('--no-calque', '--no-cq', action='store_true',
                         help='Remove presence of a calque pair with L1 from features.')
    default.add_argument('--lexical-ambiguity-models', '--lam',
                         choices=['gpt-5.2', 'deepseek-v3.2'], nargs='+',
                         default=[
                             'deepseek-v3.2',
                             'gpt-5.2'
                             ],  # deepseek-v3.2 > gpt-5.2
                         help='Default: deepseek-v3.2 gpt-5.2.')
    default.add_argument(
        '--no-lexical-ambiguity', '--no-la', action='store_true',
        help='Remove presence of a lexical ambiguity pair with L1 from features.'
        )
    # Both calque and lexical ambiguity are helpful.
    default.add_argument('--no-cefr-pos', action='store_true',
                         help='Ignore POS for CEFR word lookup (EVP, CEFR-J)')
    # POS helps for both EVP and CEFR-J across languages.
    default.add_argument('--similarity-transform', choices=SIMILARITY_TRANSFORMS,
                         default='square',  # more elegant than cutoff, same result
                         help='Cutoff simply cuts off values lower than 1/3.')
    default.add_argument('--tubelex', choices=['frequency', 'range', 'no'],
                         nargs='+', default=['frequency', 'range'],
                         help='Use frequency/range (channels). Default: both.')
    default.add_argument('--bnc', choices=['spoken', 'written', 'total', 'no'],
                         nargs='+',
                         default=['spoken'],
                         help=('Use spoken/written BNC frequencies. Default: spoken.'))
    default.add_argument('--no-evp', action='store_true',
                         help='Remove EVP CEFR levels from features.')
    default.add_argument('--no-frequency', action='store_true',
                         help='Remove all frequencies from features.')
    default.add_argument('--no-length', action='store_true',
                         help='Remove word length from features.')
    default.add_argument('--no-similarity', action='store_true',
                         help='Remove L1-English word similarity from features.')
    default.add_argument(
        '--no-prompting', action='store_true',
        help='Remove prompting-based features (trickiness, calque, lexical ambiguity).'
        )
    default.add_argument(
        '--provider', choices=['deepinfra', 'deepseek'], default='deepinfra',
        help='Preferred provider for prompting-based features. Default: deepinfra'
        )
    default.add_argument(
        '--no-default-features', action='store_true',
        help=('Equivalent to --no-evp --no-frequency --no-length '
              '--no-similarity --no-prompting.')
        )
    # Written BNC: < 0.01 improvement
    # All of the above defaults are helpful

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
