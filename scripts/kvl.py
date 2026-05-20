import pandas as pd
from functools import reduce
from pathlib import Path
import sys
import json
from typing import NamedTuple

L1_CODES = ['cn', 'de', 'es']
LANG2NAME = {'cn': 'Chinese', 'de': 'German', 'es': 'Spanish'}
LANG2ISO = {'cn': 'zh', 'de': 'de', 'es': 'es'}

SUBSETS = ['train', 'dev']
ALL_SUBSETS = {'train', 'dev', 'full', 'test'}
ID_COL = 'item_id'
DROP_COLS = ['L1', 'en_target_clue']
COMMON_COLS = ['en_target_word', 'en_target_pos']
SPECIFIC_COLS = ['L1_source_word', 'L1_context', 'GLMM_score']


def data_path(subset, lang, no_labels=False):
    subset_dir = (
        (subset + '_no_labels') if no_labels else
        subset
        )
    return Path('data') / subset_dir / lang / f'kvl_shared_task_{lang}_{subset}.csv'


def baseline_pred_path(subset, lang, closed=False):
    return (
        Path('predictions') /
        ('closed' if closed else 'open') /
        subset / lang /
        (f'baseline_closed_{lang}_preds.csv' if closed else
         'baseline_open_xx_preds.csv')
        )


def prompting_pred_path(subset, prompt_model, provider=None):
    d = Path('predictions') / 'prompting' / subset
    def paths_by_preference():
        for with_provider in ((True, False) if (provider is not None) else (False,)):
            raw = d / (f'{prompt_model}--{provider}.csv' if with_provider else
                       f'{prompt_model}.csv')
            yield raw.with_suffix('.csv.xz')
            yield raw
    for p in paths_by_preference():
        if p.exists():
            return p
    # no path exists, but we return anyway, fail and report later:
    return p


def finetuned_llm_pred_path(subset, config):
    return Path('predictions') / 'finetuned_llm' / subset / f'{config}.csv'


def spaced_clue(w):
    return ' '.join('_' if i else c for i, c in enumerate(w))


def read_subset(
    subset,
    baseline_preds=None,
    prompting_preds=None,
    finetuned_llm_preds=None,
    strict_missing=False,
    finetuned_llm_short_names=None,
    provider=None,
    no_labels=False
    ):
    # Read:
    lang_dfs = [pd.read_csv(data_path(subset, lang, no_labels), index_col=ID_COL)
                for lang in L1_CODES]
    for df in lang_dfs:
        if 'GLMM_score' not in df.columns:
            df['GLMM_score'] = float('nan')
    # Fix (error in data: "TRUE", "FALSE"):
    for col in COMMON_COLS:
        lang_dfs[0][col] = lang_dfs[0][col].str.lower()
    # Merge:
    for df in lang_dfs[1:]:
        df.drop(COMMON_COLS, axis=1, inplace=True)
    for lang, df in zip(L1_CODES, lang_dfs):
        df.rename(columns={col: f'{lang}_{col}' for col in SPECIFIC_COLS}, inplace=True)
        assert (df['L1'] == lang).all()
        df.drop(DROP_COLS, axis=1, inplace=True)
    if baseline_preds is not None:
        only_closed = baseline_preds == 'closed'
        for closed, label in zip(*(
            ((False, True), ('open', 'closed')) if baseline_preds == 'both' else
            ((only_closed,), ('closed' if only_closed else 'open',))
            )):
            for lang in L1_CODES:
                pred_path = baseline_pred_path(subset, lang, closed=closed)
                pred_df = pd.read_csv(pred_path, index_col=ID_COL)
                lang_dfs.append(pred_df.rename(
                    columns={'prediction': f'{lang}_baseline_{label}_pred'})
                    )
    if prompting_preds:
        for pm in (prompting_preds):
            if subset == 'test':
                pm_path = prompting_pred_path(subset, pm, provider)
            else:
                pm_path_whole = prompting_pred_path('whole', pm, provider)
                pm_path = (
                    pm_path_whole if pm_path_whole.exists()
                    else prompting_pred_path(subset, pm, provider)
                    )
            if not pm_path.exists():
                msg = (
                    f'Prompting predictions file {pm_path} missing. '
                    f'This prompt-model combination will not be included in {subset}.'
                    )
                if strict_missing:
                    raise FileNotFoundError(msg)
                print(f'Warning: {msg}', file=sys.stderr)
                continue
            lang_dfs.append(pd.read_csv(
                pm_path, index_col=ID_COL,
                converters={
                    f'{lang}_prompting_logprobs': json.loads for lang in L1_CODES
                    }
                ).rename(
                    columns={
                        f'{lang}_prompting_logprobs':
                        f'{pm}_{lang}_prompting_logprobs'
                        for lang in L1_CODES
                        }
                ).rename(
                    columns={
                        f'{lang}_prompting_output':
                        f'{pm}_{lang}_prompting_output'
                        for lang in L1_CODES
                        }
                    )
                )
    if finetuned_llm_preds:
        for cfg in finetuned_llm_preds:
            cfg_full_name = (
                cfg if (finetuned_llm_short_names is None) else
                finetuned_llm_short_names.get(cfg, cfg)
                )
            if subset == 'test':
                cfg_path = finetuned_llm_pred_path(subset, cfg_full_name)
            else:
                cfg_path_whole = finetuned_llm_pred_path('whole', cfg_full_name)
                cfg_path = (
                    cfg_path_whole if cfg_path_whole.exists()
                    else finetuned_llm_pred_path(subset, cfg_full_name)
                    )
            if not cfg_path.exists():
                msg = (
                    f'Finetuned LLM predictions file {cfg_path} missing. '
                    f'This config will not be included in {subset}.'
                    )
                if strict_missing:
                    raise FileNotFoundError(msg)
                print(f'Warning: {msg}', file=sys.stderr)
                continue
            cfg_df = pd.read_csv(cfg_path, index_col=ID_COL)
            lang_dfs.append(cfg_df.rename(columns={
                f'{lang}_ftllm_output': f'{cfg}_{lang}_ftllm_output'
                for lang in L1_CODES
                }))
    return reduce(
        lambda l, r: pd.merge(l, r, left_index=True, right_index=True), lang_dfs
        ).reset_index()     # don't want to use item_id as index


class TDSplit(NamedTuple):
    train: list[int]
    dev: list[int]


class DataAndSplits(NamedTuple):
    data: pd.DataFrame
    cv: list[TDSplit]
    original_split: TDSplit


def _remap_split(s, *, train='train', eval='dev'):
    subset2ids = dict(zip(SUBSETS, s))
    if 'full' in (train, eval):
        subset2ids['full'] = s[0] + s[1]
    return (subset2ids[train], subset2ids[eval])


def raise_or_warn(msg, warn):
    if warn:
        print(f'Warning: {msg}', file=sys.stderr)
    else:
        raise Exception(msg)


def read_data_cv(
    splits_path=None,
    baseline_preds=None,
    prompting_preds=None,
    finetuned_llm_preds=None,
    strict_missing=False,
    train=None,     # defaults to 'train',
    eval=None,      # defaults to 'dev'
    finetuned_llm_short_names=None,
    provider=None,
    test_labels=False   # No test labels by default
    ) -> DataAndSplits:
    '''
    Return the full train+dev data read via read_subset() and a list of splits that
    can be used as a `cv` parameter in sklearn.

    splits_path=None means using the original train/dev split, possibly altered by the
    train/eval arguments.

    Otherwise, supply a path to JSON splits generated via make_splits.py.
    '''
    train = train or 'train'
    eval = eval or 'dev'
    if train not in ALL_SUBSETS:
        raise Exception(f'Unsupported train subset: {train}')
    if eval not in ALL_SUBSETS:
        raise Exception(f'Unsupported eval subset: {eval}')
    data_train, data_dev = (
        read_subset(
            s,
            baseline_preds=baseline_preds,
            prompting_preds=prompting_preds,
            finetuned_llm_preds=finetuned_llm_preds,
            strict_missing=strict_missing,
            finetuned_llm_short_names=finetuned_llm_short_names,
            provider=provider
            )
        for s in SUBSETS
        )
    data_test = (
        read_subset(
            'test',
            baseline_preds=baseline_preds,
            prompting_preds=prompting_preds,
            finetuned_llm_preds=finetuned_llm_preds,
            strict_missing=strict_missing,
            finetuned_llm_short_names=finetuned_llm_short_names,
            provider=provider,
            no_labels=not test_labels
            )
        if 'test' in (train, eval)
        else None
        )
    assert len(data_train) > len(data_dev)
    original_n = len(data_train) + len(data_dev)

    original_split = [
        data_train[ID_COL].tolist(), data_dev[ID_COL].tolist()
        ]

    if splits_path is None:
        subset2ids = {
            'train': original_split[0],
            'dev': original_split[1],
            'full': original_split[0] + original_split[1],
            }
        if data_test is not None:
            subset2ids['test'] = data_test[ID_COL].tolist()
        elif 'test' in (train, eval):
            raise Exception(
                'Requested train/eval subset "test", but test data is not available.'
                )

        tids = subset2ids[train]
        eids = subset2ids[eval]
        assert ((set(tids) == set(eids)) == (train == eval))
        if train == eval:
            print(
                'Warning: Using the same data for training and evaluation.',
                file=sys.stderr
                )
            # Do not check conditions
        else:
            tid_set = set(tids)
            eid_set = set(eids)
            if not tid_set.isdisjoint(eid_set):
                raise Exception(f'IDs in {train} and {eval} are not disjoint.')
            required_ids = tid_set.union(eid_set)
            all_known_ids = set(data_train[ID_COL]).union(
                data_dev[ID_COL]
                )
            if data_test is not None:
                all_known_ids = all_known_ids.union(data_test[ID_COL])
            if not required_ids.issubset(all_known_ids):
                raise Exception(
                    f'Unknown IDs in {train}/{eval} split assignment.'
                    )
        splits = [[tids, eids]]
    else:
        if 'test' in (train, eval):
            raise Exception(
                'Custom CV splits do not support train/eval assignment to test. '
                'Use splits_path=None for full-train + test prediction.'
                )
        custom_train_eval = (train != 'train' or eval != 'dev')
        with open(splits_path) as f:
            splits = json.load(f)
        # Simple sanity checks:
        all_ids = set(sum(original_split, []))
        if not all(set(sum(split, [])) == all_ids for split in splits):
            raise_or_warn(
                f'Splits in {splits_path} do not agree with item_ids in data.',
                warn=custom_train_eval
                )
        if not all(len(sum(split, [])) == original_n for split in splits):
            raise_or_warn(
                f'Subset sizes in {splits_path} do not match data size.',
                warn=custom_train_eval
                )
        if not all(set(train).isdisjoint(set(dev)) for train, dev in splits):
            raise Exception(
                f'Subsets {splits_path} are not disjoint.'
                )
        if custom_train_eval:
            print('Warning: Custom train/eval set assignment with CV splits!',
                  file=sys.stderr)
            splits = [_remap_split(s, train=train, eval=eval) for s in splits]

    data_parts = [data_train, data_dev]
    if data_test is not None:
        data_parts.append(data_test)
    data = pd.concat(data_parts, ignore_index=True)

    def id_split2idx_split(id_split):
        # Convert id-based splits to index (0...n-1) based splits for sklearn
        idx_split = TDSplit(*(
            list(data.index[data[ID_COL].isin(id_set)])
            for ids in id_split
            for id_set in (set(ids),)
            ))
        # Last sanity check:
        n = len(data)
        assert all(all(0 <= idx < n for idx in indices) for indices in idx_split)
        return idx_split

    return DataAndSplits(
        data=data,
        cv=[id_split2idx_split(s) for s in splits],
        original_split=id_split2idx_split(original_split)
        )
