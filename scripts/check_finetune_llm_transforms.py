import argparse
import sys

import numpy as np

from finetune_llm_prompts import (
    PROMPTS,
    build_prompt,
    prompt_is_for_difficulty,
)
from finetune_llm_spaces import (
    PointScale,
    bin_labels_to_scores,
    fit_bin_edges,
    fit_uniform_score_points,
    scores_from_active_space,
    scores_to_active_space,
    scores_to_bin_labels,
    scores_to_soft_bin_probs,
    scores_to_soft_label_probs,
)
from kvl import ID_COL, read_data_cv


PREDICTION_LANG_ORDER = ['cn', 'es', 'de']
PROB_LOSS_TYPES = {'raft', 'ce_prob', 'delta2'}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Inspect the finetune_llm target transformations and inverse '
            'mapping from scale labels back to scores.'
        )
    )
    parser.add_argument('--prompt', choices=PROMPTS, default=PROMPTS[0])
    parser.add_argument(
        '--splits',
        default='data/cv-split-ids-5.json',
        help='Splits from scripts/make_splits.py.',
    )
    cv_group = parser.add_mutually_exclusive_group()
    cv_group.add_argument(
        '--cv-mode',
        choices=['whole', 'first', 'remaining'],
        help='Run all folds, only fold 1, or folds 2..k.',
    )
    cv_group.add_argument(
        '--folds',
        type=int,
        nargs='+',
        help='Specific 1-based fold ids to inspect (e.g. --folds 1 3 5).',
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        choices=PREDICTION_LANG_ORDER,
        default=PREDICTION_LANG_ORDER,
        help='Languages to inspect. Default: cn es de.',
    )
    parser.add_argument(
        '--space',
        choices=['logit', 'probability'],
        default='logit',
        help='Target/prediction value space used for training.',
    )
    parser.add_argument(
        '--loss-type',
        choices=['ce', 'raft', 'ce_prob', 'delta2'],
        default='ce',
        help='Training loss type (used to choose target transform path).',
    )
    parser.add_argument(
        '--predict',
        choices=['auto', 'top', 'prob'],
        default='auto',
        help='Prediction decode mode (auto derives from --loss-type).',
    )
    parser.add_argument(
        '--scale-min',
        type=int,
        default=1,
        help='Minimum label on prompt/output scale.',
    )
    parser.add_argument(
        '--scale-max',
        type=int,
        default=5,
        help='Maximum label on prompt/output scale.',
    )
    parser.add_argument(
        '--show-edges',
        action='store_true',
        help='Print fitted train-bin edges for each fold/language.',
    )
    args = parser.parse_args()

    if args.cv_mode is None and args.folds is None:
        args.cv_mode = 'whole'
    if args.scale_max <= args.scale_min:
        parser.error('--scale-max must be > --scale-min')
    if args.folds is not None:
        if any(fold_i <= 0 for fold_i in args.folds):
            parser.error('--folds values must be positive (1-based)')
        if len(set(args.folds)) != len(args.folds):
            parser.error('--folds must not contain duplicates')
    return args


def default_predict_for_loss(loss_type):
    return 'prob' if loss_type in PROB_LOSS_TYPES else 'top'


def resolve_predict_mode(args):
    expected = default_predict_for_loss(args.loss_type)
    if args.predict == 'auto':
        return expected
    if args.predict != expected:
        print(
            'Warning: --predict does not match --loss-type '
            f'({args.predict=} vs recommended {expected} for '
            f'--loss-type={args.loss_type}). Proceeding anyway.',
            file=sys.stderr,
        )
    return args.predict


def build_lang_frame(df, lang, prompt, scale):
    needed = [
        ID_COL,
        'en_target_word',
        'en_target_pos',
        f'{lang}_L1_source_word',
        f'{lang}_L1_context',
        f'{lang}_GLMM_score',
    ]
    out = df[needed].copy()
    text_cols = [
        'en_target_word',
        'en_target_pos',
        f'{lang}_L1_source_word',
        f'{lang}_L1_context',
    ]
    for col in text_cols:
        out[col] = out[col].fillna('').astype(str).str.strip()
    out['prompt'] = out.apply(
        lambda row: build_prompt(lang, row, prompt, scale=scale),
        axis=1,
    )
    out['score'] = out[f'{lang}_GLMM_score'].astype(float)
    return out[[ID_COL, 'prompt', 'score']]


def reverse_scale_labels(labels, scale):
    labels = np.asarray(labels)
    return (scale.min + scale.max) - labels


def select_splits(data_cv, args):
    total_folds = len(data_cv.cv)
    splits = list(enumerate(data_cv.cv, 1))
    if args.folds:
        fold_map = dict(splits)
        invalid_folds = [
            fold_i for fold_i in args.folds
            if fold_i not in fold_map
        ]
        if invalid_folds:
            raise ValueError(
                f'Invalid fold id(s) for --folds: {invalid_folds}. '
                f'Valid range: 1..{total_folds}'
            )
        wanted = set(args.folds)
        splits = [
            (fold_i, split)
            for fold_i, split in splits
            if fold_i in wanted
        ]
    elif args.cv_mode == 'first':
        splits = splits[:1]
    elif args.cv_mode == 'remaining':
        splits = splits[1:]

    if not splits:
        raise ValueError(
            f'No splits selected for cv-mode={args.cv_mode}. '
            'Need at least 2 folds for remaining mode.'
        )
    return splits


def _fmt(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x)
        return np.array2string(arr, precision=6, floatmode='fixed')
    return f'{float(x):.6f}'


def describe_probe(
    raw_value,
    args,
    scale,
    edges,
    prob_class_values,
    predict_mode,
    output_is_difficulty,
    uses_prob_scale,
):
    active_value = scores_to_active_space(np.array([raw_value]), args.space)[0]
    internal_label = scores_to_bin_labels(
        np.array([active_value]),
        edges,
        scale=scale,
    )[0]
    soft_probs = (
        scores_to_soft_label_probs(
            np.array([active_value]),
            prob_class_values,
            scale=scale,
        )[0]
        if uses_prob_scale
        else scores_to_soft_bin_probs(
            np.array([active_value]),
            edges,
            scale=scale,
        )[0]
    )
    prompt_label = (
        int(internal_label)
        if output_is_difficulty
        else int(
            reverse_scale_labels(np.array([internal_label]), scale=scale)[0]
        )
    )

    recovered_internal = (
        prompt_label
        if output_is_difficulty
        else int(
            reverse_scale_labels(np.array([prompt_label]), scale=scale)[0]
        )
    )
    if predict_mode == 'prob':
        recovered_active = prob_class_values[recovered_internal - scale.min]
        decode_label = 'decoded active score from prob class point'
    else:
        recovered_active = bin_labels_to_scores(
            np.array([recovered_internal]),
            edges,
            scale=scale,
        )[0]
        decode_label = 'decoded bin-center active score'
    recovered_raw = scores_from_active_space(
        np.array([recovered_active]),
        args.space,
    )[0]

    print(f'    raw score:        {_fmt(raw_value)}')
    print(f'    active ({args.space}): {_fmt(active_value)}')
    print(
        f'    internal bin label (difficulty-oriented): {int(internal_label)}'
    )
    print(f'    prompt/LLM label used for training:       {prompt_label}')
    soft_label = 'soft probs on uniform score points'
    if not uses_prob_scale:
        soft_label = 'soft bin probs'
    print(f'    {soft_label} (internal label order {scale.min}..{scale.max}):')
    print(f'      {_fmt(soft_probs)}')
    print(f'    {decode_label}: {_fmt(recovered_active)}')
    print(f'    decoded raw score after inverse space: {_fmt(recovered_raw)}')


def inspect_fold_lang(args, data, train_idx, dev_idx, fold_num, lang):
    scale = PointScale(args.scale_min, args.scale_max)
    output_is_difficulty = prompt_is_for_difficulty(args.prompt)

    lang_data = build_lang_frame(data, lang, args.prompt, scale=scale)
    train_df = lang_data.iloc[train_idx].reset_index(drop=True)
    dev_df = lang_data.iloc[dev_idx].reset_index(drop=True)

    train_df['target_score'] = scores_to_active_space(
        train_df['score'].to_numpy(),
        args.space,
    )
    dev_df['target_score'] = scores_to_active_space(
        dev_df['score'].to_numpy(),
        args.space,
    )

    edges = fit_bin_edges(train_df['target_score'].to_numpy(), scale=scale)
    prob_class_values = fit_uniform_score_points(
        train_df['target_score'].to_numpy(),
        scale=scale,
    )
    uses_prob_scale = args.loss_type in PROB_LOSS_TYPES
    predict_mode = args.predict_mode
    train_df['bin_label'] = scores_to_bin_labels(
        train_df['target_score'].to_numpy(),
        edges,
        scale=scale,
    )
    train_df['soft_probs'] = list(
        (
            scores_to_soft_label_probs(
                train_df['target_score'].to_numpy(),
                prob_class_values,
                scale=scale,
            )
            if uses_prob_scale
            else scores_to_soft_bin_probs(
                train_df['target_score'].to_numpy(),
                edges,
                scale=scale,
            )
        )
    )
    dev_df['gold_bin_label'] = scores_to_bin_labels(
        dev_df['target_score'].to_numpy(),
        edges,
        scale=scale,
    )
    if not output_is_difficulty:
        train_df['bin_label'] = reverse_scale_labels(
            train_df['bin_label'].to_numpy(),
            scale=scale,
        )
        dev_df['gold_bin_label'] = reverse_scale_labels(
            dev_df['gold_bin_label'].to_numpy(),
            scale=scale,
        )
        prob_class_values = prob_class_values[::-1].copy()

    train_scores = train_df['score'].to_numpy()
    raw_min = float(np.min(train_scores))
    raw_max = float(np.max(train_scores))
    raw_mid = float((raw_min + raw_max) / 2.0)

    mid_label = (scale.min + scale.max) // 2

    print(f'Fold {fold_num} | lang={lang}')
    print(
        f'  sizes: train={len(train_df)} dev={len(dev_df)} | '
        'prompt output polarity='
        f'{"difficulty" if output_is_difficulty else "facility"}'
    )
    print(
        f'  train raw score range: min={_fmt(raw_min)} max={_fmt(raw_max)} '
        f'mean(min,max)={_fmt(raw_mid)}'
    )
    print(
        f'  train target_score ({args.space}) range: '
        f'min={_fmt(train_df["target_score"].min())} '
        f'max={_fmt(train_df["target_score"].max())}'
    )
    if args.show_edges:
        print(f'  fitted bin edges ({args.space}, train only): {_fmt(edges)}')
    if uses_prob_scale:
        print(
            '  uniform prob class values (active-space, label order '
            f'{scale.min}..{scale.max}): {_fmt(prob_class_values)}'
        )

    print('  Forward transform probes (raw target -> training representation)')
    for name, raw_value in (
        ('MIN', raw_min),
        ('MEAN(MIN,MAX)', raw_mid),
        ('MAX', raw_max),
    ):
        print(f'  [{name}]')
        describe_probe(
            raw_value,
            args,
            scale,
            edges,
            prob_class_values,
            predict_mode,
            output_is_difficulty,
            uses_prob_scale,
        )

    print('  Inverse transform probes (LLM output label -> decoded raw score)')
    if ((scale.min + scale.max) % 2) != 0:
        print(
            '    Note: no exact integer middle label on this scale; '
            f'using floor midpoint {mid_label}.'
        )
    for label_name, llm_label in (
        ('MIN_LABEL', scale.min),
        ('MID_LABEL', mid_label),
        ('MAX_LABEL', scale.max),
    ):
        internal_label = (
            llm_label
            if output_is_difficulty
            else int(
                reverse_scale_labels(np.array([llm_label]), scale=scale)[0]
            )
        )
        if predict_mode == 'prob':
            active_score = prob_class_values[internal_label - scale.min]
        else:
            active_score = bin_labels_to_scores(
                np.array([internal_label]),
                edges,
                scale=scale,
            )[0]
        raw_score = scores_from_active_space(
            np.array([active_score]),
            args.space,
        )[0]
        print(
            f'    {label_name}: prompt/LLM={llm_label} -> '
            f'internal={internal_label} -> active={_fmt(active_score)} -> '
            f'raw={_fmt(raw_score)}'
        )

    print()


def main():
    args = parse_args()
    scale = PointScale(args.scale_min, args.scale_max)
    args.predict_mode = resolve_predict_mode(args)
    polarity = (
        'difficulty'
        if prompt_is_for_difficulty(args.prompt)
        else 'facility'
    )
    data_cv = read_data_cv(args.splits)
    data = data_cv.data
    splits = select_splits(data_cv, args)
    selected_langs = [
        lang for lang in PREDICTION_LANG_ORDER
        if lang in args.languages
    ]

    print(f'Prompt: {args.prompt}')
    print(f'Prompt output polarity: {polarity}')
    print(f'Loss type: {args.loss_type}')
    print(f'Prediction mode: {args.predict_mode} (cli={args.predict})')
    print(f'Space: {args.space}')
    print(f'Scale: {scale.min}..{scale.max} ({scale.n_points} bins)')
    if args.folds:
        print(f'Folds: {" ".join(str(fold_i) for fold_i in args.folds)}')
    else:
        print(f'CV mode: {args.cv_mode} ({len(splits)} split(s))')
    print(f'Languages: {" ".join(selected_langs)}')
    print()

    for fold_i, split in splits:
        for lang in selected_langs:
            inspect_fold_lang(
                args,
                data=data,
                train_idx=split.train,
                dev_idx=split.dev,
                fold_num=fold_i,
                lang=lang,
            )


if __name__ == '__main__':
    main()
