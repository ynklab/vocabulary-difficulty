import argparse
import gc
import inspect
import re
import sys
import time
from contextlib import nullcontext, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, root_mean_squared_error
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
    )
from finetune_llm_spaces import (
    PointScale,
    bin_labels_to_scores,
    fit_bin_edges,
    fit_score_range,
    fit_uniform_score_points,
    scores_from_active_space,
    scores_to_active_space,
    scores_to_bin_labels,
    scores_to_soft_bin_probs,
    scores_to_soft_label_probs,
    )
from finetune_llm_prompts import PROMPTS, build_prompt, prompt_is_for_difficulty
from finetune_llm_adapter_meta import (
    adapter_metadata_path,
    build_adapter_metadata,
    load_adapter_metadata,
    parse_adapter_calibration,
    resolve_effective_scale_and_space,
    resolve_saved_interpretation_params,
    save_adapter_metadata,
    )
from finetune_llm_calibration import apply_calibration, fit_calibration
from finetune_llm_training import (
    CausalDataCollator,
    DeltaSquaredLossTrainer,
    FormattedLossTrainer,
    PromptOnlyCollator,
    RaftLossTrainer,
    SoftCELossTrainer,
    select_scoring_logits,
    )
from kvl import ID_COL, read_data_cv, spaced_clue
from tubelex_util import tubelex_lr_error


PREDICTION_LANG_ORDER = ['cn', 'es', 'de']
# TODO: switch to BF16 later?
USE_BF16 = True


OLD_EVAL_STRATEGY_ARG_NAME = 'evaluation_strategy'
NEW_EVAL_STRATEGY_ARG_NAME = 'eval_strategy'
PROB_LOSS_TYPES = {'raft', 'ce_prob', 'delta2'}
SAMPLE_WEIGHTING_STRATEGIES = {'no', 'inv', 'sqrt_inv'}
RUN_MODES = {'finetune-predict', 'predict', 'base-predict'}
MODEL_FAMILIES = {'causal', 'mlm'}
EPOCH_SNAPSHOT_TOL = 1e-2
REGRESSION_INPUT_STYLES = {
    'finetune_components',
    'finetune_components_with_l1_label',
    }
LANGUAGE_LABELS = {'cn': 'Chinese', 'de': 'German', 'es': 'Spanish'}
_TRAINER_INIT_PARAMS = inspect.signature(Trainer.__init__).parameters
_TRAINER_HAS_TOKENIZER_KWARG = 'tokenizer' in _TRAINER_INIT_PARAMS
_TRAINER_HAS_PROCESSING_CLASS_KWARG = (
    'processing_class' in _TRAINER_INIT_PARAMS
    )


def _epoch_dir_tag(epoch_value):
    return f'epoch_{epoch_value:07.3f}'.replace('.', 'p')


def _epoch_label(epoch_value):
    if abs(epoch_value - round(epoch_value)) < 1e-9:
        return str(int(round(epoch_value)))
    return f'{epoch_value:.3f}'.rstrip('0').rstrip('.')


def _completed_epoch_from_state(epoch_value):
    epoch_value = float(epoch_value)
    if epoch_value <= 0:
        return None
    completed_epoch = int(np.floor(epoch_value + EPOCH_SNAPSHOT_TOL))
    return completed_epoch if completed_epoch > 0 else None


def _snap_epoch_if_near_integer(epoch_value):
    epoch_value = float(epoch_value)
    nearest_epoch = round(epoch_value)
    if nearest_epoch <= 0:
        return None
    if abs(epoch_value - nearest_epoch) > EPOCH_SNAPSHOT_TOL:
        return None
    return int(nearest_epoch)


def _is_final_epoch_snapshot(epoch_value, target_epochs):
    target_epoch = _snap_epoch_if_near_integer(target_epochs)
    snap_epoch = _snap_epoch_if_near_integer(epoch_value)
    if target_epoch is None or snap_epoch is None:
        return False
    return snap_epoch == target_epoch


def tokenizer_uses_chat_template(tokenizer):
    return bool(getattr(tokenizer, 'chat_template', None)) and hasattr(
        tokenizer,
        'apply_chat_template',
        )


def format_prompt_for_tokenizer(tokenizer, prompt, chat_template_kwargs=None):
    chat_template_kwargs = chat_template_kwargs or {}
    if not tokenizer_uses_chat_template(tokenizer):
        return prompt
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs
        )


def format_prompts_for_tokenizer(tokenizer, prompts, chat_template_kwargs=None):
    if not tokenizer_uses_chat_template(tokenizer):
        return prompts
    return [
        format_prompt_for_tokenizer(tokenizer, p, chat_template_kwargs) for p in prompts
        ]


def prompt_add_special_tokens(tokenizer):
    return not tokenizer_uses_chat_template(tokenizer)


def prompt_add_special_tokens_for_mode(
    tokenizer, model_family, chat_template_kwargs=None
    ):
    if model_family == 'mlm':
        return True
    return prompt_add_special_tokens(tokenizer)


def get_single_mask_position(input_ids, mask_token_id):
    mask_positions = np.where(np.asarray(input_ids, dtype=int) == int(mask_token_id))[0]
    if len(mask_positions) != 1:
        raise ValueError(
            f'Expected exactly one mask token in encoded input, got '
            f'{len(mask_positions)}'
            )
    return int(mask_positions[0])


def encode_prompt_for_scoring(
    tokenizer,
    prompt,
    max_length,
    model_family,
    mlm_label_prefix_ids=None,
    chat_template_kwargs=None,
    ):
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f'Unsupported model_family: {model_family}')

    if model_family == 'causal':
        formatted_prompt = format_prompt_for_tokenizer(
            tokenizer,
            prompt,
            chat_template_kwargs=chat_template_kwargs,
            )
        prompt_ids = tokenizer(
            formatted_prompt,
            add_special_tokens=prompt_add_special_tokens_for_mode(
                tokenizer,
                model_family=model_family,
                chat_template_kwargs=chat_template_kwargs,
                ),
            truncation=True,
            max_length=max_length - 1,
            )['input_ids']
        return {
            'input_ids': prompt_ids,
            'attention_mask': [1] * len(prompt_ids),
            'score_position': int(len(prompt_ids) - 1),
            }

    if tokenizer.mask_token_id is None:
        raise ValueError(
            'Tokenizer has no mask token; --mlm requires a model/tokenizer '
            'with mask_token_id.'
            )
    if tokenizer.mask_token is None:
        raise ValueError(
            'Tokenizer has no mask token string; --mlm requires mask_token.'
            )

    mlm_label_prefix_ids = [int(x) for x in (mlm_label_prefix_ids or [])]
    raw_prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    n_special = tokenizer.num_special_tokens_to_add(pair=False)
    max_prompt_len = max_length - n_special - len(mlm_label_prefix_ids) - 1
    if max_prompt_len < 1:
        raise ValueError(
            f'--max-length={max_length} is too small for MLM mode '
            '(needs room for specials + one mask token)'
            )
    truncated_prompt_ids = raw_prompt_ids[:max_prompt_len]
    merged_ids = (
        truncated_prompt_ids +
        mlm_label_prefix_ids +
        [int(tokenizer.mask_token_id)]
        )
    prefix_ids = []
    suffix_ids = []
    if tokenizer.cls_token_id is not None:
        prefix_ids.append(int(tokenizer.cls_token_id))
    if tokenizer.sep_token_id is not None:
        suffix_ids.append(int(tokenizer.sep_token_id))
    elif tokenizer.eos_token_id is not None:
        suffix_ids.append(int(tokenizer.eos_token_id))
    input_ids = prefix_ids + merged_ids + suffix_ids
    score_position = get_single_mask_position(input_ids, tokenizer.mask_token_id)
    return {
        'input_ids': input_ids,
        'attention_mask': [1] * len(input_ids),
        'score_position': score_position,
        }


def make_training_args(**kwargs):
    '''
    Older version of transformers (4.38.1) uses a different argument name.
    Fall back if necessary.
    '''
    try:
        ta = TrainingArguments(**kwargs)
    except TypeError as e:
        if NEW_EVAL_STRATEGY_ARG_NAME not in kwargs:
            raise e
        # Fallback to the old parameter name:
        eval_strategy = kwargs.pop(NEW_EVAL_STRATEGY_ARG_NAME)
        kwargs[OLD_EVAL_STRATEGY_ARG_NAME] = eval_strategy
        ta = TrainingArguments(**kwargs)
    return ta


def make_trainer_compat(*, tokenizer=None, **kwargs):
    trainer_kwargs = dict(kwargs)
    if tokenizer is not None:
        if _TRAINER_HAS_TOKENIZER_KWARG:
            trainer_kwargs['tokenizer'] = tokenizer
        elif _TRAINER_HAS_PROCESSING_CLASS_KWARG:
            trainer_kwargs['processing_class'] = tokenizer
    return Trainer(**trainer_kwargs)


def _format_mib(n_bytes):
    return f'{(n_bytes / (1024 ** 2)):.0f} MiB'


def print_vram_stats(label, reset_peak=False):
    if not torch.cuda.is_available():
        return

    print(f'  VRAM [{label}]')
    for device_idx in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(device_idx)
        allocated_b = torch.cuda.memory_allocated(device_idx)
        reserved_b = torch.cuda.memory_reserved(device_idx)
        peak_alloc_b = torch.cuda.max_memory_allocated(device_idx)
        peak_reserved_b = torch.cuda.max_memory_reserved(device_idx)
        name = torch.cuda.get_device_name(device_idx)
        print(
            f'    cuda:{device_idx} ({name}) '
            f'free={_format_mib(free_b)} / total={_format_mib(total_b)} '
            f'alloc={_format_mib(allocated_b)} '
            f'reserved={_format_mib(reserved_b)} '
            f'peak_alloc={_format_mib(peak_alloc_b)} '
            f'peak_reserved={_format_mib(peak_reserved_b)}'
            )
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(device_idx)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        choices=sorted(RUN_MODES),
        default='finetune-predict',
        help=(
            'Run mode: finetune adapter and predict, predict from an '
            'existing adapter, or base-model prediction only.'
            ),
        )
    parser.add_argument(
        '--config-name',
        default='default',
        help=(
            'Configuration label used in output paths '
            'and prediction filenames.'
            ),
        )
    parser.add_argument(
        '--model-name',
        default='zai-org/glm-4-9b',
        help='HF model id. Default: zai-org/glm-4-9b.',
        )
    parser.add_argument(
        '--head-type',
        choices=['token', 'regression'],
        default='token',
        help=(
            'Prediction head type: token-based label scoring (default) '
            'or sequence-classification regression head.'
            ),
        )
    parser.add_argument(
        '--regression-input-style',
        choices=sorted(REGRESSION_INPUT_STYLES),
        default='finetune_components',
        help=(
            'Input composition style for --head-type regression. '
            '"finetune_components" matches finetune.py component merge; '
            '"finetune_components_with_l1_label" prepends language label.'
            ),
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
        help='Specific 1-based fold ids to run (e.g. --folds 1 3 5).',
        )
    cv_group.add_argument(
        '--final-data',
        action='store_true',
        help=(
            'Run one final fit trained on full train+dev data and '
            'predict on test.'
            ),
        )
    parser.add_argument(
        '--languages', '--l1s',
        nargs='+',
        choices=PREDICTION_LANG_ORDER,
        default=PREDICTION_LANG_ORDER,
        help='Languages to train/predict for. Default: cn es de.',
        )
    parser.add_argument(
        '--all-in-one',
        action='store_true',
        help=(
            'Train/load one adapter per fold for all selected languages '
            'instead of one adapter per language.'
            ),
        )
    parser.add_argument(
        '--epochs',
        type=float,
        default=1.0,
        help='Training epochs per fold/adapter. Suggested: 1-2.',
        )
    parser.add_argument(
        '--curriculum', '-c',
        choices=['none', 'hardness2'],
        default='none',
        help='Training data curriculum. Default: none.',
        )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Per-device batch size. Default: 1.',
        )
    parser.add_argument(
        '--grad-accum',
        type=int,
        default=8,
        help='Gradient accumulation steps. Default: 8.',
        )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-4,
        help='Learning rate. Default: 2e-4.',
        )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0,
        help='Weight decay. Default: 0.0.',
        )
    parser.add_argument(
        '--warmup-ratio',
        type=float,
        default=0.0,
        help='Warmup ratio in [0, 1]. Default: 0.0.',
        )
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=0,
        help=(
            'Early stopping patience in evaluation calls. '
            '0 disables early stopping.'
            ),
        )
    parser.add_argument(
        '--early-stop-threshold',
        type=float,
        default=0.001,
        help='Minimum eval_loss improvement to reset early-stop patience.',
        )
    parser.add_argument(
        '--eval-steps',
        type=int,
        default=50,
        help='Evaluation/save interval in steps when early stopping is on.',
        )
    parser.add_argument(
        '--lr-scheduler',
        choices=[
            'linear',
            'cosine',
            'cosine_with_restarts',
            'polynomial',
            'constant',
            'constant_with_warmup',
            'inverse_sqrt',
            'reduce_lr_on_plateau',
            ],
        default='constant',
        help='Learning-rate scheduler type. Default: constant.',
        )
    parser.add_argument(
        '--loss-decimals',
        type=int,
        default=4,
        help='Digits after decimal point for logged loss values.',
        )
    parser.add_argument(
        '--space',
        choices=['logit', 'probability'],
        default='logit',
        help=(
            'Target/prediction value space: original logits or expit-'
            'transformed probabilities. Losses are reported in this space.'
            ),
        )
    parser.add_argument(
        '--leeway',
        type=float,
        default=0.0,
        help=(
            'Symmetric fractional expansion of the train-time logit-space '
            'range used for probability-weighted predictions and soft/prob '
            'loss targets, applied before any --space transform. Example: '
            '0.1 adds 5%% below and 5%% above.'
            ),
        )
    parser.add_argument(
        '--reweight',
        '--rw',
        dest='reweight',
        choices=['no', 'inv', 'sqrt_inv'],
        default='no',
        help=(
            'Sample reweighting for soft/prob losses based on the nearest '
            'scale point (argmax target_prob): no, inverse frequency, or '
            'inverse square-root frequency.'
            ),
        )
    parser.add_argument(
        '--loss-type',
        choices=['ce', 'raft', 'ce_prob', 'delta2'],
        default='ce_prob',
        help=(
            'Training loss type: hard CE next-token, RAFT expected-value '
            'MSE, or soft CE over class-token probabilities.'
            ),
        )
    parser.add_argument(
        '--predict',
        '-p',
        choices=['auto', 'top', 'prob'],
        default='auto',
        help=(
            'Prediction mode: argmax class token, probability-weighted, or '
            'auto (derived from --loss-type).'
            ),
        )
    parser.add_argument(
        '--no-finetune',
        action='store_true',
        help=argparse.SUPPRESS,
        )
    parser.add_argument(
        '--mlm',
        action='store_true',
        help=(
            'Use masked-language-model mode (single masked token prediction) '
            'instead of causal next-token mode.'
            ),
        )
    parser.add_argument(
        '--max-length',
        type=int,
        default=256,
        help='Tokenization max length. Default: 256.',
        )
    parser.add_argument(
        '--token-form',
        choices=['auto', 'space', 'bare'],
        default='auto',
        help=(
            'Scale-label token form: auto (choose one global form that works '
            'for all labels: space preferred, else bare), space (require '
            'leading-space token for all labels), or bare (require bare '
            'token for all labels).'
            ),
        )  # TODO: "space" smh doesn't work – no big deal actually(?)
    parser.add_argument(
        '--lora-r',
        type=int,
        default=8,
        help='LoRA rank. Default: 8.',
        )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=16,
        help='LoRA alpha. Default: 16.',
        )
    parser.add_argument(
        '--lora-dropout',
        type=float,
        default=0.05,
        help='LoRA dropout. Default: 0.05.',
        )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed. Default: 42.',
        )
    parser.add_argument(
        '--scale-min',
        type=int,
        default=1,
        help=(
            'Minimum difficulty label on the prompt/output scale. '
            'Default: 1.'
            ),
        )
    parser.add_argument(
        '--scale-max',
        type=int,
        default=5,
        help=(
            'Maximum difficulty label on the prompt/output scale. '
            'Default: 5.'
            ),
        )
    parser.add_argument(
        '--output-dir',
        default='models',
        help='Base output dir for adapters/checkpoints.',
        )
    parser.add_argument(
        '--predictions-dir',
        default=str(Path('predictions') / 'finetuned_llm'),
        help='Base output dir for merged fold prediction CSVs.',
        )
    parser.add_argument(
        '--prediction-suffix',
        default='',
        help=(
            'Optional suffix appended to prediction CSV basename '
            '(e.g. _fold1).'
            ),
        )
    parser.add_argument(
        '--overwrite-predictions',
        action='store_true',
        help='Allow overwriting existing prediction CSV files.',
        )
    parser.add_argument(
        '--predict-train',
        action='store_true',
        help='Also write and report predictions on the fold training split.',
        )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help=(
            'Fit a post-hoc linear calibration on training predictions in '
            'active space and apply it to reported predictions/metrics. In '
            '--mode predict, saved adapter calibration is auto-applied when '
            'available (even without this flag).'
            ),
        )
    parser.add_argument(
        '--predict-checkpoint',
        help=(
            'In --mode=predict, load a checkpoint adapter instead of the '
            'final adapter. Accepts an exact checkpoint tag '
            '(e.g. epoch_001p000) or a 1-based integer index into lexical '
            'order of epoch_checkpoints/.'
            ),
        )
    parser.add_argument(
        '--disable-adapter-calibration',
        action='store_true',
        help=(
            'In --mode predict, ignore calibration saved in adapter metadata '
            'and keep raw active-space predictions.'
            ),
        )
    parser.add_argument(
        '--results-path',
        help=(
            'CSV path for fold/language metrics. '
            'Default: results/finetuned_llm/{config}--{model}.csv'
            ),
        )
    parser.add_argument(
        '--stdout-file',
        help='Write this script\'s stdout to a file instead of console.',
        )
    parser.add_argument(
        '--sanity-check',
        action='store_true',
        help='Run one-step loss/gradient sanity check before training.',
        )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Enable trust_remote_code for model/tokenizer loading.',
        )
    parser.add_argument(
        '--device-map',
        choices=['auto', 'cuda'],
        default='auto',
        help=(
            'Model placement strategy for from_pretrained: auto (Accelerate '
            'placement) or cuda (force all modules onto cuda:0).'
            ),
        )
    parser.add_argument(
        '--no-gradient-checkpointing',
        action='store_true',
        help='Disable gradient checkpointing.',
        )
    parser.add_argument(
        '-v',
        '--vram-stats',
        action='store_true',
        help='Print CUDA VRAM stats at key checkpoints.',
        )
    args = parser.parse_args()
    if args.no_finetune:
        args.mode = 'base-predict'
    if args.cv_mode is None and args.folds is None and not args.final_data:
        args.cv_mode = 'whole'
    if args.scale_max <= args.scale_min:
        parser.error('--scale-max must be > --scale-min')
    if (
            args.mode == 'finetune-predict' and
            args.curriculum == 'hardness2' and
            abs(args.epochs - round(args.epochs)) > 1e-9
            ):
        parser.error('--curriculum hardness2 requires integer --epochs')
    if args.leeway < 0:
        parser.error('--leeway must be >= 0')
    if args.folds is not None:
        if any(fold_i <= 0 for fold_i in args.folds):
            parser.error('--folds values must be positive (1-based)')
        if len(set(args.folds)) != len(args.folds):
            parser.error('--folds must not contain duplicates')
    if args.predict_checkpoint and args.mode != 'predict':
        parser.error('--predict-checkpoint requires --mode predict')
    if args.disable_adapter_calibration and args.mode != 'predict':
        parser.error('--disable-adapter-calibration requires --mode predict')
    if args.disable_adapter_calibration and args.calibrate:
        parser.error(
            '--disable-adapter-calibration cannot be combined with --calibrate'
            )
    if not (0.0 <= args.warmup_ratio <= 1.0):
        parser.error('--warmup-ratio must be in [0, 1]')
    if args.weight_decay < 0:
        parser.error('--weight-decay must be >= 0')
    if ('/' in args.prediction_suffix) or ('\\' in args.prediction_suffix):
        parser.error('--prediction-suffix must not contain path separators')
    args.model_family = 'mlm' if args.mlm else 'causal'
    if args.model_family == 'mlm':
        if args.token_form == 'space':
            parser.error('--token-form space is not supported with --mlm')
        if any(
                arg == opt or arg.startswith(f'{opt}=')
                for arg in sys.argv
                for opt in ('--lora-r', '--lora-alpha', '--lora-dropout')
                ):
            parser.error(
                '--lora-* options are not supported with --mlm '
                '(MLM mode uses full fine-tuning).'
                )
    if args.head_type == 'regression':
        if args.mlm:
            parser.error('--head-type regression does not support --mlm')
        if args.predict_checkpoint:
            parser.error(
                '--head-type regression does not support --predict-checkpoint'
                )
        if args.disable_adapter_calibration:
            parser.error(
                '--head-type regression does not support '
                '--disable-adapter-calibration'
                )
        if any(
                arg == opt or arg.startswith(f'{opt}=')
                for arg in sys.argv
                for opt in ('--loss-type', '--predict', '--token-form')
                ):
            parser.error(
                '--head-type regression is incompatible with '
                '--loss-type/--predict/--token-form'
                )
        if args.calibrate:
            parser.error(
                '--head-type regression does not support --calibrate'
                )
        if any(
                arg == opt or arg.startswith(f'{opt}=')
                for arg in sys.argv
                for opt in ('--space', '--leeway', '--reweight', '--rw')
                ):
            parser.error(
                '--head-type regression is incompatible with '
                '--space/--leeway/--reweight'
                )
    return args


def default_predict_for_loss(loss_type):
    return 'prob' if loss_type in PROB_LOSS_TYPES else 'top'


def resolve_predict_mode(args):
    if args.head_type == 'regression':
        return 'regression'
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


def _safe_corr(fn, y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    try:
        return float(fn(y_true, y_pred).statistic)
    except ValueError:
        return np.nan


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        'rmse': float(root_mean_squared_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'pearson': _safe_corr(pearsonr, y_true, y_pred),
        'spearman': _safe_corr(spearmanr, y_true, y_pred),
        }


def compute_sample_weights_from_soft_probs(soft_probs, strategy):
    probs = np.asarray(soft_probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f'soft_probs must be 2D, got shape {probs.shape}')
    if strategy not in SAMPLE_WEIGHTING_STRATEGIES:
        raise ValueError(f'Unsupported sample weighting strategy: {strategy}')
    if strategy == 'no':
        return np.ones(len(probs), dtype=float)

    nearest_idx = np.argmax(probs, axis=1)
    counts = np.bincount(nearest_idx, minlength=probs.shape[1]).astype(float)
    sample_counts = counts[nearest_idx]
    if strategy == 'inv':
        weights = 1.0 / sample_counts
    elif strategy == 'sqrt_inv':
        weights = 1.0 / np.sqrt(sample_counts)
    else:
        raise ValueError(f'Unsupported sample weighting strategy: {strategy}')

    mean_w = float(np.mean(weights))
    if mean_w <= 0:
        raise ValueError('Invalid sample weights: non-positive mean')
    return weights / mean_w


def safe_model_name(name):
    return re.sub(r'[^A-Za-z0-9._-]+', '--', name).strip('-')


def run_name(args):
    return (
        f'{safe_model_name(args.config_name)}'
        f'--{safe_model_name(args.model_name)}'
        )


def prediction_run_name(args):
    return f'{run_name(args)}{args.prediction_suffix}'


def path_with_suffix_inserted(path, suffix):
    path = Path(path)
    return path.with_name(f'{path.stem}{suffix}{path.suffix}')


def fold_selection_suffix(folds):
    if not folds:
        return None
    if len(folds) == 1:
        return f'_fold{folds[0]}'
    joined = '-'.join(str(fold_i) for fold_i in folds)
    return f'_folds{joined}'


def maybe_suffix_output_path(path_value, folds):
    if path_value is None or not folds:
        return path_value
    return str(path_with_suffix_inserted(path_value, fold_selection_suffix(folds)))


def fold_artifacts_dir(args, fold_num):
    return (
        Path(args.output_dir) /
        run_name(args) /
        f'fold_{fold_num:02d}'
        )


def list_epoch_checkpoint_tags(epoch_checkpoint_root):
    if not epoch_checkpoint_root.exists():
        return []
    return sorted(
        p.name for p in epoch_checkpoint_root.iterdir()
        if p.is_dir()
        )


def adapter_scope_tag(args, lang):
    return 'all' if args.all_in_one else lang


def fold_adapter_dir(args, fold_num, lang):
    return fold_artifacts_dir(args, fold_num) / adapter_scope_tag(args, lang)


def resolve_predict_adapter_path(args, fold_num, lang):
    fold_out = fold_adapter_dir(args, fold_num, lang)
    if args.predict_checkpoint is None:
        adapter_path = fold_out / 'adapter'
        if not adapter_path.exists():
            raise FileNotFoundError(
                f'Adapter not found for --mode predict: {adapter_path}'
                )
        return adapter_path

    epoch_checkpoint_root = fold_out / 'epoch_checkpoints'
    tags = list_epoch_checkpoint_tags(epoch_checkpoint_root)
    checkpoint_value = str(args.predict_checkpoint)

    if checkpoint_value.isdigit():
        if not tags:
            raise FileNotFoundError(
                'No epoch checkpoints found at '
                f'{epoch_checkpoint_root} for --predict-checkpoint='
                f'{checkpoint_value}'
                )
        idx = int(checkpoint_value)
        if idx <= 0 or idx > len(tags):
            raise ValueError(
                '--predict-checkpoint index out of range '
                f'({idx}); available 1..{len(tags)}: {tags}'
                )
        checkpoint_tag = tags[idx - 1]
    else:
        checkpoint_tag = checkpoint_value

    adapter_path = epoch_checkpoint_root / checkpoint_tag / 'adapter'
    if not adapter_path.exists():
        extra = f'; available checkpoint tags: {tags}' if tags else ''
        raise FileNotFoundError(
            'Checkpoint adapter not found for --mode predict: '
            f'{adapter_path}{extra}'
            )
    return adapter_path


def fold_prediction_output_path(args, fold_i, total_folds, split_name):
    suffix = '' if split_name == 'dev' else f'-{split_name}'
    return (
        Path(args.predictions_dir) /
        f'fold-{fold_i}-of-{total_folds}{suffix}' /
        f'{prediction_run_name(args)}.csv'
        )


def final_data_prediction_output_path(args):
    return (
        Path(args.predictions_dir) /
        'test' /
        f'{prediction_run_name(args)}.csv'
        )


def assert_prediction_output_writable(path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(
            'Prediction file already exists '
            f'({path}). Use --overwrite-predictions to replace it.'
            )


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
    for c in text_cols:
        out[c] = out[c].fillna('').astype(str).str.strip()
    out['prompt'] = out.apply(
        lambda row: build_prompt(
            lang,
            row,
            prompt,
            scale=scale,
            ),
        axis=1,
        )
    out['score'] = out[f'{lang}_GLMM_score'].astype(float)
    return out[[ID_COL, 'prompt', 'score']]


def _regression_separator(tokenizer):
    return f' {tokenizer.sep_token} ' if tokenizer.sep_token else ' '


def _build_regression_input_text(row, lang, style, sep_token):
    parts = [
        row[f'{lang}_L1_source_word'],
        row[f'{lang}_L1_context'],
        row['en_target_clue'],
        row['en_target_word'],
        ]
    parts = [str(p).strip() for p in parts]
    if style == 'finetune_components_with_l1_label':
        parts = [LANGUAGE_LABELS[lang]] + parts
    return sep_token.join(parts)


def build_lang_frame_regression(df, lang, style, sep_token):
    needed = [
        ID_COL,
        'en_target_word',
        f'{lang}_L1_source_word',
        f'{lang}_L1_context',
        f'{lang}_GLMM_score',
        ]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(
            'Regression input style requires missing columns: '
            f'{missing}'
            )
    out = df[needed].copy()
    text_cols = [
        'en_target_word',
        f'{lang}_L1_source_word',
        f'{lang}_L1_context',
        ]
    for c in text_cols:
        out[c] = out[c].fillna('').astype(str).str.strip()
    # Match LLM prompt construction: derive clue text from target word.
    out['en_target_clue'] = out['en_target_word'].map(spaced_clue)
    out['input_text'] = out.apply(
        lambda row: _build_regression_input_text(
            row,
            lang=lang,
            style=style,
            sep_token=sep_token,
            ),
        axis=1,
        )
    out['score'] = out[f'{lang}_GLMM_score'].astype(float)
    return out[[ID_COL, 'input_text', 'score']]


def get_truncated_item_ids(
    df,
    tokenizer,
    max_length,
    model_family,
    mlm_label_prefix_ids=None,
    chat_template_kwargs=None
    ):
    mlm_label_prefix_ids = [int(x) for x in (mlm_label_prefix_ids or [])]
    max_prompt_len = (
        max_length - 1
        if model_family == 'causal'
        else (
            max_length -
            tokenizer.num_special_tokens_to_add(pair=False) -
            len(mlm_label_prefix_ids) -
            1
            )
        )
    truncated_item_ids = []

    for row in df[[ID_COL, 'prompt']].itertuples(index=False):
        if model_family == 'causal':
            formatted_prompt = format_prompt_for_tokenizer(
                tokenizer,
                row.prompt,
                chat_template_kwargs=chat_template_kwargs,
                )
            prompt_ids = tokenizer(
                formatted_prompt,
                add_special_tokens=prompt_add_special_tokens_for_mode(
                    tokenizer,
                    model_family=model_family,
                    chat_template_kwargs=chat_template_kwargs,
                    ),
                truncation=False,
                )['input_ids']
        else:
            prompt_ids = tokenizer(
                row.prompt,
                add_special_tokens=False,
                truncation=False,
                )['input_ids']
        if len(prompt_ids) > max_prompt_len:
            truncated_item_ids.append(row.item_id)

    return truncated_item_ids


def get_truncated_item_ids_regression(
    df, tokenizer, max_length
    ):
    truncated_item_ids = []
    for row in df[[ID_COL, 'input_text']].itertuples(index=False):
        prompt_ids = tokenizer(
            row.input_text,
            add_special_tokens=True,
            truncation=False,
            )['input_ids']
        if len(prompt_ids) > max_length:
            truncated_item_ids.append(row.item_id)
    return truncated_item_ids


def encode_train_dataset(
    df,
    tokenizer,
    max_length,
    label_value_to_token_id,
    model_family,
    mlm_label_prefix_ids=None,
    chat_template_kwargs=None,
    ):
    rows = []
    for row in df.itertuples(index=False):
        label_id = label_value_to_token_id[int(row.bin_label)]
        encoded_prompt = encode_prompt_for_scoring(
            tokenizer,
            row.prompt,
            max_length=max_length,
            model_family=model_family,
            mlm_label_prefix_ids=mlm_label_prefix_ids,
            chat_template_kwargs=chat_template_kwargs,
            )

        if model_family == 'causal':
            input_ids = encoded_prompt['input_ids'] + [label_id]
            labels = ([-100] * (len(input_ids) - 1)) + [label_id]
            attention_mask = [1] * len(input_ids)
            score_position = len(input_ids) - 1
        else:
            input_ids = encoded_prompt['input_ids']
            attention_mask = encoded_prompt['attention_mask']
            score_position = encoded_prompt['score_position']
            labels = [-100] * len(input_ids)
            labels[score_position] = label_id

        rows.append(
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'score_position': int(score_position),
                }
            )

    return Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)


def encode_regression_dataset(df, tokenizer, max_length):
    enc = tokenizer(
        df['input_text'].tolist(),
        truncation=True,
        max_length=max_length,
        )
    return Dataset.from_dict(
        {
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels': df['score'].astype(float).tolist(),
            }
        )


def encode_prompt_dataset(
    df,
    tokenizer,
    max_length,
    model_family,
    mlm_label_prefix_ids=None,
    chat_template_kwargs=None
    ):
    rows = []
    for row in df.itertuples(index=False):
        encoded_prompt = encode_prompt_for_scoring(
            tokenizer,
            row.prompt,
            max_length=max_length,
            model_family=model_family,
            mlm_label_prefix_ids=mlm_label_prefix_ids,
            chat_template_kwargs=chat_template_kwargs,
            )
        prompt_ids = encoded_prompt['input_ids']
        attention_mask = encoded_prompt['attention_mask']
        rows.append(
            {
                'input_ids': prompt_ids,
                'attention_mask': attention_mask,
                'target_probs': row.soft_probs,
                'sample_weight': float(
                    getattr(row, 'sample_weight', 1.0)
                    ),
                'score_position': int(encoded_prompt['score_position']),
                }
            )

    return Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)


def get_first_model_device(model):
    return next(model.parameters()).device


def reverse_scale_labels(labels, scale):
    labels = np.asarray(labels)
    return (scale.min + scale.max) - labels


def reverse_scale_probs(probs):
    probs = np.asarray(probs, dtype=float)
    return probs[:, ::-1].copy()


def get_scale_label_token_ids(
    tokenizer, scale, token_form='auto', model_family='causal'
    ):
    if token_form not in {'auto', 'space', 'bare'}:
        raise ValueError(f'Invalid token_form: {token_form}')
    label_values = list(range(scale.min, scale.max + 1))
    label_prefix_ids = []
    if model_family == 'mlm' and token_form == 'space':
        raise ValueError('--token-form space is not supported with --mlm')

    if model_family == 'mlm':
        mlm_bare_ids = {}
        all_bare_single = True
        for label_value in label_values:
            bare_ids = tokenizer(
                str(label_value),
                add_special_tokens=False,
                )['input_ids']
            if len(bare_ids) != 1:
                all_bare_single = False
                break
            mlm_bare_ids[label_value] = int(bare_ids[0])
        if all_bare_single and (
                len(set(mlm_bare_ids.values())) == len(label_values)
                ):
            return mlm_bare_ids, label_prefix_ids

        shared_prefix_ids = None
        mlm_label_ids = {}
        for label_value in label_values:
            token_ids = tokenizer(
                f' {label_value}',
                add_special_tokens=False,
                )['input_ids']
            if len(token_ids) < 2:
                raise ValueError(
                    'MLM label tokenization must provide at least one shared '
                    'prefix token plus one class token per label. '
                    f'For label {label_value}, got token ids: {token_ids}'
                    )
            prefix_ids = [int(x) for x in token_ids[:-1]]
            class_token_id = int(token_ids[-1])
            if shared_prefix_ids is None:
                shared_prefix_ids = prefix_ids
            elif prefix_ids != shared_prefix_ids:
                raise ValueError(
                    'MLM label tokenization requires one shared prefix token '
                    'sequence across all labels, but prefixes differ. '
                    f'First prefix={shared_prefix_ids}, label {label_value} '
                    f'prefix={prefix_ids}'
                    )
            mlm_label_ids[label_value] = class_token_id
        if len(set(mlm_label_ids.values())) != len(label_values):
            raise ValueError(
                'Tokenizer maps multiple MLM scale labels to the same class '
                f'token id for scale {scale.min}..{scale.max}'
                )
        return mlm_label_ids, shared_prefix_ids or []

    if token_form == 'auto':
        all_space = True
        all_bare = True
        for label_value in label_values:
            space_ids = tokenizer(
                f' {label_value}',
                add_special_tokens=False,
                )['input_ids']
            bare_ids = tokenizer(
                str(label_value),
                add_special_tokens=False,
                )['input_ids']
            all_space = all_space and (len(space_ids) == 1)
            all_bare = all_bare and (len(bare_ids) == 1)
            if not all_space and not all_bare:
                break
        if all_space:
            token_form = 'space'
        elif all_bare:
            token_form = 'bare'
        else:
            raise ValueError(
                'token_form=auto could not find one global single-token '
                'form for all scale labels (mixed space/bare '
                'availability). Use --token-form space or --token-form '
                'bare explicitly.'
                )

    label_value_to_token_id = {}
    for label_value in label_values:
        chosen_token_id = None
        if token_form == 'space':
            variants = (f' {label_value}',)
        elif token_form == 'bare':
            variants = (str(label_value),)
        else:
            # Prefer the leading-space variant because labels are generated
            # after a prompt, where the tokenizer often encodes the preceding
            # space into the number token itself.
            variants = (f' {label_value}', str(label_value))
        for variant in variants:
            token_ids = tokenizer(
                variant,
                add_special_tokens=False,
                )['input_ids']
            if len(token_ids) == 1:
                chosen_token_id = token_ids[0]
                break
        if chosen_token_id is None:
            if token_form == 'space':
                tried = [f' {label_value}']
            elif token_form == 'bare':
                tried = [str(label_value)]
            else:
                tried = [f' {label_value}', str(label_value)]
            raise ValueError(
                'Scale label must be a single tokenizer token '
                f'(token_form={token_form}; tried {tried})'
                )
        label_value_to_token_id[label_value] = chosen_token_id

    token_ids = list(label_value_to_token_id.values())
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(
            'Tokenizer maps multiple scale labels to the same token id '
            f'for scale {scale.min}..{scale.max}'
            )

    return label_value_to_token_id, label_prefix_ids


def predict_scale_labels_and_losses(
    model,
    tokenizer,
    prompts,
    gold_labels,
    gold_soft_probs,
    max_length,
    batch_size,
    label_value_to_token_id,
    scale,
    model_family,
    mlm_label_prefix_ids=None,
    prob_class_values=None,
    chat_template_kwargs=None
    ):
    model.eval()
    out_top = []
    out_prob = []
    lm_losses = []
    raft_losses = []
    ce_prob_losses = []
    device = get_first_model_device(model)

    label_values = list(range(scale.min, scale.max + 1))
    if prob_class_values is None:
        prob_class_values = label_values
    class_ids = torch.tensor(
        [label_value_to_token_id[v] for v in label_values],
        dtype=torch.long,
        device=device,
        )
    prob_class_values_t = torch.tensor(
        prob_class_values,
        dtype=torch.float32,
        device=device,
        )

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_gold = torch.tensor(
            gold_labels[i:i + batch_size],
            dtype=torch.long,
            device=device,
            )
        batch_soft = torch.tensor(
            gold_soft_probs[i:i + batch_size],
            dtype=torch.float32,
            device=device,
            )
        batch_encoded = [
            encode_prompt_for_scoring(
                tokenizer,
                prompt,
                max_length=max_length,
                model_family=model_family,
                mlm_label_prefix_ids=mlm_label_prefix_ids,
                chat_template_kwargs=chat_template_kwargs,
                )
            for prompt in batch_prompts
            ]
        max_seq_len = max(len(e['input_ids']) for e in batch_encoded)
        input_ids = []
        attention_mask = []
        for e in batch_encoded:
            seq_ids = list(e['input_ids'])
            seq_attn = list(e['attention_mask'])
            pad_len = max_seq_len - len(seq_ids)
            if tokenizer.padding_side == 'left':
                seq_ids = ([tokenizer.pad_token_id] * pad_len) + seq_ids
                seq_attn = ([0] * pad_len) + seq_attn
            else:
                seq_ids = seq_ids + ([tokenizer.pad_token_id] * pad_len)
                seq_attn = seq_attn + ([0] * pad_len)
            input_ids.append(seq_ids)
            attention_mask.append(seq_attn)
        score_positions = []
        for e in batch_encoded:
            pos = int(e['score_position'])
            if tokenizer.padding_side == 'left':
                pos += max_seq_len - len(e['input_ids'])
            score_positions.append(pos)
        encoded = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long, device=device),
            'attention_mask': torch.tensor(
                attention_mask, dtype=torch.long, device=device
                ),
            }
        score_positions = torch.tensor(
            score_positions,
            dtype=torch.long,
            device=device,
            )

        with torch.no_grad():
            logits = model(**encoded).logits

        next_logits = select_scoring_logits(
            logits,
            score_position=score_positions,
            )
        class_logits = next_logits.index_select(dim=1, index=class_ids)

        pred_idx = torch.argmax(class_logits, dim=1)
        out_top.extend((pred_idx + scale.min).detach().cpu().numpy().tolist())

        batch_gold_token_ids = torch.tensor(
            [
                label_value_to_token_id[int(d)]
                for d in batch_gold.detach().cpu().tolist()
                ],
            dtype=torch.long,
            device=device,
            )
        lm_ce = F.cross_entropy(next_logits.float(), batch_gold_token_ids)
        lm_losses.append(float(lm_ce.detach().cpu().item()))

        class_probs = F.softmax(class_logits.float(), dim=1)
        log_class_probs = F.log_softmax(class_logits.float(), dim=1)
        expected = (class_probs * prob_class_values_t).sum(dim=1)
        out_prob.extend(expected.detach().cpu().numpy().tolist())

        ce_prob = -(batch_soft * log_class_probs).sum(dim=1).mean()
        ce_prob_losses.append(float(ce_prob.detach().cpu().item()))

        target_expected = (batch_soft * prob_class_values_t).sum(dim=1)
        raft = F.mse_loss(expected, target_expected)
        raft_losses.append(float(raft.detach().cpu().item()))

    return (
        np.asarray(out_top, dtype=int),
        np.asarray(out_prob, dtype=float),
        float(np.mean(lm_losses)),
        float(np.mean(ce_prob_losses)),
        float(np.mean(raft_losses)),
        )


def load_model_and_tokenizer(args, adapter_path=None):
    load_tokenizer = AutoTokenizer.from_pretrained
    load_model = AutoModelForCausalLM.from_pretrained
    if (args.model_family == 'causal') and ('/Ministral-3' in args.model_name):
        # Explicit classes are required (as of transformers==5.2.0):
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
        load_tokenizer = MistralCommonBackend.from_pretrained
        load_model = Mistral3ForConditionalGeneration.from_pretrained

    tokenizer = load_tokenizer(
        args.model_name, trust_remote_code=args.trust_remote_code
        )

    if args.model_family == 'causal':
        chat_template_kwargs = (
            # Disable thinking for thiking Qwen3 models:
            dict(enable_thinking=False) if (
                ('/Qwen3-' in args.model_name) and ('-Base' not in args.model_name)
                ) else
            dict()
            )
        tokenizer.padding_side = 'left'
    else:
        chat_template_kwargs = None
        tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError('Tokenizer must define pad_token or eos_token.')
    chat_template_status = (
        'ENABLED'
        if (
            (args.model_family == 'causal') and
            tokenizer_uses_chat_template(tokenizer)
            )
        else 'disabled'
        )
    print(f'  Chat template: {chat_template_status}')

    compute_dtype = torch.bfloat16 if USE_BF16 else torch.float16

    if args.device_map == 'cuda':
        device_map = {'': 0}
    else:
        device_map = 'auto'

    print(
        '  Load settings: '
        f'device_map={args.device_map}'
        )

    if args.head_type == 'regression':
        if args.mode == 'predict' and adapter_path is None:
            raise ValueError('adapter_path is required for --mode predict')
        source_path = str(adapter_path) if (
            args.mode == 'predict'
            ) else args.model_name
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=source_path,
            num_labels=1,
            torch_dtype=compute_dtype,
            device_map=device_map,
            trust_remote_code=args.trust_remote_code,
            )
        model.config.problem_type = 'regression'
        if (
                args.mode == 'finetune-predict' and
                (not args.no_gradient_checkpointing) and
                hasattr(model, 'gradient_checkpointing_enable')
                ):
            model.gradient_checkpointing_enable()
    elif args.model_family == 'mlm':
        load_model = AutoModelForMaskedLM.from_pretrained
        if args.mode == 'predict' and adapter_path is None:
            raise ValueError('adapter_path is required for --mode predict')
        source_path = str(adapter_path) if (
            args.mode == 'predict'
            ) else args.model_name
        model = load_model(
            pretrained_model_name_or_path=source_path,
            torch_dtype=compute_dtype,
            device_map=device_map,
            trust_remote_code=args.trust_remote_code,
            )
        model.config.use_cache = False
        if args.mode == 'finetune-predict' and not args.no_gradient_checkpointing:
            model.gradient_checkpointing_enable()
    else:
        from peft import (
            LoraConfig,
            PeftModel,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
            )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            )
        load_kwargs = dict(
            pretrained_model_name_or_path=args.model_name,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map=device_map,
            trust_remote_code=args.trust_remote_code,
            )
        model = load_model(
            **load_kwargs
            )

        model.config.use_cache = False
        if args.mode == 'finetune-predict':
            if not args.no_gradient_checkpointing:
                model.gradient_checkpointing_enable()

            model = prepare_model_for_kbit_training(model)

            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias='none',
                task_type=TaskType.CAUSAL_LM,
                target_modules='all-linear',
                )
            model = get_peft_model(model, lora_config)
        elif args.mode == 'predict':
            if adapter_path is None:
                raise ValueError('adapter_path is required for --mode predict')
            model = PeftModel.from_pretrained(
                model,
                str(adapter_path),
                is_trainable=False,
                )

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, 'generation_config', None) is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    if args.mode == 'finetune-predict':
        if args.head_type == 'regression':
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f'  Trainable parameters: {trainable:,} / {total:,}')
        elif args.model_family == 'causal':
            model.print_trainable_parameters()
        else:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f'  Trainable parameters: {trainable:,} / {total:,}')
    elif args.mode == 'predict':
        if args.head_type == 'regression':
            print(
                '  Predict mode: loaded sequence-classification model from '
                f'{adapter_path}'
                )
        elif args.model_family == 'causal':
            print(f'  Predict mode: loaded adapter from {adapter_path}')
        else:
            print(f'  Predict mode: loaded fine-tuned MLM from {adapter_path}')
    else:
        print('  Base-predict mode: using base model weights only.')
    return model, tokenizer, chat_template_kwargs


def run_sanity_check(trainer):
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    batch = trainer._prepare_inputs(batch)

    trainer.model.train()
    loss = trainer.compute_loss(trainer.model, dict(batch))
    if not torch.isfinite(loss):
        raise RuntimeError(f'Non-finite loss in sanity check: {loss.item()}')

    loss.backward()

    grad_sq_norm = torch.tensor(0.0, device=loss.device)
    for p in trainer.model.parameters():
        if p.requires_grad and p.grad is not None:
            grad_sq_norm += (p.grad.detach() ** 2).sum()
    grad_norm = torch.sqrt(grad_sq_norm).item()

    if not np.isfinite(grad_norm) or grad_norm == 0.0:
        raise RuntimeError(
            f'Invalid gradient norm in sanity check: {grad_norm}'
            )

    print(
        f'Sanity check: loss={loss.item():.6f} '
        f'grad_norm={grad_norm:.6f}'
        )

    trainer.model.zero_grad(set_to_none=True)


def train_one_fold_scope_regression(
    args,
    data,
    train_idx,
    dev_idx,
    fold_num,
    languages,
    warn_on_truncation=False,
    ):
    selected_langs = [
        lang for lang in PREDICTION_LANG_ORDER if lang in set(languages)
        ]
    if not selected_langs:
        raise ValueError('No languages selected for fold scope training')

    scope_lang = selected_langs[0]
    scope_tag = adapter_scope_tag(args, scope_lang)
    set_seed(args.seed + fold_num)
    fold_out = fold_adapter_dir(args, fold_num, scope_lang)
    adapter_path = (
        resolve_predict_adapter_path(args, fold_num, scope_lang)
        if args.mode == 'predict'
        else None
        )

    if args.vram_stats:
        print_vram_stats('before model load')
    model, tokenizer, chat_template_kwargs = load_model_and_tokenizer(
        args,
        adapter_path=adapter_path,
        )
    if args.vram_stats:
        print_vram_stats('after model load')
    sep_token = _regression_separator(tokenizer)

    train_frames = []
    dev_frames = []
    all_lang_data = {}
    for lang in selected_langs:
        lang_data = build_lang_frame_regression(
            data,
            lang,
            style=args.regression_input_style,
            sep_token=sep_token,
            )
        all_lang_data[lang] = lang_data
        train_lang_df = lang_data.iloc[train_idx].reset_index(drop=True)
        dev_lang_df = lang_data.iloc[dev_idx].reset_index(drop=True)
        train_lang_df['lang'] = lang
        dev_lang_df['lang'] = lang
        train_frames.append(train_lang_df)
        dev_frames.append(dev_lang_df)
    train_df = pd.concat(train_frames, ignore_index=True)
    dev_df = pd.concat(dev_frames, ignore_index=True)
    eval_has_labels = bool(dev_df['score'].notna().all())

    if warn_on_truncation:
        for lang in selected_langs:
            truncated_ids = get_truncated_item_ids_regression(
                all_lang_data[lang],
                tokenizer,
                args.max_length,
                )
            if truncated_ids:
                joined_ids = ', '.join(str(item_id) for item_id in truncated_ids)
                print(
                    f'  WARNING: {len(truncated_ids)} input(s) will be '
                    f'truncated for lang={lang} at '
                    f'max_length={args.max_length}.'
                    )
                print(f'  Truncated item_ids: {joined_ids}')

    trainer = None
    train_start = time.perf_counter()
    train_loss = np.nan
    if args.mode != 'finetune-predict':
        if args.sanity_check:
            print(
                f'  Warning: --sanity-check ignored in --mode {args.mode}.'
                )
    else:
        use_early_stopping = args.early_stop_patience > 0
        if use_early_stopping and not eval_has_labels:
            print(
                '  Warning: disabling early stopping because evaluation '
                'split has no labels.'
                )
            use_early_stopping = False

        train_ds = encode_regression_dataset(
            train_df,
            tokenizer,
            args.max_length,
            )
        eval_df_for_ds = dev_df.copy()
        if not eval_has_labels:
            eval_df_for_ds['score'] = 0.0
        eval_ds = encode_regression_dataset(
            eval_df_for_ds,
            tokenizer,
            args.max_length,
            )
        training_args = make_training_args(
            output_dir=str(fold_out),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler,
            optim='adamw_torch',
            bf16=USE_BF16,
            fp16=not USE_BF16,
            eval_strategy=(
                'steps'
                if use_early_stopping
                else 'no'
                ),
            save_strategy=(
                'steps'
                if use_early_stopping
                else 'no'
                ),
            eval_steps=(
                args.eval_steps
                if use_early_stopping
                else None
                ),
            save_steps=(
                args.eval_steps
                if use_early_stopping
                else None
                ),
            load_best_model_at_end=use_early_stopping,
            metric_for_best_model=(
                'eval_loss'
                if use_early_stopping
                else None
                ),
            greater_is_better=(
                False
                if use_early_stopping
                else None
                ),
            logging_strategy='steps',
            logging_steps=10,
            report_to='none',
            seed=args.seed + fold_num,
            remove_unused_columns=False,
            )
        trainer = make_trainer_compat(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            )
        if use_early_stopping:
            trainer.add_callback(
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stop_patience,
                    early_stopping_threshold=args.early_stop_threshold,
                    )
                )
        if args.sanity_check:
            run_sanity_check(trainer)
        if args.vram_stats:
            print_vram_stats('before train (reset peak)', reset_peak=True)
        train_result = trainer.train()
        train_loss = float(train_result.training_loss)

    def predict_regression(eval_df_):
        eval_df_for_pred = eval_df_.copy()
        if eval_df_for_pred['score'].isna().any():
            eval_df_for_pred['score'] = 0.0
        pred_ds = encode_regression_dataset(
            eval_df_for_pred,
            tokenizer,
            args.max_length,
            )
        pred_runner = trainer
        if pred_runner is None:
            pred_args = make_training_args(
                output_dir=str(fold_out),
                per_device_eval_batch_size=args.batch_size,
                report_to='none',
                bf16=USE_BF16,
                fp16=not USE_BF16,
                remove_unused_columns=False,
                )
            pred_runner = make_trainer_compat(
                model=model,
                args=pred_args,
                tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
                )
        pred_out = pred_runner.predict(pred_ds)
        pred_values = pred_out.predictions
        if isinstance(pred_values, tuple):
            pred_values = pred_values[0]
        pred_values = np.asarray(pred_values, dtype=float)
        if pred_values.ndim > 1:
            pred_values = pred_values[:, 0]
        return pred_values

    train_by_lang = {
        lang: train_df[train_df['lang'] == lang].reset_index(drop=True)
        for lang in selected_langs
        }
    dev_by_lang = {
        lang: dev_df[dev_df['lang'] == lang].reset_index(drop=True)
        for lang in selected_langs
        }

    dev_outputs_by_lang = {}
    train_outputs_by_lang = {}
    for lang in selected_langs:
        lang_dev_df = dev_by_lang[lang]
        dev_pred_score = predict_regression(lang_dev_df)
        if eval_has_labels:
            dev_metrics = regression_metrics(
                lang_dev_df['score'].to_numpy(),
                dev_pred_score,
                )
            dev_metrics['final_eval_loss'] = np.nan
        else:
            dev_metrics = {
                'rmse': np.nan,
                'r2': np.nan,
                'pearson': np.nan,
                'spearman': np.nan,
                'final_eval_loss': np.nan,
                }
        dev_outputs_by_lang[lang] = {
            'metrics': dev_metrics,
            'item_ids': lang_dev_df[ID_COL].to_numpy(),
            'predictions': dev_pred_score,
            }

        if args.predict_train:
            lang_train_df = train_by_lang[lang]
            train_pred_score = predict_regression(lang_train_df)
            train_metrics = regression_metrics(
                lang_train_df['score'].to_numpy(),
                train_pred_score,
                )
            train_metrics['final_eval_loss'] = np.nan
            train_outputs_by_lang[lang] = {
                'metrics': train_metrics,
                'item_ids': lang_train_df[ID_COL].to_numpy(),
                'predictions': train_pred_score,
                }

    elapsed_sec = time.perf_counter() - train_start
    if args.vram_stats:
        print_vram_stats('after train+predict')
    for lang in selected_langs:
        dev_outputs_by_lang[lang]['metrics']['train_loss'] = train_loss
        dev_outputs_by_lang[lang]['metrics']['train_time_sec'] = float(elapsed_sec)

    if trainer is not None:
        model_out = fold_out / 'adapter'
        trainer.save_model(str(model_out))

    return {
        'dev_outputs_by_lang': dev_outputs_by_lang,
        'train_outputs_by_lang': train_outputs_by_lang if args.predict_train else None,
        'checkpoint_metric_rows': [],
        'scope_tag': scope_tag,
        }


def train_one_fold_scope(
    args,
    data,
    train_idx,
    dev_idx,
    fold_num,
    languages,
    prompt,
    warn_on_truncation=False,
    ):
    selected_langs = [
        lang for lang in PREDICTION_LANG_ORDER if lang in set(languages)
        ]
    if not selected_langs:
        raise ValueError('No languages selected for fold scope training')
    if args.head_type == 'regression':
        return train_one_fold_scope_regression(
            args,
            data,
            train_idx,
            dev_idx,
            fold_num,
            languages,
            warn_on_truncation=warn_on_truncation,
            )

    scope_lang = selected_langs[0]
    scope_tag = adapter_scope_tag(args, scope_lang)
    set_seed(args.seed + fold_num)
    fold_out = fold_adapter_dir(args, fold_num, scope_lang)
    epoch_checkpoint_root = fold_out / 'epoch_checkpoints'
    adapter_path = (
        resolve_predict_adapter_path(args, fold_num, scope_lang)
        if args.mode == 'predict'
        else None
        )
    adapter_metadata = (
        load_adapter_metadata(adapter_path)
        if adapter_path is not None
        else None
        )
    if args.mode == 'predict':
        saved_family = (
            'causal' if (not adapter_metadata) else
            str(adapter_metadata.get('model_family', 'causal'))
            )
        if saved_family != args.model_family:
            raise ValueError(
                'Model family mismatch for --mode predict: '
                f'checkpoint metadata has model_family={saved_family}, '
                f'but CLI requests model_family={args.model_family}.'
                )
    runtime_meta = resolve_effective_scale_and_space(
        adapter_metadata,
        mode=args.mode,
        cli_scale_min=args.scale_min,
        cli_scale_max=args.scale_max,
        cli_space=args.space,
        )
    for msg in runtime_meta['messages']:
        print(f'  {msg}')
    scale = PointScale(runtime_meta['scale_min'], runtime_meta['scale_max'])
    active_space = runtime_meta['active_space']
    output_is_difficulty = prompt_is_for_difficulty(prompt)
    predict_mode = args.predict_mode
    uses_prob_scale = args.loss_type in PROB_LOSS_TYPES

    train_frames = []
    dev_frames = []
    all_lang_data = {}
    for lang in selected_langs:
        lang_data = build_lang_frame(
            data,
            lang,
            prompt,
            scale=scale,
            )
        all_lang_data[lang] = lang_data
        train_lang_df = lang_data.iloc[train_idx].reset_index(drop=True)
        dev_lang_df = lang_data.iloc[dev_idx].reset_index(drop=True)
        train_lang_df['lang'] = lang
        dev_lang_df['lang'] = lang
        train_frames.append(train_lang_df)
        dev_frames.append(dev_lang_df)

    train_df = pd.concat(train_frames, ignore_index=True)
    dev_df = pd.concat(dev_frames, ignore_index=True)
    eval_has_labels = bool(dev_df['score'].notna().all())
    curriculum_enabled = (
        args.mode == 'finetune-predict' and args.curriculum == 'hardness2'
        )
    curriculum_trainer_kwargs = {}
    if curriculum_enabled:
        lang2hardness_by_item = {}
        for lang in selected_langs:
            hardness_source = data.iloc[train_idx][[
                ID_COL,
                'en_target_word',
                f'{lang}_GLMM_score',
                ]]
            hardness = np.abs(tubelex_lr_error(hardness_source, lang))
            lang2hardness_by_item[lang] = pd.Series(
                hardness,
                index=hardness_source[ID_COL].to_numpy()
                )
        train_df['hardness'] = train_df.apply(
            lambda row: float(
                lang2hardness_by_item[row['lang']].loc[row[ID_COL]]
                ),
            axis=1,
            )
        if train_df['hardness'].isna().any():
            raise ValueError('Failed to compute hardness for all train rows.')
        n_train = len(train_df)
        sorted_ilocs = np.argsort(train_df['hardness'].to_numpy(), kind='mergesort')
        easy_count = n_train // 2
        easy_indices = sorted_ilocs[:easy_count].tolist()
        hard_indices = sorted_ilocs[easy_count:].tolist()
        print(
            '  Curriculum: hardness2 '
            f'(N={n_train}, easy={len(easy_indices)}, hard={len(hard_indices)})'
            )
        curriculum_trainer_kwargs = {
            'curriculum_mode': 'hardness2',
            'curriculum_easy_indices': easy_indices,
            'curriculum_hard_indices': hard_indices,
            'curriculum_total_epochs': int(round(args.epochs)),
            'curriculum_seed': args.seed + fold_num,
            }
    else:
        print(f'  Curriculum: {args.curriculum}')

    train_df['target_score'] = scores_to_active_space(
        train_df['score'].to_numpy(),
        active_space,
        )
    if eval_has_labels:
        dev_df['target_score'] = scores_to_active_space(
            dev_df['score'].to_numpy(),
            active_space,
            )

    default_edges = fit_bin_edges(
        train_df['target_score'].to_numpy(),
        scale=scale,
        )
    prob_lo_logit, prob_hi_logit = fit_score_range(
        train_df['score'].to_numpy(),
        leeway=args.leeway,
        )
    saved_interp = resolve_saved_interpretation_params(
        adapter_metadata,
        n_points=scale.n_points,
        )
    for msg in saved_interp['messages']:
        print(f'  Warning: {msg}')
    saved_edges = saved_interp['bin_edges_active']
    edges = saved_edges if saved_edges is not None else default_edges

    saved_prob_class_values = saved_interp['prob_class_values_active']
    if saved_prob_class_values is not None:
        prob_class_values = saved_prob_class_values
        meta_data_min = saved_interp['data_min']
        meta_data_max = saved_interp['data_max']
        if meta_data_min is not None and meta_data_max is not None:
            prob_lo_logit = meta_data_min
            prob_hi_logit = meta_data_max
    else:
        meta_data_min = saved_interp['data_min']
        meta_data_max = saved_interp['data_max']
        if meta_data_min is not None and meta_data_max is not None:
            prob_lo_logit = meta_data_min
            prob_hi_logit = meta_data_max
        prob_endpoints_active = scores_to_active_space(
            np.array([prob_lo_logit, prob_hi_logit], dtype=float),
            active_space,
            )
        prob_class_values = fit_uniform_score_points(
            prob_endpoints_active,
            scale=scale,
            leeway=0.0,
            )
    train_df['bin_label'] = scores_to_bin_labels(
        train_df['target_score'].to_numpy(),
        edges,
        scale=scale,
        )
    train_soft_probs = (
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
    train_df['soft_probs'] = list(train_soft_probs)
    if eval_has_labels:
        dev_df['gold_bin_label'] = scores_to_bin_labels(
            dev_df['target_score'].to_numpy(),
            edges,
            scale=scale,
            )
        dev_soft_probs = (
            scores_to_soft_label_probs(
                dev_df['target_score'].to_numpy(),
                prob_class_values,
                scale=scale,
                )
            if uses_prob_scale
            else scores_to_soft_bin_probs(
                dev_df['target_score'].to_numpy(),
                edges,
                scale=scale,
                )
            )
    else:
        dev_df['gold_bin_label'] = int(scale.min)
        dev_soft_probs = np.zeros((len(dev_df), scale.n_points), dtype=float)
        dev_soft_probs[:, 0] = 1.0
    if not output_is_difficulty:
        # Internal bin helpers are difficulty-oriented; flip for facility prompts.
        train_df['bin_label'] = reverse_scale_labels(
            train_df['bin_label'].to_numpy(),
            scale=scale,
            )
        train_df['soft_probs'] = list(
            reverse_scale_probs(np.stack(train_df['soft_probs'].to_numpy()))
            )
        dev_df['gold_bin_label'] = reverse_scale_labels(
            dev_df['gold_bin_label'].to_numpy(),
            scale=scale,
            )
        dev_soft_probs = reverse_scale_probs(dev_soft_probs)
        prob_class_values = prob_class_values[::-1].copy()
    dev_df['soft_probs'] = list(dev_soft_probs)
    train_df['sample_weight'] = compute_sample_weights_from_soft_probs(
        np.stack(train_df['soft_probs'].to_numpy()),
        args.reweight,
        )
    dev_df['sample_weight'] = 1.0

    if args.vram_stats:
        print_vram_stats('before model load')
    model, tokenizer, chat_template_kwargs = load_model_and_tokenizer(
        args,
        adapter_path=adapter_path,
        )
    (
        label_value_to_token_id,
        mlm_label_prefix_ids,
        ) = get_scale_label_token_ids(
        tokenizer,
        scale=scale,
        token_form=args.token_form,
        model_family=args.model_family,
        )
    if args.vram_stats:
        print_vram_stats('after model load')
    if warn_on_truncation:
        for lang in selected_langs:
            truncated_ids = get_truncated_item_ids(
                all_lang_data[lang],
                tokenizer,
                args.max_length,
                model_family=args.model_family,
                mlm_label_prefix_ids=mlm_label_prefix_ids,
                chat_template_kwargs=chat_template_kwargs,
                )
            if truncated_ids:
                joined_ids = ', '.join(str(item_id) for item_id in truncated_ids)
                print(
                    f'  WARNING: {len(truncated_ids)} prompt(s) will be '
                    f'truncated for lang={lang} at '
                    f'max_length={args.max_length}.'
                    )
                print(f'  Truncated item_ids: {joined_ids}')
    scale_values = list(range(scale.min, scale.max + 1))
    print(
        f'  Scale label token ids: '
        f'{", ".join(f"{v}:{label_value_to_token_id[v]}" for v in scale_values)}'
        )
    if args.model_family == 'mlm' and mlm_label_prefix_ids:
        print(
            '  MLM label prefix token ids: '
            f'{mlm_label_prefix_ids}'
            )
    if uses_prob_scale:
        print(
            '  Prob class values (active-space, label order '
            f'{scale.min}..{scale.max}): '
            f'{np.array2string(prob_class_values, precision=6)}'
            )

    trainer = None
    seen_epoch_snapshots = set()
    checkpoint_metric_rows = []
    adapter_calibration = parse_adapter_calibration(adapter_metadata)
    auto_calibration = (
        (args.mode == 'predict') and
        (adapter_calibration is not None) and
        (not args.calibrate) and
        (not args.disable_adapter_calibration)
        )
    if args.mode == 'predict' and args.disable_adapter_calibration:
        print('  Calibration: disabled via --disable-adapter-calibration')
    elif auto_calibration:
        print('  Calibration: enabled from adapter metadata')
    elif args.calibrate:
        if args.mode == 'predict' and adapter_calibration is not None:
            print('  Calibration: using adapter metadata')
        elif args.mode == 'predict':
            print(
                '  Warning: --calibrate requested in --mode predict '
                'but adapter metadata has no saved parameters; skipping.'
                )
    use_calibration = bool(
        args.calibrate or auto_calibration
        )

    def predict_split_raw(
        eval_df, gold_label_col, gold_soft_probs, model_, tokenizer_,
        chat_template_kwargs=None
        ):
        (
            pred_top,
            pred_prob,
            eval_lm_loss,
            eval_ce_prob_loss,
            eval_raft_loss,
            ) = predict_scale_labels_and_losses(
            model=model_,
            tokenizer=tokenizer_,
            prompts=eval_df['prompt'].tolist(),
            gold_labels=eval_df[gold_label_col].to_numpy(),
            gold_soft_probs=gold_soft_probs,
            max_length=args.max_length,
            batch_size=args.batch_size,
            label_value_to_token_id=label_value_to_token_id,
            scale=scale,
            model_family=args.model_family,
            mlm_label_prefix_ids=mlm_label_prefix_ids,
            prob_class_values=prob_class_values.tolist(),
            chat_template_kwargs=chat_template_kwargs
            )

        if predict_mode == 'prob':
            pred_score_active = pred_prob
        else:
            if not output_is_difficulty:
                pred_top = reverse_scale_labels(pred_top, scale=scale)
            pred_score_active = bin_labels_to_scores(
                pred_top,
                edges,
                scale=scale,
                )

        final_eval_loss = (
            float(eval_raft_loss)
            if (args.loss_type == 'raft')
            else (
                float(eval_ce_prob_loss)
                if (args.loss_type in ('ce_prob', 'delta2'))
                else float(eval_lm_loss)
                )
            )
        return pred_score_active, final_eval_loss

    def evaluate_split(
        eval_df,
        gold_label_col,
        gold_soft_probs,
        model_,
        tokenizer_,
        chat_template_kwargs,
        calibration=None,
        has_labels=True,
        ):
        pred_score_active, final_eval_loss = predict_split_raw(
            eval_df,
            gold_label_col=gold_label_col,
            gold_soft_probs=gold_soft_probs,
            model_=model_,
            tokenizer_=tokenizer_,
            chat_template_kwargs=chat_template_kwargs
            )
        pred_score_active = apply_calibration(
            pred_score_active,
            calibration,
            )
        if has_labels:
            pred_score_logit = scores_from_active_space(pred_score_active, active_space)
            metrics = regression_metrics(
                eval_df['score'].to_numpy(),
                pred_score_logit,
                )
            metrics['final_eval_loss'] = final_eval_loss
        else:
            metrics = {
                'rmse': np.nan,
                'r2': np.nan,
                'pearson': np.nan,
                'spearman': np.nan,
                'final_eval_loss': np.nan,
                }
        return metrics, pred_score_active

    def fit_current_calibration(model_, tokenizer_, chat_template_kwargs):
        if not use_calibration:
            return None
        if args.mode == 'predict':
            if args.disable_adapter_calibration:
                return None
            return adapter_calibration
        train_pred_active_raw, _ = predict_split_raw(
            train_df,
            gold_label_col='bin_label',
            gold_soft_probs=np.stack(train_df['soft_probs'].to_numpy()),
            model_=model_,
            tokenizer_=tokenizer_,
            chat_template_kwargs=chat_template_kwargs
            )
        calibration = fit_calibration(
            train_pred_active_raw,
            train_df['target_score'].to_numpy(),
            )
        print(
            '  Calibration fit '
            f'(coef={calibration["coef"]:.6f}, '
            f'intercept={calibration["intercept"]:.6f})'
            )
        return calibration

    def save_epoch_snapshot(epoch_value, reason, model_, tokenizer_,
                            chat_template_kwargs):
        if trainer is None:
            return None
        epoch_value = float(epoch_value)
        tag = _epoch_dir_tag(epoch_value)
        if tag in seen_epoch_snapshots:
            return None
        seen_epoch_snapshots.add(tag)

        checkpoint_dir = epoch_checkpoint_root / tag
        adapter_dir = checkpoint_dir / 'adapter'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(adapter_dir))
        calibration = fit_current_calibration(model_, tokenizer_, chat_template_kwargs)
        save_adapter_metadata(
            adapter_dir,
            build_adapter_metadata(
                scale=scale,
                active_space=active_space,
                data_min=prob_lo_logit,
                data_max=prob_hi_logit,
                bin_edges_active=edges,
                prob_class_values_active=prob_class_values,
                calibration=calibration,
                model_family=args.model_family,
                ),
            )

        overall_metrics, pred_score_active = evaluate_split(
            dev_df,
            gold_label_col='gold_bin_label',
            gold_soft_probs=dev_soft_probs,
            chat_template_kwargs=chat_template_kwargs,
            calibration=calibration,
            model_=model_,
            tokenizer_=tokenizer_,
            has_labels=eval_has_labels,
            )
        pred_path = checkpoint_dir / 'dev_predictions.csv'
        out = {
            ID_COL: dev_df[ID_COL].to_numpy(),
            'prediction': pred_score_active,
            }
        if args.all_in_one:
            out['lang'] = dev_df['lang'].to_numpy()
        pd.DataFrame(out).to_csv(pred_path, index=False)

        label = _epoch_label(epoch_value)
        print(
            f'    Checkpoint epoch={label} ({reason}) '
            f'rmse={overall_metrics["rmse"]:.4f} '
            f'r2={overall_metrics["r2"]:.4f} '
            f'pearson={overall_metrics["pearson"]:.4f} '
            f'spearman={overall_metrics["spearman"]:.4f} '
            f'final_eval_loss={overall_metrics["final_eval_loss"]:.4f}'
            )
        print(f'    Saved checkpoint adapter: {adapter_dir}')
        print(
            '    Saved checkpoint adapter metadata: '
            f'{adapter_metadata_path(adapter_dir)}'
            )
        print(f'    Saved checkpoint predictions: {pred_path}')
        if args.all_in_one:
            for lang in selected_langs:
                lang_dev_df = dev_df[dev_df['lang'] == lang].reset_index(drop=True)
                lang_dev_soft_probs = np.stack(
                    lang_dev_df['soft_probs'].to_numpy()
                    )
                lang_metrics, _ = evaluate_split(
                    lang_dev_df,
                    gold_label_col='gold_bin_label',
                    gold_soft_probs=lang_dev_soft_probs,
                    chat_template_kwargs=chat_template_kwargs,
                    calibration=calibration,
                    model_=model_,
                    tokenizer_=tokenizer_,
                    has_labels=eval_has_labels,
                    )
                checkpoint_metric_rows.append(
                    {
                        'lang': lang,
                        'checkpoint_epoch': float(epoch_value),
                        'checkpoint_reason': reason,
                        **lang_metrics,
                        }
                    )
        else:
            checkpoint_metric_rows.append(
                {
                    'checkpoint_epoch': float(epoch_value),
                    'checkpoint_reason': reason,
                    **overall_metrics,
                    }
                )
        return overall_metrics, pred_score_active

    class EpochCheckpointCallback(TrainerCallback):
        def __init__(self, tokenizer_):
            self._tokenizer = tokenizer_

        def on_epoch_end(self, args_, state, control, **kwargs):
            epoch_val = getattr(state, 'epoch', None)
            if epoch_val is None:
                return control
            completed_epoch = _completed_epoch_from_state(epoch_val)
            if completed_epoch is None:
                return control
            if _is_final_epoch_snapshot(completed_epoch, args.epochs):
                return control
            was_training = kwargs['model'].training
            save_epoch_snapshot(
                completed_epoch,
                reason='epoch_end',
                model_=kwargs['model'],
                tokenizer_=self._tokenizer,
                chat_template_kwargs=chat_template_kwargs,
                )
            if was_training:
                kwargs['model'].train()
            return control

        def on_train_end(self, args_, state, control, **kwargs):
            epoch_val = getattr(state, 'epoch', None)
            if epoch_val is None:
                return control
            epoch_val = float(epoch_val)
            if epoch_val <= 0:
                return control
            # Prefer an integer epoch label when trainer state is slightly off.
            snapped_epoch = _snap_epoch_if_near_integer(epoch_val)
            epoch_to_save = snapped_epoch if snapped_epoch is not None else epoch_val
            if _is_final_epoch_snapshot(epoch_to_save, args.epochs):
                return control
            was_training = kwargs['model'].training
            save_epoch_snapshot(
                epoch_to_save,
                reason='train_end',
                model_=kwargs['model'],
                tokenizer_=self._tokenizer,
                chat_template_kwargs=chat_template_kwargs,
                )
            if was_training:
                kwargs['model'].train()
            return control

    train_start = time.perf_counter()
    train_loss = np.nan
    if args.mode != 'finetune-predict':
        if args.sanity_check:
            print(
                f'  Warning: --sanity-check ignored in --mode {args.mode}.'
                )
    else:
        use_early_stopping = args.early_stop_patience > 0
        if use_early_stopping and not eval_has_labels:
            print(
                '  Warning: disabling early stopping because evaluation '
                'split has no labels.'
                )
            use_early_stopping = False

        if uses_prob_scale:
            train_ds = encode_prompt_dataset(
                train_df,
                tokenizer,
                args.max_length,
                model_family=args.model_family,
                mlm_label_prefix_ids=mlm_label_prefix_ids,
                chat_template_kwargs=chat_template_kwargs,
                )
            eval_ds = encode_prompt_dataset(
                dev_df,
                tokenizer,
                args.max_length,
                model_family=args.model_family,
                mlm_label_prefix_ids=mlm_label_prefix_ids,
                chat_template_kwargs=chat_template_kwargs,
                )
        else:
            train_ds = encode_train_dataset(
                train_df,
                tokenizer,
                args.max_length,
                label_value_to_token_id,
                model_family=args.model_family,
                mlm_label_prefix_ids=mlm_label_prefix_ids,
                chat_template_kwargs=chat_template_kwargs,
                )
            eval_ds = encode_train_dataset(
                dev_df.assign(bin_label=dev_df['gold_bin_label']),
                tokenizer,
                args.max_length,
                label_value_to_token_id,
                model_family=args.model_family,
                mlm_label_prefix_ids=mlm_label_prefix_ids,
                chat_template_kwargs=chat_template_kwargs,
                )
        training_args = make_training_args(
            output_dir=str(fold_out),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler,
            optim='paged_adamw_8bit',
            bf16=USE_BF16,
            fp16=not USE_BF16,
            eval_strategy=(
                'steps'
                if use_early_stopping
                else 'no'
                ),
            save_strategy=(
                'steps'
                if use_early_stopping
                else 'no'
                ),
            eval_steps=(
                args.eval_steps
                if use_early_stopping
                else None
                ),
            save_steps=(
                args.eval_steps
                if use_early_stopping
                else None
                ),
            load_best_model_at_end=use_early_stopping,
            metric_for_best_model=(
                'eval_loss'
                if use_early_stopping
                else None
                ),
            greater_is_better=(
                False
                if use_early_stopping
                else None
                ),
            logging_strategy='steps',
            logging_steps=10,
            report_to='none',
            seed=args.seed + fold_num,
            remove_unused_columns=False,
            )

        if args.loss_type == 'raft':
            trainer = RaftLossTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tokenizer,
                data_collator=PromptOnlyCollator(tokenizer),
                loss_decimals=args.loss_decimals,
                class_token_ids=[
                    label_value_to_token_id[v]
                    for v in scale_values
                    ],
                class_values=prob_class_values.tolist(),
                **curriculum_trainer_kwargs,
                )
            trainer.label_names = ['target_probs', 'sample_weight']
        elif args.loss_type == 'ce_prob':
            trainer = SoftCELossTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tokenizer,
                data_collator=PromptOnlyCollator(tokenizer),
                loss_decimals=args.loss_decimals,
                class_token_ids=[
                    label_value_to_token_id[v]
                    for v in scale_values
                    ],
                class_values=prob_class_values.tolist(),
                **curriculum_trainer_kwargs,
                )
            trainer.label_names = ['target_probs', 'sample_weight']
        elif args.loss_type == 'delta2':
            trainer = DeltaSquaredLossTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tokenizer,
                data_collator=PromptOnlyCollator(tokenizer),
                loss_decimals=args.loss_decimals,
                class_token_ids=[
                    label_value_to_token_id[v]
                    for v in scale_values
                    ],
                class_values=prob_class_values.tolist(),
                **curriculum_trainer_kwargs,
                )
            trainer.label_names = ['target_probs', 'sample_weight']
        else:
            trainer = FormattedLossTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tokenizer,
                data_collator=CausalDataCollator(tokenizer),
                loss_decimals=args.loss_decimals,
                **curriculum_trainer_kwargs,
                )
        trainer.add_callback(EpochCheckpointCallback(tokenizer))
        if args.vram_stats:
            print_vram_stats('after trainer init')

        if use_early_stopping:
            trainer.add_callback(
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stop_patience,
                    early_stopping_threshold=args.early_stop_threshold,
                    )
                )

        if args.sanity_check:
            run_sanity_check(trainer)

        if args.vram_stats:
            print_vram_stats('before train (reset peak)', reset_peak=True)
        train_result = trainer.train()
        train_loss = float(train_result.training_loss)

    train_by_lang = {
        lang: train_df[train_df['lang'] == lang].reset_index(drop=True)
        for lang in selected_langs
        }
    dev_by_lang = {
        lang: dev_df[dev_df['lang'] == lang].reset_index(drop=True)
        for lang in selected_langs
        }

    final_calibration = fit_current_calibration(model, tokenizer, chat_template_kwargs)
    dev_outputs_by_lang = {}
    train_outputs_by_lang = {}
    for lang in selected_langs:
        lang_dev_df = dev_by_lang[lang]
        lang_dev_soft_probs = np.stack(lang_dev_df['soft_probs'].to_numpy())
        dev_metrics, dev_pred_score_active = evaluate_split(
            lang_dev_df,
            gold_label_col='gold_bin_label',
            gold_soft_probs=lang_dev_soft_probs,
            chat_template_kwargs=chat_template_kwargs,
            calibration=final_calibration,
            model_=model,
            tokenizer_=tokenizer,
            has_labels=eval_has_labels,
            )
        dev_outputs_by_lang[lang] = {
            'metrics': dev_metrics,
            'item_ids': lang_dev_df[ID_COL].to_numpy(),
            'predictions': dev_pred_score_active,
            }

        if args.predict_train:
            lang_train_df = train_by_lang[lang]
            lang_train_soft_probs = np.stack(lang_train_df['soft_probs'].to_numpy())
            train_metrics, train_pred_score_active = evaluate_split(
                lang_train_df,
                gold_label_col='bin_label',
                gold_soft_probs=lang_train_soft_probs,
                chat_template_kwargs=chat_template_kwargs,
                calibration=final_calibration,
                model_=model,
                tokenizer_=tokenizer,
                )
            train_outputs_by_lang[lang] = {
                'metrics': train_metrics,
                'item_ids': lang_train_df[ID_COL].to_numpy(),
                'predictions': train_pred_score_active,
                }

    elapsed_sec = time.perf_counter() - train_start
    if args.vram_stats:
        print_vram_stats('after train+predict')
    for lang in selected_langs:
        dev_outputs_by_lang[lang]['metrics']['train_loss'] = train_loss
        dev_outputs_by_lang[lang]['metrics']['train_time_sec'] = float(elapsed_sec)

    if trainer is not None:
        adapter_path = fold_out / 'adapter'
        trainer.save_model(str(adapter_path))
        save_adapter_metadata(
            adapter_path,
            build_adapter_metadata(
                scale=scale,
                active_space=active_space,
                data_min=prob_lo_logit,
                data_max=prob_hi_logit,
                bin_edges_active=edges,
                prob_class_values_active=prob_class_values,
                calibration=final_calibration,
                model_family=args.model_family,
                ),
            )

    return {
        'dev_outputs_by_lang': dev_outputs_by_lang,
        'train_outputs_by_lang': train_outputs_by_lang if args.predict_train else None,
        'checkpoint_metric_rows': checkpoint_metric_rows,
        'scope_tag': scope_tag,
        }


def save_fold_predictions(
    args,
    fold_i,
    total_folds,
    fold_predictions,
    split_name='dev',
    ):
    selected_langs = [
        lang for lang in PREDICTION_LANG_ORDER if lang in args.languages
        ]
    missing_langs = [
        lang for lang in selected_langs if lang not in fold_predictions
        ]
    if missing_langs:
        raise ValueError(
            f'Missing language predictions for fold {fold_i}: {missing_langs}'
            )

    fold_df = None
    for lang in selected_langs:
        pred_df = fold_predictions[lang].copy()
        pred_df.rename(
            columns={'prediction': f'{lang}_ftllm_output'},
            inplace=True,
            )
        if fold_df is None:
            fold_df = pred_df
        else:
            fold_df = fold_df.merge(pred_df, on=ID_COL, how='inner')

    fold_df = fold_df[
        [ID_COL, *[f'{lang}_ftllm_output' for lang in selected_langs]]
        ]
    fold_df.sort_values(by=ID_COL, inplace=True)

    out_path = fold_prediction_output_path(args, fold_i, total_folds, split_name)
    assert_prediction_output_writable(
        out_path,
        overwrite=args.overwrite_predictions,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out_path, index=False)
    print(f'  Saved {split_name} fold predictions: {out_path}')


def save_final_data_predictions(args, full_predictions):
    selected_langs = [
        lang for lang in PREDICTION_LANG_ORDER if lang in args.languages
        ]
    missing_langs = [
        lang for lang in selected_langs if lang not in full_predictions
        ]
    if missing_langs:
        raise ValueError(
            f'Missing language predictions for final-data run: {missing_langs}'
            )

    full_df = None
    for lang in selected_langs:
        pred_df = full_predictions[lang].copy()
        pred_df.rename(
            columns={'prediction': f'{lang}_ftllm_output'},
            inplace=True,
            )
        if full_df is None:
            full_df = pred_df
        else:
            full_df = full_df.merge(pred_df, on=ID_COL, how='inner')
        if full_df.empty:
            raise ValueError(
                f'No overlapping {ID_COL} rows while merging final-data '
                f'predictions (latest lang={lang}).'
                )

    out_path = final_data_prediction_output_path(args)
    assert_prediction_output_writable(out_path, args.overwrite_predictions)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f'  Saved final-data test predictions: {out_path}')


def print_results(df):
    print('\nPer fold/language metrics')
    print('=========================')
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    metric_cols = ['rmse', 'r2', 'pearson', 'spearman']

    by_lang = (
        df.groupby('lang')[metric_cols].agg(['mean', 'std']).sort_index()
        )
    print('\nMean±SD across folds by language')
    print('================================')
    for lang in by_lang.index:
        line = [f'{lang}:']
        for metric in metric_cols:
            mean = by_lang.loc[lang, (metric, 'mean')]
            std = by_lang.loc[lang, (metric, 'std')]
            if np.isnan(std):
                line.append(f'{metric}={mean:.4f}')
            else:
                line.append(f'{metric}={mean:.4f}±{std:.4f}')
        print(' '.join(line))

    macro = df[metric_cols].mean(numeric_only=True)
    print('\nMacro-average (all folds/languages)')
    print('===================================')
    print(' '.join(f'{m}={macro[m]:.4f}' for m in metric_cols))


def main(args):
    run_start = time.perf_counter()
    dtype_label = 'BF16' if USE_BF16 else 'FP16'
    scale = PointScale(args.scale_min, args.scale_max)
    args.predict_mode = resolve_predict_mode(args)
    if args.curriculum != 'none' and args.mode != 'finetune-predict':
        print(
            f'Warning: ignoring --curriculum {args.curriculum} in '
            f'--mode {args.mode}.',
            file=sys.stderr,
            )
        args.curriculum = 'none'

    data_cv = read_data_cv(
        (None if args.final_data else args.splits),
        train=('full' if args.final_data else None),
        eval=('test' if args.final_data else None)
        )
    data = data_cv.data

    total_folds = len(data_cv.cv)
    selected_langs = [
        lang for lang in PREDICTION_LANG_ORDER if lang in args.languages
        ]
    if not selected_langs:
        raise ValueError('No languages selected.')
    if args.final_data and args.predict_train:
        raise ValueError(
            '--predict-train is not supported with --final-data '
            '(evaluation runs on unlabeled test data in this mode).'
            )

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
    else:
        assert args.final_data or args.cv_mode == 'whole'
        # Do not do anything.

    if not splits:
        raise ValueError(
            f'No splits selected for cv-mode={args.cv_mode}. '
            f'Need at least 2 folds for remaining mode.'
            )

    print(f'Model: {args.model_name}')
    print(f'Head type: {args.head_type}')
    print(f'Model family: {args.model_family}')
    if args.head_type == 'regression':
        print(f'Regression input style: {args.regression_input_style}')
    elif args.model_family == 'causal':
        print(f'QLoRA config: 4-bit, NF4, {dtype_label} compute, double-quant ON')
    else:
        print(f'MLM full-FT config: dtype={dtype_label}, no QLoRA')
    print(f'Mode: {args.mode}')
    if args.head_type == 'regression':
        print('Loss type: regression (model default MSE)')
        print('Calibration (requested): no')
        print('Prediction mode: regression')
        print(
            'Task: Sequence classification regression, predict one '
            f'scalar score ({args.scale_min}-{args.scale_max})'
            )
    else:
        print(f'Loss type: {args.loss_type}')
        print(f'Reweighting: {args.reweight}')
        print(f'Space: {args.space}')
        print(
            'Calibration (requested): '
            f'{"yes" if args.calibrate else "no"}'
            )
        print(
            f'Prediction mode: {args.predict_mode}'
            f' (cli={args.predict})'
            )
        print(f'Token form: {args.token_form}')
        print(
            f'Prompt output polarity: '
            f'{"difficulty" if prompt_is_for_difficulty(args.prompt) else "facility"}'
            )
        if args.model_family == 'causal':
            print(
                'Task: Causal LM, generate one scale label token '
                f'({args.scale_min}-{args.scale_max}) from prompted input'
                )
        else:
            print(
                'Task: Masked LM, predict one masked scale label token '
                f'({args.scale_min}-{args.scale_max}) from prompted input'
                )
    print(
        f'Scale: {args.scale_min}..{args.scale_max} '
        f'({scale.n_points} bins)'
        )
    print(f'Leeway: {args.leeway:g}')
    if args.final_data:
        print('Run mode: final-data (train=train+dev, predict=test)')
    elif args.folds:
        print(
            f'Folds: {" ".join(str(fold_i) for fold_i in args.folds)} '
            f'({len(splits)} split(s))'
            )
    else:
        print(f'CV mode: {args.cv_mode} ({len(splits)} split(s))')
    print(f'Languages: {" ".join(selected_langs)}')
    print(
        'Adapter scope: '
        f'{"all-in-one" if args.all_in_one else "per-language"}'
        )
    print(f'Epochs: {args.epochs}, batch_size: {args.batch_size}')
    print(
        f'Weight decay: {args.weight_decay:g}, '
        f'warmup ratio: {args.warmup_ratio:g}'
        )
    print(f'Curriculum: {args.curriculum}')
    if (
            args.head_type == 'token' and
            args.loss_type == 'ce' and
            args.reweight != 'no'
            ):
        print(
            'Warning: --reweight/--rw affects only soft/prob losses '
            '(raft, ce_prob, delta2); it is ignored for --loss-type=ce.'
            )
    if args.final_data:
        print(
            'Predictions dir: '
            f'{Path(args.predictions_dir) / "test"}'
            )
    else:
        print(f'Predictions dir: {args.predictions_dir}')
    print(
        'Overwrite predictions: '
        f'{"yes" if args.overwrite_predictions else "no"}'
        )
    print(
        'Predict train split: '
        f'{"yes" if args.predict_train else "no"}'
        )
    if args.mode == 'predict' and args.predict_checkpoint:
        print(f'Predict checkpoint selector: {args.predict_checkpoint}')

    rows = []
    eval_split_label = 'test' if args.final_data else 'dev'
    for run_idx, (fold_i, split) in enumerate(splits, 1):
        fold_start = time.perf_counter()
        train_idx = split.train
        dev_idx = split.dev
        fold_dev_predictions = {}
        fold_train_predictions = {}
        if args.final_data:
            print(
                f'\nFinal-data run ({run_idx}/{len(splits)} selected): '
                f'train={len(train_idx)} eval={len(dev_idx)} '
                '(test evaluation split)'
                )
        else:
            print(
                f'\nFold {fold_i} ({run_idx}/{len(splits)} selected): '
                f'train={len(train_idx)} dev={len(dev_idx)}'
                )

        if args.all_in_one:
            print(
                f'  Adapter scope: all (languages={" ".join(selected_langs)})'
                )
            fold_scope_out = train_one_fold_scope(
                args,
                data=data,
                train_idx=train_idx,
                dev_idx=dev_idx,
                fold_num=fold_i,
                languages=selected_langs,
                prompt=args.prompt,
                warn_on_truncation=True,
                )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.vram_stats:
                print_vram_stats('after cleanup')

            for lang in selected_langs:
                dev_out = fold_scope_out['dev_outputs_by_lang'][lang]
                dev_metrics = dev_out['metrics']
                fold_dev_predictions[lang] = pd.DataFrame(
                    {
                        ID_COL: dev_out['item_ids'],
                        'prediction': dev_out['predictions'],
                        }
                    )
                if args.predict_train:
                    train_out = fold_scope_out['train_outputs_by_lang'][lang]
                    fold_train_predictions[lang] = pd.DataFrame(
                        {
                            ID_COL: train_out['item_ids'],
                            'prediction': train_out['predictions'],
                            }
                        )
                rows.append(
                    {
                        'fold': fold_i,
                        'lang': lang,
                        'row_type': 'final',
                        'adapter_scope': 'all',
                        'checkpoint_epoch': np.nan,
                        'checkpoint_reason': None,
                        **dev_metrics,
                        }
                    )
                print(
                    f'    {lang} {eval_split_label}   rmse={dev_metrics["rmse"]:.4f} '
                    f'r2={dev_metrics["r2"]:.4f} '
                    f'pearson={dev_metrics["pearson"]:.4f} '
                    f'spearman={dev_metrics["spearman"]:.4f} '
                    f'train_loss={dev_metrics["train_loss"]:.4f} '
                    f'train_time_sec={dev_metrics["train_time_sec"]:.1f}'
                    )
                if args.predict_train:
                    train_metrics = (
                        fold_scope_out['train_outputs_by_lang'][lang]['metrics']
                        )
                    print(
                        f'    {lang} train rmse={train_metrics["rmse"]:.4f} '
                        f'r2={train_metrics["r2"]:.4f} '
                        f'pearson={train_metrics["pearson"]:.4f} '
                        f'spearman={train_metrics["spearman"]:.4f}'
                        )
            for ckpt_metrics in fold_scope_out['checkpoint_metric_rows']:
                rows.append(
                    {
                        'fold': fold_i,
                        'lang': ckpt_metrics.get('lang'),
                        'row_type': 'checkpoint',
                        'adapter_scope': 'all',
                        **{
                            k: v for k, v in ckpt_metrics.items()
                            if k != 'lang'
                            },
                        # Checkpoint rows don't represent full fold runtime.
                        'train_time_sec': np.nan,
                        'train_loss': np.nan,
                        }
                    )
        else:
            for lang in selected_langs:
                print(f'  Language: {lang}')
                fold_scope_out = train_one_fold_scope(
                    args,
                    data=data,
                    train_idx=train_idx,
                    dev_idx=dev_idx,
                    fold_num=fold_i,
                    languages=[lang],
                    prompt=args.prompt,
                    warn_on_truncation=True,
                    )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if args.vram_stats:
                    print_vram_stats('after cleanup')
                dev_out = fold_scope_out['dev_outputs_by_lang'][lang]
                dev_metrics = dev_out['metrics']
                fold_dev_predictions[lang] = pd.DataFrame(
                    {
                        ID_COL: dev_out['item_ids'],
                        'prediction': dev_out['predictions'],
                        }
                    )
                if args.predict_train:
                    train_out = fold_scope_out['train_outputs_by_lang'][lang]
                    fold_train_predictions[lang] = pd.DataFrame(
                        {
                            ID_COL: train_out['item_ids'],
                            'prediction': train_out['predictions'],
                            }
                        )
                rows.append(
                    {
                        'fold': fold_i,
                        'lang': lang,
                        'row_type': 'final',
                        'adapter_scope': 'lang',
                        'checkpoint_epoch': np.nan,
                        'checkpoint_reason': None,
                        **dev_metrics,
                        }
                    )
                for ckpt_metrics in fold_scope_out['checkpoint_metric_rows']:
                    rows.append(
                        {
                            'fold': fold_i,
                            'lang': lang,
                            'row_type': 'checkpoint',
                            'adapter_scope': 'lang',
                            **ckpt_metrics,
                            # Checkpoint rows don't represent full fold runtime.
                            'train_time_sec': np.nan,
                            'train_loss': np.nan,
                            }
                        )
                print(
                    f'    {eval_split_label}   rmse={dev_metrics["rmse"]:.4f} '
                    f'r2={dev_metrics["r2"]:.4f} '
                    f'pearson={dev_metrics["pearson"]:.4f} '
                    f'spearman={dev_metrics["spearman"]:.4f} '
                    f'train_loss={dev_metrics["train_loss"]:.4f} '
                    f'train_time_sec={dev_metrics["train_time_sec"]:.1f}'
                    )
                if args.predict_train:
                    train_metrics = train_out['metrics']
                    print(
                        f'    train rmse={train_metrics["rmse"]:.4f} '
                        f'r2={train_metrics["r2"]:.4f} '
                        f'pearson={train_metrics["pearson"]:.4f} '
                        f'spearman={train_metrics["spearman"]:.4f}'
                        )

        if args.final_data:
            save_final_data_predictions(args, fold_dev_predictions)
        else:
            save_fold_predictions(
                args,
                fold_i,
                total_folds,
                fold_dev_predictions,
                split_name='dev',
                )
            if args.predict_train:
                save_fold_predictions(
                    args,
                    fold_i,
                    total_folds,
                    fold_train_predictions,
                    split_name='train',
                    )
        fold_elapsed = time.perf_counter() - fold_start
        if args.final_data:
            print(f'  Final-data elapsed_sec={fold_elapsed:.1f}')
        else:
            print(f'  Fold elapsed_sec={fold_elapsed:.1f}')

    results_df = pd.DataFrame(rows)
    total_elapsed_sec = time.perf_counter() - run_start
    results_df['total_elapsed_sec'] = float(total_elapsed_sec)
    final_results_df = (
        results_df[results_df['row_type'] == 'final'].copy()
        if 'row_type' in results_df.columns
        else results_df
        )

    if len(splits) > 1:
        print_results(final_results_df)
    elif args.final_data:
        print('\nFinal-data run: reporting test predictions.')
    else:
        print('\nSingle-fold run: skipping cross-fold aggregate summary.')

    if args.results_path:
        out_path = Path(maybe_suffix_output_path(args.results_path, args.folds))
    else:
        out_path = (
            Path('results') /
            'finetuned_llm' /
            f'{run_name(args)}.csv'
            )
        if args.folds:
            out_path = path_with_suffix_inserted(
                out_path,
                fold_selection_suffix(args.folds),
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')
    print(f'Total elapsed_sec={total_elapsed_sec:.1f}')


if __name__ == '__main__':
    args = parse_args()
    if args.stdout_file:
        stdout_path = Path(maybe_suffix_output_path(args.stdout_file, args.folds))
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open('w', encoding='utf-8') as f:
            with redirect_stdout(f):
                main(args)
    else:
        with nullcontext():
            main(args)
