from urllib.request import urlretrieve
import os
import sys
import argparse
import shlex
from typing import Optional, Iterable
from nltk.metrics.distance import edit_distance
import unicodedata
import re
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error, r2_score, make_scorer
from sklearn.base import BaseEstimator, RegressorMixin, clone as sklearn_clone
from scipy.special import logit


def _mkdir_parent(path: str) -> None:
    parent, __ = os.path.split(path)
    if parent:  # may be ''
        os.makedirs(parent, exist_ok=True)


def download_if_necessary(
    url: str,
    path: Optional[str] = None     # None -> return random (temporary) filename
    ) -> Optional[str]:
    if path is not None:
        if os.path.exists(path):
            return None
        _mkdir_parent(path)
    sys.stderr.write(f'Downloading data from "{url}"...\n')
    urlretrieve(url, filename=path)
    sys.stderr.write(f'Finished download to "{path}".\n')
    return path


def normalized_edit_distance(a: str, b: str) -> float:
    return edit_distance(a, b) / max(len(a), len(b))


def remove_accents(text: str) -> str:
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(
        char for char in normalized
        if unicodedata.category(char) != 'Mn'
        )


ONE_THIRD = 1 / 3
SIMILARITY_TRANSFORMS = ['cutoff', 'square', 'cube', 'no']
# Cutoff at 1/3 and squaring have a similar effect.


def similarity_to_l1(en_w: str, l1_w: str, exact=False, transform=None) -> float:
    # Remove accents, capitalization:
    en_w = remove_accents(en_w.lower())
    l1_w = remove_accents(l1_w.lower())

    # Remove parenthesized/bracketed from L1:
    l1_w = re.sub(r'\([^()]+\)|\[[^\[\]]+\]', '', l1_w)
    # Split, remove blanks:
    l1_ws = [re.sub(r' +', '', w).strip() for w in re.split('[,;/]', l1_w)]

    if exact:
        # Exact match found:
        return float(en_w in l1_ws)

    # Lowest distnace:
    d = min(normalized_edit_distance(en_w, w) for w in l1_ws)

    if transform == 'no' or not transform:
        s = 1 - d
    elif transform == 'cutoff':
        # cut off values < ONE_THIRD,
        s = (max(1 - d, ONE_THIRD) - ONE_THIRD) / (1 - ONE_THIRD)
    elif transform == 'square':
        s = (1 - d) ** 2
    elif transform == 'cube':
        s = (1 - d) ** 3

    assert 0 <= s <= 1, s

    return s


def bin_to_midpoints(
    x: pd.Series, n_bins: int,
    min: float | None = None,
    max: float | None = None,
    include_outside: bool = False
    ) -> pd.Series:
    if min is None:
        min = x.min()
    if max is None:
        max = x.max()
    bins = np.linspace(min, max, n_bins + 1)
    midpoints = (bins[:-1] + bins[1:]) / 2
    if include_outside:
        # Add outer real values to the bins
        bins[0] = -np.inf
        bins[-1] = +np.inf
    return pd.cut(x, bins=bins, labels=midpoints, include_lowest=True).astype(float)


class ClippedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = sklearn_clone(self.estimator)
        self.estimator_.fit(X, y)
        self.min_value = y.min()
        self.max_value = y.max()
        return self

    def predict(self, X):
        y_pred = self.estimator_.predict(X)
        return np.clip(y_pred, self.min_value, self.max_value)


class BinnedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, n_bins):
        self.estimator = estimator
        self.n_bins = n_bins

    def fit(self, X, y):
        self.estimator_ = sklearn_clone(self.estimator)
        self.estimator_.fit(X, y)
        self.min_value = y.min()
        self.max_value = y.max()
        return self

    def predict(self, X):
        y_pred = self.estimator_.predict(X)
        return bin_to_midpoints(
            y_pred,
            n_bins=self.n_bins, min=self.min_value, max=self.max_value,
            include_outside=True
            )


class PassthroughRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        # ignore X, y
        return self

    def predict(self, X):
        # Pass through a single feature, or average multiple features row-wise.
        assert (len(X.shape)) == 2 and (X.shape[-1] >= 1), X.shape
        if X.shape[-1] == 1:
            return X.squeeze()
        return np.asarray(X.mean(axis=1))


def pearson(target, pred):
    return pearsonr(target, pred).statistic


def spearman(target, pred):
    return spearmanr(target, pred).statistic


METRICS = {
    'rmse': root_mean_squared_error,
    'r2': r2_score,
    'pearson': pearson,
    'spearman': spearman
    }

METRIC_IS_SCORE = {
    'rmse': False,
    'r2': True,
    'pearson': True,
    'spearman': True
    }


def metric2scorer(metric: str):
    return make_scorer(METRICS[metric], greater_is_better=METRIC_IS_SCORE[metric])


def int_or_none(arg):
    if arg.lower() in {'no', 'none'}:
        return None
    try:
        return int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'Expected an integer or "no"/"none", got "{arg}"'
            )


def str_float_pair(arg, sep='=') -> tuple[str, float]:
    try:
        key, value = arg.split(sep)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'Expected form: key{sep}value, got: "{arg}".'
            )
    try:
        return (key, float(value))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'Expected a float value, got "{value}" in "{arg}".'
            )


def cli_command(argv):
    return 'python ' + ' '.join(shlex.quote(a) for a in argv)


def final_predict_command(argv, final_model_name):
    out = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == '--final-train':
            i += 2
            continue
        if token.startswith('--final-train='):
            i += 1
            continue
        if token == '--final-train-folds':
            i += 1
            while i < len(argv) and (not argv[i].startswith('-')):
                i += 1
            continue
        if token.startswith('--final-train-folds='):
            i += 1
            continue
        out.append(token)
        i += 1

    out.extend(['--final-predict', final_model_name])
    has_track = any(
        t == '--track' or t.startswith('--track=')
        for t in out
        )
    has_no_save = any(t == '--no-save' for t in out)
    if (not has_track) and (not has_no_save):
        out.extend(['--track', '<open|closed>'])
    return cli_command(out)


class OptionalLogit:
    choices = [
        'transliteration', 'tl',
        'difficulty', 'D',
        'trickiness', 'T',
        'pronounce-difficulty', 'pd',
        'calque', 'cq',
        'lexical-ambiguity', 'la',
        'all'
        ]
    variables = {
        'transliteration',
        'difficulty',
        'trickiness',
        'pronounce_difficulty',
        'calque',
        'lexical_ambiguity'
        }

    def __init__(self, apply_to: Iterable[str], eps=0.0001):
        applies_to = set()
        for c in apply_to:
            match c:
                case 'transliteration' | 'tl':
                    applies_to.add('transliteration')
                case 'difficulty' | 'D':
                    applies_to.add('difficulty')
                case 'trickiness' | 'T':
                    applies_to.add('trickiness')
                case 'pronounce-difficulty' | 'pd':
                    applies_to.add('pronounce_difficulty')
                case 'calque' | 'cq':
                    applies_to.add('calque')
                case 'lexical-ambiguity' | 'la':
                    applies_to.add('lexical_ambiguity')
                case 'all':
                    applies_to = OptionalLogit.variables
                case _:
                    raise Exception(f'Not implemented: {c}')

        self.applies_to = applies_to
        self.eps = eps

    def logit(self, data):
        eps = self.eps
        return logit(np.clip(data, eps, 1 - eps))

    def __getattr__(self, variable):
        if variable not in OptionalLogit.variables:
            raise AttributeError(f'Unsupported variable name: {variable}')
        return self.logit if (variable in self.applies_to) else (lambda x: x)
