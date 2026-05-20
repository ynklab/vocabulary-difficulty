import numpy as np
from scipy.special import expit, logit
from typing import NamedTuple


class PointScale(NamedTuple):
    min: int
    max: int

    @property
    def n_points(self):
        return (self.max - self.min) + 1


def fit_bin_edges(train_scores, scale):
    n = scale.n_points
    quantiles = np.linspace(0.0, 1.0, n + 1)
    edges = np.quantile(np.asarray(train_scores, dtype=float), quantiles)
    edges = np.maximum.accumulate(edges)
    if np.unique(edges).size < (n + 1):
        lo = float(np.min(train_scores))
        hi = float(np.max(train_scores))
        if lo == hi:
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, n + 1)
    return edges


def scores_to_active_space(scores, space):
    scores = np.asarray(scores, dtype=float)
    if space == 'logit':
        return scores
    if space == 'probability':
        return expit(scores)
    raise ValueError(f'Unsupported space: {space}')


def scores_from_active_space(scores, space, eps=1e-6):
    scores = np.asarray(scores, dtype=float)
    if space == 'logit':
        return scores
    if space == 'probability':
        return logit(np.clip(scores, eps, 1.0 - eps))
    raise ValueError(f'Unsupported space: {space}')


def scores_to_bin_labels(scores, edges, scale):
    scores = np.asarray(scores, dtype=float)
    bin_idx = np.digitize(scores, bins=edges[1:-1], right=False)
    return scale.max - bin_idx


def bin_labels_to_scores(labels, edges, scale):
    labels = np.asarray(labels, dtype=int)
    bin_idx = scale.max - labels
    left = edges[bin_idx]
    right = edges[bin_idx + 1]
    return (left + right) / 2.0


def scores_to_soft_bin_probs(scores, edges, scale):
    scores = np.asarray(scores, dtype=float)
    n = scale.n_points
    probs = np.zeros((len(scores), n), dtype=float)

    for i, score in enumerate(scores):
        idx = np.digitize([score], bins=edges[1:-1], right=False)[0]
        cur_label = scale.max - idx
        probs[i, cur_label - scale.min] = 1.0

        if idx >= (n - 1):
            continue

        left = edges[idx]
        right = edges[idx + 1]
        width = right - left
        if width <= 0:
            continue

        ratio = float(np.clip((score - left) / width, 0.0, 1.0))
        next_label = cur_label - 1
        if next_label >= scale.min:
            probs[i, cur_label - scale.min] = 1.0 - ratio
            probs[i, next_label - scale.min] = ratio

    return probs


def soft_label_to_scores(pred_labels, edges, scale):
    pred_labels = np.asarray(pred_labels, dtype=float)
    labels = np.arange(scale.min, scale.max + 1)
    bin_centers = bin_labels_to_scores(labels, edges, scale)
    return np.interp(pred_labels, labels, bin_centers)


def fit_score_range(train_scores, leeway=0.0):
    scores = np.asarray(train_scores, dtype=float)
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi == lo:
        hi = lo + 1e-6
    if leeway < 0:
        raise ValueError(f'leeway must be >= 0, got {leeway}')
    span = hi - lo
    if span > 0.0 and leeway > 0.0:
        pad = span * (leeway / 2.0)
        lo -= pad
        hi += pad
    return lo, hi


def fit_uniform_score_points(train_scores, scale, leeway=0.0):
    lo, hi = fit_score_range(train_scores, leeway=leeway)
    # Label order is scale.min..scale.max (easy..hard), which corresponds
    # to descending score values in the current difficulty-oriented setup.
    return np.linspace(hi, lo, scale.n_points)


def scores_to_soft_label_probs(scores, label_score_points, scale):
    scores = np.asarray(scores, dtype=float)
    points = np.asarray(label_score_points, dtype=float)
    if points.shape != (scale.n_points,):
        raise ValueError(
            'label_score_points must have shape '
            f'({scale.n_points},), got {points.shape}'
            )

    # Points must be strictly decreasing in label order for interpolation.
    if not np.all(np.diff(points) < 0):
        raise ValueError('label_score_points must be strictly decreasing')

    probs = np.zeros((len(scores), scale.n_points), dtype=float)
    for i, score in enumerate(scores):
        if score >= points[0]:
            probs[i, 0] = 1.0
            continue
        if score <= points[-1]:
            probs[i, -1] = 1.0
            continue

        # Find j such that points[j] >= score >= points[j + 1].
        for j in range(scale.n_points - 1):
            hi = points[j]
            lo = points[j + 1]
            if lo <= score <= hi:
                width = hi - lo
                if width <= 0:
                    probs[i, j] = 1.0
                    break
                w_hi = (score - lo) / width
                w_lo = 1.0 - w_hi
                probs[i, j] = w_hi
                probs[i, j + 1] = w_lo
                break

    return probs
