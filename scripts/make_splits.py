import json
import argparse
import pandas as pd
from kvl import read_subset, SUBSETS, L1_CODES, ID_COL
from scipy.spatial.distance import cdist
import numpy as np
from numpy.linalg import norm

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold


def energy_distance(X, Y):
    '''
    Distance between discrete distributions.
    Like from scipy.stats.energy_distance, but supports multi-variate distributions,
    i.e. X, Y can be 2D arrays.
    '''
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Convert 1D arrays to (n, 1)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError('Inputs must be 1D or 2D arrays')

    dXY = cdist(X, Y)
    dXX = cdist(X, X)
    dYY = cdist(Y, Y)
    return np.sqrt((2 * dXY.mean() - dXX.mean() - dYY.mean()))


def mean_difference(X, Y, decimals=2):
    return np.round(np.abs(
        np.asarray(X).mean(axis=0) - np.asarray(Y).mean(axis=0)
        ), decimals=decimals)


SCORE_COLS = [f'{l1}_GLMM_score' for l1 in L1_CODES]


def main(args: argparse.Namespace):
    train, dev = (read_subset(subset) for subset in SUBSETS)
    assert len(dev) < len(train)
    data = pd.concat([train, dev])
    n_clusters = 125  # roughly 1.5 * np.sqrt(len(data))
    print(f'Clusters for stratification: {n_clusters}')
    print(f'Folds:                       {args.folds}')
    print()

    # [[ID_COL, *SCORE_COLS]]

    labels = KMeans(n_clusters, random_state=args.seed).fit_predict(data[SCORE_COLS])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    splits = [
        (data.iloc[train_idx], data.iloc[dev_idx])
        for train_idx, dev_idx in skf.split(data, labels)
        ]
    all_ids = pd.concat((dev_set[ID_COL] for _, dev_set in splits))

    # Check sets valid:
    assert len(all_ids) == len(data)          # Same number
    assert set(all_ids) == set(data[ID_COL])  # Same values

    print('Energy distances, mean differences')
    print('==================================')
    old_d = energy_distance(dev[SCORE_COLS], data[SCORE_COLS])
    old_md = mean_difference(dev[SCORE_COLS], data[SCORE_COLS])
    print(f'Old dev set vs. whole: {old_d:.3f}, {old_md}')
    print()

    max_d = 0
    max_md_norm = 0
    for i, (new_train, new_dev) in enumerate(splits, 1):
        d = energy_distance(new_dev[SCORE_COLS], data[SCORE_COLS])
        if d > max_d:
            max_d = d
        md = mean_difference(new_dev[SCORE_COLS], data[SCORE_COLS])
        md_norm = norm(d)
        if md_norm > max_md_norm:
            max_md_norm = md_norm
        print(f'New dev set #{i} vs. whole: {d:.3f}, {md}')
    print()

    if max_d > old_d:
        raise Exception('Booh, max. energy distance greater than in the old split.')

    if max_md_norm > norm(old_md):
        raise Exception('Booh, max. mean difference norm greater than than in the '
                        'old split.')

    print('Looks good compared to the old split!')
    print()

    if not args.output:
        print('Use --output to output the splits as JSON.')
        return

    print(f'Writing splits as [[train IDs, test IDs], ...] to {args.output}.')
    with open(args.output, 'w') as f:
        json.dump(
            [[list(train[ID_COL]), list(dev[ID_COL])] for train, dev in splits], f
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', '-k', default=5, type=int)
    parser.add_argument('--seed', '-s', default=42, type=int)
    parser.add_argument(
        '--output', '-o',
        help='Output splits as [[train IDs, test IDs], ...] JSON.'
        )
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
