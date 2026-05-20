import csv
import pandas as pd
import numpy as np

from util import download_if_necessary


COCA_COL2TOTAL = {
    # based on https://www.english-corpora.org/coca/
    'freq': 1_001_610_938,
    'TVM': 128_013_334,
    'spok': 127_396_916,
    'range': 485_202
    }


def get_frequencies_and_missing(
    tubelex: str | None = None,
    opensubtitles: str | None = None,
    coca: bool = False,
    path: str | None = None,
    log_smooth: bool = False,
    min_count: int = 0,
    column: str = 'count'
    ):
    if tubelex is not None:
        assert (opensubtitles is None) and (path is None) and (not coca)
        path = f'data/downloads/tubelex-{tubelex}.tsv.xz'
        download_if_necessary(
            url=(f'https://github.com/naist-nlp/tubelex/raw/'
                 f'7cb5fb36add76b83a266d1967536e1a1d3faa513/'
                 f'frequencies/tubelex-{tubelex}.tsv.xz'),
            path=path
            )
    elif opensubtitles is not None:
        assert (tubelex is None) and (path is None) and (not coca)
        path = f'data/downloads/os_{opensubtitles}.tsv.xz'
        download_if_necessary(
            url=(f'https://github.com/naist-nlp/tubelex/raw/'
                 f'7cb5fb36add76b83a266d1967536e1a1d3faa513/'
                 f'data/os_{opensubtitles}.tsv.xz'),
            path=path
            )
    elif coca:
        assert (tubelex is None) and (path is None) and (opensubtitles is None)
        path = 'data/downloads/COCA_WordFrequency.csv'
        download_if_necessary(
            url=('https://github.com/brucewlee/COCA-WordFrequency/raw/'
                 'refs/heads/main/COCA_WordFrequency.csv'),
            path=path
            )
    else:
        assert path is not None
        assert (opensubtitles is None) and (tubelex is None)

    assert path is not None

    if coca:
        freq = pd.read_csv(
            path, index_col='lemma',
            na_filter=False  # read "as is"
            ).groupby(level=0).sum()                # TODO we ignore PoS
    else:
        freq = pd.read_table(
            path, index_col='word',
            quoting=csv.QUOTE_NONE, na_filter=False  # read "as is"
            )

    if '[TOTAL]' in freq.index:
        freq_total = freq.loc['[TOTAL]', column]
        freq.drop('[TOTAL]', inplace=True)
    elif coca:
        freq_total = COCA_COL2TOTAL[column]
    else:
        freq_total = freq[column].sum()

    if min_count > 1:
        freq = freq[freq[column] >= min_count]

    if log_smooth:
        if column in {'videos', 'channels', 'range'}:
            log_freq = np.log10((freq[column] + 1) / (freq_total + 1))
            missing_log_freq = np.log10(1 / (freq_total + 1))
        else:
            log_freq = np.log10((freq[column] + 1) / (freq_total + len(freq)))
            missing_log_freq = np.log10(1 / (freq_total + len(freq)))

        return (log_freq, missing_log_freq)
    else:
        return ((freq[column] / freq_total), 0)
