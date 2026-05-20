import pandas as pd
from sklearn.linear_model import LinearRegression

from frequencies import get_frequencies_and_missing


_TUBELEX_LOG_FREQ = None
_TUBELEX_MISSING_LOG_FREQ = None


def get_tubelex_log_frequency():
    global _TUBELEX_LOG_FREQ, _TUBELEX_MISSING_LOG_FREQ
    if _TUBELEX_LOG_FREQ is not None:
        return _TUBELEX_LOG_FREQ, _TUBELEX_MISSING_LOG_FREQ
    (_TUBELEX_LOG_FREQ, _TUBELEX_MISSING_LOG_FREQ) = get_frequencies_and_missing(
        tubelex='en',
        log_smooth=True,
        column='count'
        )
    return _TUBELEX_LOG_FREQ, _TUBELEX_MISSING_LOG_FREQ


def tubelex_lr_error(data: pd.DataFrame, l1: str) -> pd.Series:
    log_freq, missing_log_freq = get_tubelex_log_frequency()
    x = log_freq.reindex(data['en_target_word'].str.lower()).fillna(missing_log_freq)
    y = data[f'{l1}_GLMM_score'].astype(float)
    assert x.notna().all()
    assert y.notna().all()
    x_fit = x.to_numpy().reshape(-1, 1)
    y_fit = y.to_numpy()
    model = LinearRegression().fit(x_fit, y_fit)
    y_pred = model.predict(x_fit)
    return y_pred - y_fit
