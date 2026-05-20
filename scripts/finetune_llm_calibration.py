import numpy as np
from sklearn.linear_model import LinearRegression


def apply_calibration(values, calibration):
    values = np.asarray(values, dtype=float)
    if calibration is None:
        return values
    return (
        (values * float(calibration['coef'])) +
        float(calibration['intercept'])
        )


def fit_calibration(y_pred_active, y_true_active):
    x = np.asarray(y_pred_active, dtype=float).reshape(-1, 1)
    y = np.asarray(y_true_active, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            'Calibration fit requires equal-length predictions and '
            f'targets, got {x.shape[0]} and {y.shape[0]}'
            )
    if x.shape[0] == 0:
        raise ValueError('Calibration fit requires at least one sample')
    model = LinearRegression()
    model.fit(x, y)
    return {
        'coef': float(model.coef_[0]),
        'intercept': float(model.intercept_),
        }
