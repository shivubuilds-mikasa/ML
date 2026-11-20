from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mse = float(mean_squared_error(actual, predicted))
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(actual, predicted)),
    }
