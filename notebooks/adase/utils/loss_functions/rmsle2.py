import numpy as np


def rmsle2(true, predicted):
    # assert len(true) == len(predicted)
    log_diff = np.log1p(predicted) - np.log1p(true)
    squared = np.power(np.square(log_diff), 0.5)
    # rmsle = np.square(np.log(y_pred + 1dataset) - np.log(y_true + 1dataset)).mean() ** 0.5
    return np.mean(squared)