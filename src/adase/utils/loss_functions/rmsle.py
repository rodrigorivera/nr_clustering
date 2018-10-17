import numpy as np


def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.square(np.log(np.array(y_pred) + 1) - np.log(np.array(y_true) + 1)).mean() ** 0.5