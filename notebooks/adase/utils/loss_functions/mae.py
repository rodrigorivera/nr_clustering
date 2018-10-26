import numpy as np

# MAE on log1p
def mae(true, predicted):
    true_o = np.log1p(true)
    pred_o = np.log1p(predicted)
    error = np.abs(true_o - pred_o) / 2
    return np.mean(error)