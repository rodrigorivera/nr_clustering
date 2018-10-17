import numpy as np

# SMAPE as Kaggle calculates it
def kaggle_smape(true, predicted):
    true_o = true
    pred_o = predicted
    summ = np.abs(true_o) + np.abs(pred_o)
    smape = np.where(summ == 0, 0, np.abs(pred_o - true_o) / summ)
    return np.mean(smape)