import numpy as np


# SMAPE, rounded up to the closest integet
def rounded_smape(true, predicted):
    true_o = np.round(true).astype(np.int)
    pred_o = np.round(predicted).astype(np.int32)
    summ = np.abs(true_o) + np.abs(pred_o)
    smape = np.where(summ == 0, 0, np.abs(pred_o - true_o) / summ)
    return np.mean(smape)