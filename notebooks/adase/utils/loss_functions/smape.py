import numpy as np


#def smape(y_true, y_pred):
#    denominator = (np.abs(y_true) + np.abs(y_pred))
#    diff = np.abs(y_true - y_pred) * 200 / denominator
#    diff[denominator == 0] = 0.0
#    return np.mean(diff)

def smape(y_true, y_pred):
    weight = (np.abs(y_true) + np.abs(y_pred)) / 2
    output = np.divide(np.abs(y_true - y_pred), weight, where=weight > 0,
                       out=np.full_like(weight, np.nan))

    return output  # np.nanmean(output, axis=1)