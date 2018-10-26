from sklearn.linear_model import Lasso


def lasso(fit_intercept=True, normalize=True):
    return Lasso(fit_intercept=fit_intercept, normalize=normalize)
