from sklearn.svm import SVR


def svr(C=1.0, epsilon=0.2):
    return SVR(C=C, epsilon=epsilon)
