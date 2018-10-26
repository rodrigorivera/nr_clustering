from sklearn.linear_model import Ridge


def ridge(alpha=1.0, normalize=True, random_state=None):
    return Ridge(alpha=alpha,normalize=normalize,random_state=random_state)
