from sklearn.ensemble import AdaBoostRegressor


def adaboost(n_estimators=5000, random_state=42, learning_rate=0.01):
    return AdaBoostRegressor(n_estimators=n_estimators, random_state=random_state, learning_rate=learning_rate)
