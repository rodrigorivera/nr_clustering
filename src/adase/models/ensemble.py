import xgboost as xgb
import warnings
from sklearn.pipeline import make_pipeline, make_union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')


# Converts estimator to transform in order to ensemble many estimators
class EstimatorToTransform(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, *args):
        self.estimator.fit(X, *args)
        return self

    def transform(self, X):
        pred = self.estimator.predict(X)
        return pred.reshape(-1, 1)


def ensemble():
    pipeline_ensemble = make_pipeline(
        make_union(
            EstimatorToTransform(
                xgb.XGBRegressor(
                    booster='gbtree',
                    nthread=10,
                    max_depth=5,
                    eta=0.2,
                    silent=1,
                    subsample=0.7,
                    objective='reg:linear',
                    eval_metric='rmse',
                    colsample_bytree=0.7
                ),
            ),
            EstimatorToTransform(
                xgb.XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    objective='reg:linear',
                ),
            ),
            EstimatorToTransform(
                xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=1,
                    colsample_bytree=1,
                    objective='reg:linear',
                ),
            ),
        ),
    )

    xgb_pipeline = make_pipeline(
        pipeline_ensemble,
        LinearRegression()

    )

    return xgb_pipeline
