from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import pandas as pd
import gc
import sys
import numpy as np
import logging
from typing import Dict

from ..models import arima_auto, arimax


def _matrix_models():
    return None


def _arima_models(df_train: pd.DataFrame,
                  target: str,
                  model_name: str):

    y_train = df_train[[x for x in df_train.columns.values if x == target]]

    if model_name == 'auto_arima':
        trained_model = arima_auto(y_train)
        del y_train

        return trained_model

    else:

        model = arimax(y_train, target)
        trained_model = model.fit("MLE")
        del y_train

        return model


def _regression_models(df_train: pd.DataFrame, future_flag, target, model):

    whitelist = [x for x in df_train.columns.values if
                 ('original_quantity_encode' not in x) &
                 ('original_quantity_type' not in x) &
                 ('original_quantity_siblings' not in x) &
                 ('original_quantity_parents' not in x) &
                 ('original_quantity_children' not in x) &
                 ('original_quantity_cluster' not in x) &
                 ('original_quantity_group_encode' not in x) &
                 ('original_quantity_group_type' not in x) &
                 ('original_quantity_group_siblings' not in x) &
                 ('original_quantity_group_parents' not in x) &
                 ('original_quantitys_group_children' not in x) &
                 ('original_quantity_group_cluster' not in x) &
                 ('original_quantity_group' not in x) &
                 ('type' not in x) &
                 ('item_code' not in x) &
                 ('future_flag' not in x) &
                 ('rpd' not in x) &
                 ('category_code' not in x) &
                 ('cluster_code' not in x)
                 ]

    if future_flag == 1:
        whitelist = [x for x in whitelist if 'original_quantity_cumulative_3' not in x]

    elif future_flag == 2:
        whitelist = [x for x in whitelist if
                     ('original_quantity_cumulative_3' not in x) &
                     ('original_quantity_cumulative_2' not in x) &
                     ('original_quantity_1shift_cumulative_1' not in x)

                     ]

    elif future_flag == 3:
        whitelist = [x for x in whitelist if
                     ('original_quantity_cumulative_' not in x) &
                     ('original_quantity_1shift_cumulative' not in x)
                     ]

    categorical_columns = [
        # 'item_code', 'future_flag', 'rpd', 'category_code', 'type', 'cluster_code',
        'rpd_since_max', 'rpd_since_max_encode',
        'rpd_since_max_type',
        'rpd_since_max_cluster', 'item_code_parent',
        'rpd_since_max_siblings', 'item_code_child',
        'rpd_since_max_parents',
        'rpd_since_max_children', 'rpd_0quantity',
        'is_rpd37_0qty', 'rpd_creation', 'month',
        '3months', '6months', 'quarter', 'semester',
        'year', 'is_beginning_quarter', 'is_end_quarter',
        'is_beginning_semester', 'is_end_semester',
        'is_beginning_year', 'is_end_year', 'is_same_creation_quarter',
        'is_same_creation_semester', 'is_same_creation_year', 'is_zero_quantity',
    ]

    signed_columns = [
        'original_quantity_1shift_diff',
        'original_quantity_1shift_pct_change',
        'original_quantity_1shift_group_encode',
        'original_quantity_1shift_diff_encode',
        'original_quantity_1shift_pct_change_encode',
        'original_quantity_1shift_increase_encode',
        'original_quantity_1shift_group_type',
        'original_quantity_1shift_diff_type',
        'original_quantity_1shift_pct_change_type',
        'original_quantity_1shift_increase_type',
        'original_quantity_1shift_group_cluster',
        'original_quantity_1shift_diff_cluster',
        'original_quantity_1shift_pct_change_cluster',
        'original_quantity_1shift_increase_cluster',
        'original_quantity_group_siblings',
        'original_quantity_1shift_group_siblings',
        'original_quantity_1shift_diff_siblings',
        'original_quantity_1shift_pct_change_siblings',
        'original_quantity_1shift_diff_parents',
        'original_quantity_1shift_pct_change_parents',
        'original_quantity_1shift_diff_children',
        'original_quantity_1shift_pct_change_children',

        'change_quantity_1shift_cluster',
        'change_quantity_1shift_type',
        'change_quantity_1shift_encode',
        'change_quantity_1shift',
        'change_quantity_1shift_children',
        'change_quantity_1shift_parents',
        'change_quantity_1shift_siblings'
    ]

    numeric_columns = [col for col in df_train.columns
                       if col not in categorical_columns]

    positive_columns = [col for col in numeric_columns
                        if col not in signed_columns]

    transformer = Pipeline([
        ("log1p", FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)),
        ("scale", StandardScaler())
    ])

    col_transformer = ColumnTransformer([
        ("positive", transformer, positive_columns),
        ("negative", StandardScaler(), signed_columns),
    ], remainder="passthrough")

    # logger.info('{0} - {1} - {2} - {3}'.format(df_train.shape, target, model, model_name))

    # def make_pipeline(estimator):
    #    return Pipeline([
    #        ("preproc", clone(col_transformer)),
    #        ("model", TransformedTargetRegressor(
    #            clone(estimator),
    #            transformer=clone(col_transformer)))
    #    ])

    def make_pipeline(estimator):
        return Pipeline([
            # ("preproc", clone(transformer)),
            ("model", TransformedTargetRegressor(
                clone(estimator),
                transformer=clone(transformer)))
        ])

    x_train = df_train[[x for x in whitelist if x != target]].copy()
    y_train = df_train[[x for x in whitelist if x == target]].copy()

    # logger.info('(x: {0} -- y: {1})'.format(x_train.shape, y_train.shape))
    if np.isinf(x_train).any().any() or np.isinf(y_train).any().any():
        sys.exit("STOPPING PROCESS!")

    assert np.all(np.isfinite(x_train)), """Bad `X` DATA!"""
    assert np.all(np.isfinite(y_train)), """Bad `y` DATA!"""

    fitted_model = make_pipeline(model).fit(x_train, y_train)

    del x_train
    del y_train

    return fitted_model


def train_dataset(
        df_train: pd.DataFrame,
        target: str,
        model,
        model_name:str
)-> Dict:

    logger = logging.getLogger(__name__)

    dict_models = dict()
    dict_models[model_name] = {}

    #logger.info('ENTERED TRAIN MODEL : {0}'.format(dict_models.keys))
    list_items_code = df_train['item_code'].unique()
    df_temp = df_train.set_index(['item_code', 'future_flag']).sort_index()

    for item_code in list_items_code:
        for future_flag in df_temp.loc[item_code].index.unique().tolist():

            df_to_train = df_temp\
                .loc[[item_code, future_flag]]\
                .reset_index()\
                .copy()

            fitted_model = None

            if 'arima' in model_name:

                fitted_model = _arima_models(df_to_train, target, model)
                dict_models[model_name][future_flag, item_code] = fitted_model

            elif 'matrix' in model_name:

                pass

            else:

                #logger.info('NOT ARIMA')

                fitted_model = _regression_models(
                    df_to_train,
                    future_flag,
                    target,
                    model
                )

                dict_models[model_name][future_flag, item_code] = fitted_model
                #logger.info('fitted model len values: item code: {0}, ff: {1} -- {2}'.format(
                #    item_code, future_flag, len(dict_models.values())))

            #logger.info('fitted model: {0}'.format(fitted_model))
            del fitted_model
            gc.collect()

    #logger.info('fitted model: {}'.format({k: len(v) for k, v in dict_models.items()}))
    return dict_models