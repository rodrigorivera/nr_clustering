import pandas as pd
import logging
from tqdm import tqdm
import numpy as np
import sys

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from typing import Tuple, Dict


def _matrix_models():
    return None


def _arima_models(
        df_test: pd.DataFrame,
        target: str,
        model_name: str,
        model
) -> pd.DataFrame:

    y_test = df_test[[x for x in df_test.columns.values if x == target]].reset_index(drop=True)
    item_code = df_test['item_code'].unique()[0]
    future_flag = df_test['future_flag'].unique()[0]
    rpd = df_test['rpd'].unique()
    original_quantity = y_test.values
    predictions = pd.DataFrame

    if model_name == 'auto_arima':
        list_df = []
        for val in range(0, len(y_test)):
            list_prediction = []
            prediction = model.predict(1)
            list_prediction = [item_code, future_flag, prediction[0]]

            list_df.append(list_prediction)

            model.add_new_observations([y_test.values[val]])

        predictions = pd.DataFrame.from_records(list_df, columns=['item_code', 'future_flag', 'predicted'])
        predictions['model'] = model_name
        predictions['actual'] = original_quantity
        predictions['rpd'] = rpd+37

    else:
        predictions = model \
            .predict(h=len(y_test.values), oos_data=y_test) \
            .rename(columns={'original_quantity': 'predicted'})

        predictions['item_code'] = item_code
        predictions['future_flag'] = future_flag
        predictions['model'] = model_name
        predictions['actual'] = original_quantity
        predictions['rpd'] = rpd+37

    return predictions


def _regression_models(
        df_test: pd.DataFrame,
        future_flag: int,
        target: str,
        model_name: str,
        fitted_model
) -> pd.DataFrame:

    df_test = df_test.reset_index(drop=True)

    whitelist = [x for x in df_test.columns.values if
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

    X_test = df_test[[x for x in whitelist if x != target]].copy()
    y_test = df_test[[x for x in whitelist if x == target]].copy()

    prediction = fitted_model.predict(X_test)

    y_test['predicted'] = prediction
    y_test['item_code'] = df_test['item_code'].unique()[0]
    y_test['future_flag'] = df_test['future_flag'].unique()[0]
    y_test['model'] = model_name
    y_test['rpd'] = df_test['rpd'].unique()
    y_test = y_test.rename(columns={'original_quantity': 'actual'})

    return y_test


def predict_dataset(
        df_test: pd.DataFrame,
        target: str,
        dict_model_fitted: Dict,
        model_name: str
) -> pd.DataFrame:

    logger = logging.getLogger(__name__)

    dict_models = dict()
    dict_models[model_name] = {}

    list_df = []

    for future_flag in df_test.future_flag.unique():
        for item_code in df_test.item_code.unique():

            ctest = (df_test['item_code'] == item_code) & \
                    (df_test['future_flag'] == future_flag)

            model = dict_model_fitted.get((future_flag, item_code))

            if (len(df_test[ctest].values) != 0) and model is not None:

                test_condition = (df_test['item_code'] == item_code) & (df_test['future_flag'] == future_flag)

                df_to_test = df_test[test_condition][:8].reset_index(drop=True).copy()

                if 'arima' in model_name:

                    results = _arima_models(df_to_test, target, model_name, model)
                    list_df.append(results)

                elif 'matrix' in model_name:

                    pass

                else:

                    results = _regression_models(
                        df_to_test,
                        future_flag,
                        target,
                        model_name,
                        model
                    )
                    list_df.append(results)

    predictions = pd.concat(list_df)

    return predictions
