import pandas as pd
import numpy as np
from typing import Dict


def check_if_same_period(
        df: pd.DataFrame,
        periods: Dict,
        col_delivery_date: str = 'rpd',
        col_creation_date: str = 'rpd_creation',
) -> pd.DataFrame:
    '''
    :param df: pd.DataFrame
    :param periods: Dict={'year': 12, 'semester': 6, 'quarter': 3}
    :param col_delivery_date: str = 'rpd'
    :param col_creation_date: str = 'rpd_creation'
    :return: pd.DataFrame

    It checks, if the creation period and the delivery date are within the same time frame (i.e., same
    quarter) and creates a new feature is_same_creation_quarter with boolean value
    '''

    df_temp: pd.DataFrame = df.copy()

    for key, value in periods.items():
        col_title = 'is_same_creation_{}'.format(key)
        condition = (np.ceil(df_temp[col_delivery_date] / value) == np.ceil(df_temp[col_creation_date] / value)) * 1
        df_temp[col_title] = condition

    return df_temp
