import pandas as pd
import numpy as np
from typing import Dict


def check_is_beginning_end(
        df: pd.DataFrame,
        periods: Dict,
        phase: Dict
) -> pd.DataFrame:

    '''

    :param df: pd.DataFrame
    :param periods: Dict = {'year': 12, 'semester': 6, 'quarter': 3}
    :param phase: Dict = {'beginning': 1, 'end': 0}
    :return:

    Verify if the date of an item_code is at the beginning or end of a year,
    semester or quarter and create a new column. Example:
    is_beginning_year or is_end_quarter each with a boolean.

    In total 6 new columns are added.
    '''

    df_temp: pd.DataFrame = df.copy()

    for key, value in periods.items():
        for key2, value2 in phase.items():
            condition = (df_temp['month'] % value) == value2
            col_title = 'is_{0}_{1}'.format(key2, key)
            df_temp[col_title] = np.where(condition, 1, 0)

    return df_temp

