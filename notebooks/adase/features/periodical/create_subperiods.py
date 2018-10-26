import pandas as pd
import numpy as np
from typing import Dict


def create_subperiods(
        df: pd.DataFrame,
        periods: Dict,
        col_delivery_date: str = 'rpd'
) -> pd.DataFrame:
    '''

    :param df: pd.DataFrame
    :param periods: Dict={'month': 12, '3months': 3, '6months': 6}
    :param col_delivery_date: str
    :return: pd.DataFrame

    It creates 3 new columns, each with the respective year, quarter or semester
    '''
    df_temp: pd.DataFrame = df.copy()

    for key, value in periods.items():
        df_temp[key] = df_temp\
            .apply(lambda row: int(row[col_delivery_date] % value), axis=1)

        df_temp[key] = np.where(
            (df_temp[key]) < 1,
            (df_temp[key] + value),
            df_temp[key]
        )

    return df_temp
