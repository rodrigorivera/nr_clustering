import pandas as pd
from typing import List


def create_dummy_variables(df: pd.DataFrame,
                           cols: List
                           ) -> pd.DataFrame:
    '''

    :param df:
    :param cols:
    :return: pd.DataFrame

    It creates many columns, one for each category or entry within a list or series.

    '''

    data: pd.DataFrame = df.copy()
    df_cols = data[cols].copy()

    df_temp = pd.get_dummies(data, columns=cols, dummy_na=True)
    df_temp = pd.concat([df_temp, df_cols], axis=1, sort=False)

    return df_temp

