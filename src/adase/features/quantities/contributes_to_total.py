import pandas as pd
from typing import List


def contributes_to_total(df: pd.DataFrame,
                         quantity: str,
                         total_quantity: str,
                         group_condition: List[str]
                         ) -> pd.DataFrame:

    df_temp: pd.DataFrame = df.copy()

    df_temp = df_temp.set_index(group_condition)

    df_temp['{}_contribution'.format(quantity)] = pd\
        .to_numeric(df_temp[quantity] / df_temp[total_quantity])\
        .fillna(0)

    df_temp['{}_contribution_median'.format(quantity)] = df_temp\
        .groupby(group_condition)['{}_contribution'.format(quantity)]\
        .transform(lambda x: x.median())
    df_temp = df_temp.reset_index()

    return df_temp
