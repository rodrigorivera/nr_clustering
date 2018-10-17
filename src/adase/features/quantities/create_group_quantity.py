import pandas as pd
from typing import List


def create_group_quantity(df: pd.DataFrame,
                          target: str,
                          group_conditions: List[str]
                          ) -> pd.DataFrame:

    df_temp: pd.DataFrame = df.copy()
    #target_stripped = target.replace('original_', '')
    title = '{0}_{1}'.format(target, 'group')
    df_temp[title] = df_temp\
        .groupby(group_conditions)[target]\
        .transform('sum')\
        .fillna(0)

    return df_temp
