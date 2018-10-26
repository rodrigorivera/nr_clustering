import pandas as pd
from typing import List


def count_period_with_zero_quantity(
        df: pd.DataFrame,
        target: str,
        group_condition: List[str],
        col_period: str,
        periods: int
) -> pd.DataFrame:

    df_temp: pd.DataFrame = df.copy()

    condition = (df_temp[col_period] < periods) &\
                (df_temp[target] == 0)

    new_df_temp_reduced = df_temp[condition]\
        .groupby(group_condition)[target]\
        .agg('count')\
        .fillna(0)\
        .reset_index()\
        .rename(columns={target: 'rpd_0quantity'})

    df_temp = df_temp.merge(new_df_temp_reduced,
                            how='left',
                            on=group_condition,
                            validate='many_to_one',
                            suffixes=['', '_r'])

    return df_temp
