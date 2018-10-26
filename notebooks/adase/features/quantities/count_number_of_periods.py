import pandas as pd
from typing import List


def count_number_of_periods(df: pd.DataFrame,
                            target: str,
                            group_condition: List[str]
                            ) -> pd.DataFrame:

    df_temp: pd.DataFrame = df.copy()
    new_df_temp_reduced = df_temp.groupby(group_condition)[
        target].agg('count').reset_index().rename(columns={target: 'rpd_count'})

    df_temp = df_temp.merge(new_df_temp_reduced,
                            how='left',
                            on=group_condition,
                            validate='many_to_one',
                            suffixes=['', '_r'])


    return df_temp