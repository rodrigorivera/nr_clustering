import pandas as pd
from typing import List


def create_cumulative_target(df: pd.DataFrame,
                             targets: List[str],
                             group_conditions: List[str],
                             ff_ranges: List[List[int]]
                             ) -> pd.DataFrame:
    '''

    :param df: pd.DataFrame
    :param ff_ranges: [[1, 2, 3], [2, 3], [3]]
    :param target: str
    :param conditions: List[str]
    :return: pd.DataFrame

    It creates 3 new columns called quantity_cumulative_Xffs where the X is a running index.
    The add up the quantities from other future flags. Thus, providing an idea of the build-up
    in quantity.

    '''

    df_temp: pd.DataFrame = df.copy()
    len_ff_ranges = len(ff_ranges)

    for target in targets:
        for idx, ff_range in enumerate(ff_ranges):
            condition = (df_temp['future_flag'].isin(ff_range))
            df_temp_reduced = df_temp[condition]
            col_idx = len_ff_ranges - idx
            col_name = target + '_cumulative_{}ffs'.format(col_idx)

            df_temp_reduced = df_temp_reduced.groupby(group_conditions)[target]\
                .agg(sum)\
                .reset_index()\
                .rename(columns={target: col_name})

            df_temp = df_temp.merge(df_temp_reduced,
                                    how='left',
                                    on=group_conditions,
                                    validate='many_to_one',
                                    suffixes=['', '_r'])

    return df_temp
