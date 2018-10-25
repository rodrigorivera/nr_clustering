import pandas as pd
from typing import List


def merge_dfs(df_left: pd.DataFrame,
              df_right: pd.DataFrame,
              merge_how: str,
              merge_left_on: List[str],
              merge_right_on: List[str],
              merge_validate: str,
              cols_sort_values: List[str]
              ) -> pd.DataFrame:

    df_temp_left: pd.DataFrame = df_left.copy()
    df_temp_right: pd.DataFrame = df_right.copy()
    df_temp_left.reset_index(inplace=True, drop=True)
    df_temp_right.reset_index(inplace=True, drop=True)

    df_temp: pd.DataFrame = pd.merge(
        df_temp_left,
        df_temp_right,
        how=merge_how,
        left_on=merge_left_on,
        right_on=merge_right_on,
        validate=merge_validate,
        suffixes=['', '_r']
    )

    df_temp = df_temp.sort_values(by=cols_sort_values, ascending=True)
    df_temp = df_temp.reset_index(drop=True)

    return df_temp
