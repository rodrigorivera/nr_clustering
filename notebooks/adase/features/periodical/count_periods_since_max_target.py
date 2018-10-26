import pandas as pd
from typing import List


def count_periods_since_max_target(df: pd.DataFrame,
                                   target: str,
                                   max_target: str,
                                   condition: List[str]
                                   ) -> pd.DataFrame:

    df_temp: pd.DataFrame = df.copy()
    df_temp_compact: pd.DataFrame = df_temp[[target, max_target] + condition]
    df_temp_compact['is_max'] = (df_temp_compact[target] == df_temp_compact[max_target])
    df_temp_compact['is_max_cumsum'] = df_temp_compact['is_max'].cumsum()
    df_temp_compact['rpd_since_max'] = df_temp_compact\
        .groupby(
        condition+['is_max_cumsum']
    )\
        .is_max\
        .apply(lambda x: (~x).cumsum())

    title = 'rpd_since_max'
    df_temp[title] = df_temp_compact['rpd_since_max']

    return df_temp

