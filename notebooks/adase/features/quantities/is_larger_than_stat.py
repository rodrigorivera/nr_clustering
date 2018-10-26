import pandas as pd


def is_larger_than_stat(df: pd.DataFrame,
                        target: str
                        ) -> pd.DataFrame:

    df_temp: pd.DataFrame = df.copy()
    statistics = ['_cum_mean', '_cum_median']
    full_target = target
    title = full_target

    for stat in statistics:
        to_compare = full_target + stat
        if (to_compare in set(df_temp.columns.values)) &\
                (target in set(df_temp.columns.values)):
            df_temp[title + '_lg_than' + stat] = (df_temp[target] > df_temp[to_compare]) * 1

    return df_temp
