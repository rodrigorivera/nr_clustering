import pandas as pd
from typing import List
from functools import reduce


def create_rolling_statistics(df: pd.DataFrame,
                              targets: List[str],
                              condition: List[str],
                              width: int
                              ) -> pd.DataFrame:

    '''

    :param df:
    :param target:
    :param target_code_ff
    :param width_windows:
    :return:

    It creates for each target 7*6 new columns for each target with rolling statistics
    such as mean, min, max, std, sum and median for multiple time windows.

    '''

    data: pd.DataFrame = df.copy()

    width_adjust = width - 1

    cols_mean = {}
    cols_min = {}
    cols_max = {}
    cols_std = {}
    cols_sum = {}
    cols_median = {}
    for target in targets:
        cols_mean[target] = '{0}_mean{1}'.format(target, width)
        cols_min[target] = '{0}_min{1}'.format(target, width)
        cols_max[target] = '{0}_max{1}'.format(target, width)
        cols_std[target] = '{0}_std{1}'.format(target, width)
        cols_sum[target] = '{0}_sum{1}'.format(target, width)
        cols_median[target] = '{0}_median{1}'.format(target, width)

    to_use = data[condition+targets]
    df_grouped = to_use.groupby(condition)[targets]

    df_grouped_mean = df_grouped.transform(
        lambda x: x.shift(width_adjust).fillna(0).rolling(window=width).mean()
    )\
        .fillna(0)\
        .rename(columns=cols_mean)

    df_grouped_min = df_grouped.transform(
        lambda x: x.shift(width_adjust).fillna(0).rolling(window=width).min()
    )\
        .fillna(0)\
        .rename(columns=cols_min)

    df_grouped_max = df_grouped.transform(
        lambda x: x.shift(width_adjust).fillna(0).rolling(window=width).max()
    )\
        .fillna(0)\
        .rename(columns=cols_max)

    df_grouped_std = df_grouped.transform(
        lambda x: x.shift(width_adjust).fillna(0).rolling(window=width).std()
    )\
        .fillna(0)\
        .rename(columns=cols_std)

    df_grouped_sum = df_grouped.transform(
        lambda x: x.shift(width_adjust).fillna(0).rolling(window=width).sum()
    )\
        .fillna(0)\
        .rename(columns=cols_sum)
    df_grouped_median = df_grouped.transform(
        lambda x: x.shift(width_adjust).fillna(0).rolling(window=width).median()
    )\
        .fillna(0)\
        .rename(columns=cols_median)

    dfs = [df_grouped_mean, df_grouped_min,
           df_grouped_max, df_grouped_std,
           df_grouped_sum, df_grouped_median]

    dfs_merged = reduce(lambda x, y: x.join(y, lsuffix='_L', rsuffix='_R'), dfs)
    data = data.join(dfs_merged)

    return data