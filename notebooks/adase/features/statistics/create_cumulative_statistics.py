import pandas as pd
from typing import List
from functools import reduce


def create_cumulative_statistics(df: pd.DataFrame,
                                 targets: List[str],
                                 condition: List[str]
                                 ) -> pd.DataFrame:
    '''

    :param df:
    :param target:
    :return:
    It created cumulative statistics, one new column for each of the statistics such as mean, min, max,
    std, sum, median. In total 6 for each target
    '''

    data: pd.DataFrame = df.copy()
    cum = 'cum'

    cols_mean = {}
    cols_min = {}
    cols_max = {}
    cols_std = {}
    cols_sum = {}
    cols_median = {}
    for target in targets:
        cols_mean[target] = '{0}_{1}_mean'.format(target, cum)
        cols_min[target] = '{0}_{1}_min'.format(target, cum)
        cols_max[target] = '{0}_{1}_max'.format(target, cum)
        cols_std[target] = '{0}_{1}_std'.format(target, cum)
        cols_sum[target] = '{0}_{1}_sum'.format(target, cum)
        cols_median[target] = '{0}_{1}_median'.format(target, cum)

    to_use = data[condition + targets]
    df_grouped = to_use.groupby(condition)[targets]

    df_grouped_mean = df_grouped.transform(
        lambda x: x.expanding().mean()
    )\
        .fillna(0)\
        .rename(columns=cols_mean)

    df_grouped_min = df_grouped.transform(
        lambda x: x.expanding().min()
    )\
        .fillna(0)\
        .rename(columns=cols_min)

    df_grouped_max = df_grouped.transform(
        lambda x: x.expanding().max()
    )\
        .fillna(0)\
        .rename(columns=cols_max)

    df_grouped_std = df_grouped.transform(
        lambda x: x.expanding().std()
    )\
        .fillna(0).rename(columns=cols_std)

    df_grouped_sum = df_grouped.transform(
        lambda x: x.expanding().sum()
    )\
        .fillna(0)\
        .rename(columns=cols_sum)

    df_grouped_median = df_grouped.transform(
        lambda x: x.expanding().median()
    )\
        .fillna(0)\
        .rename(columns=cols_median)

    dfs = [df_grouped_mean, df_grouped_min,
           df_grouped_max, df_grouped_std,
           df_grouped_sum, df_grouped_median]

    dfs_merged = reduce(lambda x, y: x.join(y, lsuffix='_L', rsuffix='_R'), dfs)
    data = data.join(dfs_merged)


    return data
