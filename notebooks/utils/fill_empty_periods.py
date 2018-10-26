import pandas as pd
import numpy as np
from typing import List


def fill_empty_periods(df: pd.DataFrame,
                       group_condition: List[str],
                       ) -> pd.DataFrame:
    '''

    :param df: pd.DataFrame
    :param group_condition: List[str]
    :return: pd.DataFrame

    This creates new entries and cannot go into the pipeline of scikit. It has to be executed in advance
    In case that an item_code does not have an entry for a given rpd, it creates a new one with quantity 0.
    It also includes a column identifying if it was created or not.
    '''

    df_temp: pd.DataFrame = df.copy()
    df_temp = df_temp.set_index(group_condition)

    rpd = 'rpd'
    df_elements = df_temp.groupby(group_condition)[rpd] \
        .apply(np.array) \
        .reset_index() \
        .rename(columns={'rpd': 'rpd_elements'})

    def missing_items(rpd_list):
        list_missing_items = np.setdiff1d(np.array(range(1, 46)),
                                          rpd_list.values[0],
                                          assume_unique=True)
        if list_missing_items.size > 0:
            t = list_missing_items
        else:
            t = None
        return t

    df_missing_elements = df_elements.groupby(group_condition)['rpd_elements'] \
        .apply(missing_items) \
        .dropna() \
        .reset_index() \
        .rename(columns={'rpd_elements': rpd})

    lst_col = rpd
    df_missing = pd.DataFrame(
        {
            col: np.repeat(df_missing_elements[col].values,
                           df_missing_elements[lst_col].str.len()
                           ) for col in df_missing_elements.columns.difference([lst_col])
        }).assign(**{lst_col: np.concatenate(df_missing_elements[lst_col].values)})[
        df_missing_elements.columns.tolist()]

    return pd.concat([df, df_missing], sort=False)
