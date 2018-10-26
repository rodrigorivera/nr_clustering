import pandas as pd
from typing import List


def create_z_score(df: pd.DataFrame,
                   targets: List[str],
                   group_condition: List[str]
                   ) -> pd.DataFrame:

    '''

    :param df:
    :param target:
    :param target_code_ff:
    :return:

    it creates a new column with a cumulative z-score corresponding to each target
    '''

    df_temp: pd.DataFrame = df.copy()
    target_stripped = [x.replace('original_', '') for x in targets]

    df_temp = df_temp.set_index(group_condition)
    col_names = ['zscore_{}'.format(x) for x in target_stripped]

    df_temp[col_names] = df_temp.groupby(group_condition)[targets].transform(
        lambda x: (x - x.expanding().mean()) / x.expanding().std()).fillna(0)
    df_temp = df_temp.reset_index()

    return df_temp
