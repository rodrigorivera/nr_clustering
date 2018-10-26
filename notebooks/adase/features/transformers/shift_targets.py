import pandas as pd
from typing import List


def shift_targets(df: pd.DataFrame,
                  targets: List[str],
                  list_group_condition: List[str],
                  shift: int,
                  suffix: str
                  ) -> pd.DataFrame:
    '''

    :param df:
    :param group_condition:
    :param targets:
    :param shift:
    :param name:
    :return:

    Shift the target by 1 period and store it in a new column.
    NA are replaced with 0

    '''

    data: pd.DataFrame = df.copy()

    data = data.set_index(list_group_condition)
    col_names = ['{0}_{1}{2}'.format(x, shift, suffix) for x in targets]

    data[col_names] = data.groupby(list_group_condition)[targets]\
        .shift(shift)\
        .fillna(0)

    data = data.reset_index()


    return data


