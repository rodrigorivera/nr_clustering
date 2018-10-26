import pandas as pd
import numpy as np
from typing import List


def create_log_targets(df: pd.DataFrame,
                       targets: List[str]
                       ) -> pd.DataFrame:

    '''

    :param df:
    :param targets:
    :return:

    It creates a new column with a log transformation of the target
    '''

    df_temp: pd.DataFrame = df.copy()

    for target in targets:
        target_stripped = target.replace('original_', '')
        title = 'log_{}'.format(target_stripped)
        df_temp[title] = np.log1p(df_temp[target] + 1)

    return df_temp
