import pandas as pd
import logging
from typing import Dict

from src.adase.utils.pandas import write_df


def create_smape_table(
        df: pd.DataFrame,
        config_file: Dict,
        project: str
) -> None:

    df_normal = df.groupby(['dataset', 'target', 'column', 'future_flag', 'model', 'predicted'])['SMAPE'].median()
    df_temp = pd.DataFrame(df_normal, columns=['SMAPE'])


    path = config_file['path_sources_processing']
    write_df(df_normal, path, project, 'naivefct')