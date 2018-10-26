import pandas as pd
import math
from typing import Dict


def create_quarter_semester_year(
        df: pd.DataFrame,
        periods: Dict
) -> pd.DataFrame:
    '''

    :param df: pd.DataFrame
    :param periods: Dict = {'quarter': 3, 'semester': 6, 'year':12}
    :return: pd.DataFrame
    It creates three new features one for the corresponding month within the semester
    and the other within the quarter and within the year
    '''

    df_temp: pd.DataFrame = df.copy()

    for key, value in periods.items():
        df_temp[key] = df_temp\
            .apply(lambda row: int(math.ceil(row.month / value)), axis=1)

    return df_temp

