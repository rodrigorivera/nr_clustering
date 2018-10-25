import pandas as pd


def create_creation_date(
        df: pd.DataFrame,
        delivery_date: str = 'rpd',
        delivery_time: str = 'future_flag'
) -> pd.DataFrame:

    '''

    :param df:
    :param delivery_date: str = 'rpd'
    :param delivery_time: str = 'future_flag'
    :return:

    Add a new column with a creation date for each entry
    '''
    df_temp: pd.DataFrame = df.copy()
    df_temp[delivery_date + '_creation'] = df_temp[delivery_date] - df_temp[delivery_time]
    df_temp[delivery_date + '_creation'] = df_temp[delivery_date + '_creation'].astype(int)

    return df_temp
