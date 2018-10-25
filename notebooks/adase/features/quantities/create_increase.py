import pandas as pd


def create_increase(df: pd.DataFrame,
                    target: str,
                    ) -> pd.DataFrame:

    '''

    :param df:
    :param target:
    :param target_code_ff:
    :return:

    '''
    df_temp: pd.DataFrame = df.copy()

    title = target + '_increase'

    first_rpd = df_temp[df_temp['rpd'] == 2][target]

    df_temp[title] = (df_temp['{0}_diff'.format(target)]) / first_rpd
    df_temp[title] = df_temp[title].fillna(0)

    return df_temp
