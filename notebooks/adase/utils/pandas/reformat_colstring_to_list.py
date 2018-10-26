import pandas as pd


def reformat_colstring_to_list(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df_temp: pd.DataFrame = df.copy()
    df_temp["{0}_format".format(name)] = df_temp[name].apply(lambda x: x.split())
    df_temp["{0}_format".format(name)] = df_temp["{0}_format".format(name)].apply(lambda x: [s.strip('[]') for s in x])

    return df
