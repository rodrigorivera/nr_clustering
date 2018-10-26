import pandas as pd


def pivot_df(df: pd.DataFrame,
             index_name,
             columns_name,
             values_name,
             aggfunc=sum) -> pd.DataFrame:

    df_temp: pd.DataFrame = df.copy()
    df_temp = df_temp.pivot_table(
        index=index_name,
        columns=columns_name,
        aggfunc=aggfunc,
        values=values_name,
        fill_value=0,
        margins=True)

    return df_temp
