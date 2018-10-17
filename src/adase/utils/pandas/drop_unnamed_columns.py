import pandas as pd


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:

    if not df.filter(regex='Unnamed: ').columns.empty:
        to_drop = df.filter(regex='Unnamed: ').columns
        df = df.drop(to_drop, axis=1)

    return df
