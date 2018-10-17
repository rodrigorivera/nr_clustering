import pandas as pd
import time


def write_df(df: pd.DataFrame,
             path: str,
             project: str,
             step: str,
             format: str = 'csv') -> None:

    title = path + time.strftime("%Y%m%d_%H%M") + '_' + project + '__' + step + '.' + format

    if format == 'csv':
        df.to_csv(title)

    if format == 'h5':
        df.to_hdf(title, step, table=True, mode='a')