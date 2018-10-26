import datetime
import glob
import logging
import os
import time

from typing import List, Dict
import numpy as np
import pandas as pd


def loc_expand(df, loc):
    rows = []
    for i, row in df.iterrows():
        vs = row.at[loc]
        new = row.copy()
        for v in vs:
            new.at[loc] = v
            rows.append(new)

    return pd.DataFrame(rows)


def format_col_list(df, targets):
    df_temp = df.copy()
    for target in targets:
        df_temp[target + '_format'] = df_temp[target].apply(lambda x: x.split())
        df_temp[target + '_format'] = df_temp[target + '_format'].apply(lambda x: [s.strip('[]') for s in x])
        df_temp[target + '_format'] = df_temp[target + '_format'].apply(lambda x: list(filter(None, x)))
        df_temp[target + '_format'] = df_temp[target + '_format'].apply(lambda x: [float(s) for s in x])
        df_temp[target + '_format'] = df_temp[target + '_format'].apply(lambda x: np.array(x))
        # new_df['y_pred_format'] = new_df['y_pred_format'].map(lambda x: [s for s in list(x)[0]])

    return df_temp


def remove_square_brackets(df, targets):
    df_temp = df.copy()

    for target in targets:
        df_temp[target] = df_temp[target].map(lambda x: x.lstrip('[').rstrip(']'))
        df_temp[target] = df_temp[target].astype('int')

    return df_temp


def get_random_items_future_flags(df):
    product_id = np.random.choice(df['item_code'].values, 1)[0]
    future_flag = np.random.choice(df['future_flag'].values, 1)[0]

    return product_id, future_flag
