import pandas as pd
import re
import logging
from tqdm import tqdm
from typing import List, Dict


def read_df(names: List[str],
            cols_dtypes: object = None
            ) -> Dict[str, pd.DataFrame]:
    dict_dfs = {}
    logger = logging.getLogger(__name__)

    for name in tqdm(names):

        def strip_name(name:str) -> str:
            return re.sub(r'.*__|.*/', '', name).split('.', 1)[0]

        if name.endswith('.csv?dl=1') or name.endswith('.csv'):
            dict_dfs.update({strip_name(name): pd.read_csv(name, dtype=cols_dtypes)})
            logger.info('name: {0} --  shape: {1}'.format(name, len(dict_dfs)))

        elif name.endswith('.h5'):
            dict_dfs.update({strip_name(name): pd.read_hdf(name, dtype=cols_dtypes)})
            print('h5')

        else:
            logger.info('Nothing to read')

    return dict_dfs
