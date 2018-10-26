import pickle
import time

from typing import Dict


def write_pickle(model: Dict,
                 path: str,
                 project: str,
                 step: str
                 ) -> None:

    title = path + time.strftime("%Y%m%d_%H%M") + '_' + project + '__' + step + '.sav'
    pickle.dump(model, open(title, 'wb'))