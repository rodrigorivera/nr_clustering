from typing import List
import logging
import glob


def get_all_files(path: str) -> List:
    if not path.endswith('*'):
        logging.debug('Path does not end with *')
        path = path + '*'
        logging.debug('Added path: {}'.format(path))

    logging.debug('Entered get_all_files')
    list_of_files = glob.glob(path)  # * means all if need specific format then *.csv
    logging.debug('get_all_files / full_path: {}'.format(list_of_files))

    return list_of_files
