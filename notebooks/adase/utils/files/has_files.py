import os
import logging


def has_files(path: str) -> bool:
    logging.debug('Entered has_files')
    for dirpath, dirnames, files in os.walk(path):
        logging.debug('has_files / dirpath: {}'.format(dirpath))
        logging.debug('has_files / dirnames: {}'.format(dirnames))
        logging.debug('has_files / files: {}'.format(files))
        if files:
            logging.debug('has_files / files - TRUE: {}'.format(files))
            return True

        if not files:
            logging.debug('has_files / files - FALSE: {}'.format(files))
            return False
