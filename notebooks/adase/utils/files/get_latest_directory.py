import logging
import datetime
import os
import glob

from .get_all_files import get_all_files
from .directory_path import directory_path


def get_latest_directory(path: str):

    #logging.info('Entered get_latest_directory')
    full_path = os.path.join(path, datetime.datetime.now().strftime('%Y/'))
    #logging.info('get_latest_directory / full_path: {}'.format(full_path))
    temp_path = full_path + '*/'
    if glob.glob(temp_path):
        latest_dir = max(glob.glob(temp_path), key=os.path.getmtime)
    else:
        dir_path = directory_path(path)
        latest_dir = max(glob.glob(temp_path), key=os.path.getmtime)

    #logging.info('get_latest_directory / temp_path: {}'.format(latest_dir))
    list_of_directories = get_all_files(latest_dir)
    #logging.info('get_latest_directory / list_of_directories: {}'.format(list_of_directories))
    latest_directory = max(list_of_directories, key=os.path.getctime)
    #logging.info('get_latest_directory / latest_directory: {}'.format(latest_directory))
    latest_directory = latest_directory + '/'
    #logging.info('get_latest_directory / latest_directory: {}'.format(latest_directory))

    return latest_directory
