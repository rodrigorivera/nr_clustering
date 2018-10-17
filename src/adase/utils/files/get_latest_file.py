import os

from .get_all_files import get_all_files
from heapq import nlargest


def get_latest_file(path: str,
                    number_of_files: int = 1):
    temp_path = path + '*'
    list_of_files = get_all_files(temp_path)
    files_sorted = sorted(list_of_files, key=os.path.getctime)
    latest_file = files_sorted[-number_of_files:]

    return latest_file
