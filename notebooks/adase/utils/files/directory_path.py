import datetime
import os
import logging


def directory_path(path: str):
    full_path = os.path.join(path, datetime.datetime.now().strftime('%Y/%m/%d/'))
    logger = logging.getLogger(__name__)

    if os.path.isdir(full_path):
        logger.info('{} --  path exists'.format(full_path))

    else:
        os.makedirs(full_path)
        logger.info('{} --  creating path'.format(full_path))

    return full_path
