import requests

from .get_file_name import get_file_name


def get_raw_file(path: str,
                 url: str
                 ) -> None:
    r = requests.get(url, allow_redirects=True)
    file_name = get_file_name(url)
    file_name_location = path + file_name
    open(file_name_location, 'wb').write(r.content)