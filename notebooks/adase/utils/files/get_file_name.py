def get_file_name(url: str) -> str:
    file_name = None
    if url.find('/'):
        file_name = url.rsplit('/', 1)[1]

    return file_name
