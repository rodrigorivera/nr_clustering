import pickle


def read_pickle(name: str):
    return pickle.load(open(name, 'rb'))