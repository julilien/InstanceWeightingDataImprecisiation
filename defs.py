import os
import configparser

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = 'conf/run.ini'
CONFIG_FILE_ENCODING = 'utf-8-sig'


def get_config(path=None):
    config = configparser.ConfigParser()
    if path is None:
        path = CONFIG_FILE

    config.read(os.path.join(ROOT_DIR, path), encoding=CONFIG_FILE_ENCODING)
    return config
