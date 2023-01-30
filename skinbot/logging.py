import logging
from skinbot.config import Config
C = Config()


def info(message):
    if C.config is not None and C.config['LOGGER']['logtofile'] == 'True':
        print(message)
    logging.info(message)

def error(message):
    if C.config is not None and C.config['LOGGER']['logtofile'] == 'True':
        print(message)
    logging.error(message)

def warn(message):
    if C.config is not None and C.config['LOGGER']['logtofile'] == 'True':
        print(message)
    logging.warning(message)

def debug(message):
    logging.debug(message)
