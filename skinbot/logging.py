import logging


def info(message):
    print(message)
    logging.info(message)

def error(message):
    print(message)
    logging.error(message)

def warn(message):
    print(message)
    logging.warning(message)

def debug(message):
    print(message)
    logging.debug(message)
