import logging
import os


def validate_target_mode(_target_mode, comparable_items):
    if any([item in _target_mode.lower() for item in comparable_items]):
        return True
    else:
        return False

def get_log_path(config):
    logger_dir = config['LOGGER']['logfilepath']
    logger_fname = config['LOGGER']['logfilename']
    return os.path.join(logger_dir, logger_fname)

def configure_logging(config):
    # configure the logging system
    log_level = config['LOGGER']['loglevel']
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)

    if config['LOGGER']['logtofile'] == 'True':
        logger_dir = config['LOGGER']['logfilepath']
        logger_fname = config['LOGGER']['logfilename']
        logger_path = os.path.join(logger_dir, logger_fname)
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        print('Logging to: ', logger_path, ' with level: ', log_level)
        logging.basicConfig(filename=logger_path, filemode='w', level=numeric_level)
    else:
        logging.basicConfig(level=numeric_level)



