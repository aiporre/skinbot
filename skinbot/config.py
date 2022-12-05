import configparser

def read_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


