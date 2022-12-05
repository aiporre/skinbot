import configparser
import os
import logging


def create_config_file():
    config_file = configparser.ConfigParser()

    # adding default configurations

    config_file.add_section("DATASET")
    config_file.set("DATASET", "root", "./data")

    # adding logger configuration

    config_file.add_section("LOGGER")

    config_file.set("LOGGER", "LogFilePath", "./logs")
    config_file.set("LOGGER", "LogFileName", "skin.log")
    config_file.set("LOGGER", "LogLevel", "Info")
    config_file.set("LOGGER", "logtofile", "False")

    # save the config file

    with open(r"./config.ini", "w") as config_file_handler:
        config_file.write(config_file_handler)
        config_file_handler.flush()
        config_file_handler.close()

    logging.info("configuration file created with default values")
    with open("./config.ini", "r") as f:
        content = f.read()
        logging.info(content)


def main():
    if os.path.exists("./config.ini"):
        logging.info("config file exists")
    else:
        logging.info("Initalizing config file")
        create_config_file()


if __name__ == "__main__":
    main()
