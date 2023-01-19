import configparser
from threading import Lock
from skinbot.singleton import SingletonMeta


class LabelConstantsAll:
    target_str_to_num = {
        'contact': 0,
        'vasculitis': 1,
        'necrosis': 2,
        'malignant': 3,
        'pyoderma': 4,
        'infection': 5,
        'bland': 6
    }

    target_weights = {
        'contact': 0.08,
        'vasculitis': 0.05,
        'necrosis': 0.05,
        'malignant': 0.23,
        'pyoderma': 0.07,
        'infection': 0.20,
        'bland': 0.29
    }

    fixed_error_labels = {
        'vaskulitis': 'vasculitis',
        'dermatitis': 'contact',
    }

    num_classes = len(target_str_to_num)


class LabelConstantsInfection:
    target_str_to_num = {
        'normal': 0,
        'infection': 1
    }

    target_weights = {
        'normal': 0.77,
        'infection': 0.20,
    }

    fixed_error_labels = {
        'vaskulitis': 'normal',
        'dermatitis': 'normal',
        'contact': 'normal',
        'vasculitis': 'normal',
        'necrosis': 'normal',
        'malignant': 'normal',
        'pyoderma': 'normal',
        'bland': 'normal'
    }

    num_classes = len(target_str_to_num)


class LabelSegmentation:
    target_str_to_num = {
        'background': 0,
        'blandSkin': 1,
        'granulationTissue': 2,
        'fibrin': 3,
        'scar': 4,
        'hyperpigmentation': 5,
        'xerosis': 6,
        'erythema': 7,
        'maceratedSkin': 8,
        'necrosis': 9,
        'hematoma': 10,
        'vessel': 11,
        'poikoldermSkin': 12,
        'hypertrophic': 13,
        'scale': 14
    }

    target_weights = {
        'background': 1,
        'blandSkin': 1,
        'granulationTissue': 1,
        'fibrin': 1,
        'scar': 1,
        'hyperpigmentation': 1,
        'xerosis': 1,
        'erythema': 1,
        'maceratedSkin': 1,
        'necrosis': 1,
        'hematoma': 1,
        'vessel': 1,
        'poikoldermSkin': 1,
    }

    fixed_error_labels = {
    }

    num_classes = len(target_str_to_num)

class LabelConstantsDemo:
    target_str_to_num = {
        'blue': 0,
        'red': 1
    }

    target_weights = {
        'blue': 0.5,
        'infection': 0.5,
    }

    fixed_error_labels = {
        'dummy': 'blue'
    }

    num_classes = len(target_str_to_num)

def read_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


class Config(metaclass=SingletonMeta):
    config = None
    labels = None

    _instances = {}

    _lock: Lock = Lock()

    def set_config(self, config):
        self.config = config
        if config['DATASET']['labels'].lower() == 'all':
            self.labels = LabelConstantsAll
        elif config['DATASET']['labels'].lower() == 'infection':
            self.labels = LabelConstantsInfection
        elif config['DATASET']['labels'].lower() == 'demo':
            self.labels = LabelConstantsDemo
        elif config['DATASET']['labels'].lower() == 'segmentation':
            self.labels = LabelSegmentation
        else:
            raise Exception('Dataset configuration not found.')

    def is_config(self):
        return self.config is not None

    def label_setting(self) -> str:
        if self.config is not None:
            return str(self.config['DATASET']['labels'])
        else:
            raise Exception('Config not initialized.')

