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
        'vasculitis': 0.20,
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

class LabelConstantsMalignant:
    target_str_to_num = {
        'normal': 0,
        'malignant': 1
    }

    target_weights = {
        'normal': 0.5,
        'malignant': 0.05,
    }

    fixed_error_labels = {
        'vaskulitis': 'normal',
        'dermatitis': 'normal',
        'contact': 'normal',
        'vasculitis': 'normal',
        'necrosis': 'normal',
        'infection': 'normal',
        'pyoderma': 'normal',
        'bland': 'normal'
    }

    num_classes = len(target_str_to_num)


class LabelConstantsInfection:
    target_str_to_num = {
        'normal': 0,
        'infection': 1
    }

    target_weights = {
        'normal': 0.77,
        'infection': 0.1,
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

class LabelConstantsBland:
    target_str_to_num = {
        'normal': 0,
        'bland': 1
    }

    target_weights = {
        'normal': 0.5,
        'bland': 0.5,
    }

    fixed_error_labels = {
        'vaskulitis': 'normal',
        'dermatitis': 'normal',
        'contact': 'normal',
        'vasculitis': 'normal',
        'necrosis': 'normal',
        'malignant': 'normal',
        'pyoderma': 'normal',
        'infection': 'normal',
        'malignant': 'normal'
    }

    num_classes = len(target_str_to_num)


class LabelConstantsSpecial:
    target_str_to_num = {
        'special': 0,
        'malignant': 1,
        'infection': 2,
        'bland': 3 
    }

    target_weights = {
        'special': 0.20, #0.08 + 0.05 + 0.05 + 0.07,
        'malignant': 0.23,
        'infection': 0.20,
        'bland': 0.29
    }

    fixed_error_labels = {
        'vaskulitis': 'special',
        'dermatitis': 'special',
        'contact': 'special',
        'vasculitis': 'special',
        'necrosis': 'special',
        'pyoderma': 'special',
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

class LabelSegmentationWound:
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
        'wound': 1,
    }

    fixed_error_labels = {
        'background': 'background',
        'blandSkin': 'blandSkin',
        'granulationTissue': 'wound',
        'fibrin': 'wound',
        'scar': 'wound',
        'hyperpigmentation': 'wound',
        'xerosis': 'wound',
        'erythema': 'wound',
        'maceratedSkin': 'wound',
        'necrosis': 'wound',
        'hematoma': 'wound',
        'vessel': 'wound',
        'poikoldermSkin': 'wound',
        'hypertrophic': 'wound',
        'scale':'blandSkin' 
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


class LabelConstantsDetection:
    target_str_to_num = {
        # 'background': 0,
        'lesion': 0,
        'scale': 1
    }

    target_weights = {
        # 'background': 1,
        'lesion': 1,
        'scale': 1
    }

    fixed_error_labels = {
        'dummy': 'blue'
    }

    num_classes = len(target_str_to_num)

class LabelConstantsMNIST:
    target_str_to_num = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 2,
        '4': 2,
        '5': 2,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9
    }

    target_weights = {
        '0': 1,
        '1': 1,
        '2': 1,
        '3': 1,
        '4': 1,
        '5': 1,
        '6': 1,
        '7': 1,
        '8': 1,
        '9': 1
    }

    fixed_error_labels = {
        'dummy': 'blue'
    }

    num_classes = len(target_str_to_num)

class LabelConstantsCIFAR10:
    # 'plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    target_str_to_num = {
        'plane': 0,
        'car': 1,
        'bird': 2,
        'cat': 2,
        'deer': 2,
        'dog': 2,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9
    }

    target_weights = {
        'plane': 1,
        'car': 1,
        'bird': 1,
        'cat': 1,
        'deer': 1,
        'dog': 1,
        'frog': 1,
        'horse': 1,
        'ship': 1,
        'truck': 1
    }

    fixed_error_labels = {
        'dummy': 'blue'
    }

    num_classes = len(target_str_to_num)
    
    
class LabelConstantsHAM10000:
    target_str_to_num = {
        'nv': 0,
        'mel': 1,
        'bkl': 2,
        'bcc': 3,
        'akiec': 4,
        'vasc': 5,
        'df': 6
    }

    target_weights = {
        'nv': 1,
        'mel': 1,
        'bkl': 1,
        'bcc': 1,
        'akiec': 1,
        'vasc': 1,
        'df':  1
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
    segmentation = None

    _instances = {}

    _lock: Lock = Lock()

    def set_config(self, config):
        self.config = config
        if config['DATASET']['labels'].lower() == 'all':
            self.labels = LabelConstantsAll
        elif config['DATASET']['labels'].lower() == 'infection':
            self.labels = LabelConstantsInfection
        elif config['DATASET']['labels'].lower() == 'bland':
            self.labels = LabelConstantsBland
        elif config['DATASET']['labels'].lower() == 'malignant':
            self.labels = LabelConstantsMalignant 
        elif config['DATASET']['labels'].lower() == 'special':
            self.labels = LabelConstantsSpecial
        elif config['DATASET']['labels'].lower() == 'demo':
            self.labels = LabelConstantsDemo
        elif config['DATASET']['labels'].lower() == 'segmentation':
            self.labels = LabelSegmentation
        elif config['DATASET']['labels'].lower() == 'detection':
            self.labels = LabelConstantsDetection
        elif config['DATASET']['labels'].lower() == 'mnist':
            self.labels = LabelConstantsMNIST
        elif config['DATASET']['labels'].lower() == 'cifar10':
            self.labels = LabelConstantsCIFAR10
        elif config['DATASET']['labels'].lower() == 'woundsegmentation':
            self.labels = LabelSegmentationWound
        elif config['DATASET']['labels'].lower() == 'ham10000':
            self.labels = LabelConstantsHAM10000
        else:
            raise Exception('Dataset configuration not found.')

        try:
            p_parsed = config['DATASET']['segmentation_patch'].split(',')
            p_size = int(p_parsed[0]), int(p_parsed[1])
            class Segmentation:
                patch_size = p_size
                overlap = int(config['DATASET']['segmentation_overlap'])
        except ValueError as e:
            msg = "Error parsing the config.ini check inputs segmentation and overlap must be integer numbers."
            print(f"ERROR: {msg}")
            raise ValueError(msg)

        self.segmentation = Segmentation

    def is_config(self):
        return self.config is not None

    def label_setting(self) -> str:
        if self.config is not None:
            return str(self.config['DATASET']['labels'])
        else:
            raise Exception('Config not initialized.')
