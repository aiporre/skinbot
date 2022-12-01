import os
import random
import csv

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from skinbot.transformers import TargetOneHot, TargetValue, Pretrained, target_str_to_num, FuzzyTargetValue

# TODO: move to a config file
LABEL_COLUMN_NAMES = ['Image Capture',
                      'Rationale for decision',
                      'Percentages of diagnoses',
                      'Scale',
                      'Several angles',
                      'Several timepoints',
                      'Timepoint',
                      'Angle']

target_fix_names = {
    'vasculitis': 'vasculitis',
    'bland': 'bland',
    'pyoderma': 'pyoderma',
    'malignant': 'malignant',
    'ulceration': 'necrosis',
    'bland.': 'bland',
    'necrosis': 'necrosis',
    'blande': 'bland',
    'contact dermatitis': 'contact',
    'infection': 'infection',
    'vasculitits': 'vasculitis',
    'balnd': 'bland',
    '(contact) dermatitis': 'contact',
    'vaskulitis': 'vasculitis',
    '(contact)dermatits': 'contact',
    '(contact)dermatitis': 'contact',
    'infecrion': 'infection',
    'dermatitis': 'dermatitis',
}

def fix_target(labels):
    labels_fixed = {}
    for fname, fuzzy_labels in labels.items():
        # changes names to the ones corrected in the dictionary target_fix_names
        fuzzy_labels_fixed = { target_fix_names[k]: v for k,v in fuzzy_labels.items()}
        # setting zero to all the labels that are not present
        for k in target_str_to_num.keys():
            if k not in fuzzy_labels_fixed:
                fuzzy_labels_fixed[k] = 0
        labels_fixed[fname] = fuzzy_labels_fixed
    return labels_fixed

class WoundImages(Dataset):
    def __init__(self, root_dir,
                 fold_iteration=None,
                 cross_validation_folds=10,
                 test=False,
                 fuzzy_labels=False,
                 detection=False,
                 transform=None,
                 target_transform=None):
        # or excusive assertion between detection and fuzzy_labels
        assert not (detection and fuzzy_labels), 'detection and fuzzy_labels are mutually exclusive'
        if fold_iteration is None:
            self.image_fnames= [f for f in os.listdir(os.path.join(root_dir, "images")) if not '_mask.' in f]
        else:
            self.kfold = KFold(root_dir, k=cross_validation_folds)
            self.image_fnames = self.kfold.get_split(fold_iteration, test=test)
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.transform = transform
        self.target_transform = target_transform
        self.fuzzy_labels = None
        if fuzzy_labels:
            self.load_fuzzy_labels()
        if detection:
            self.load_detection()

    def load_detection(self):
        self.detection = {}
        for fname in self.image_fnames:
            mask_path = os.path.join(self.images_dir,
                                     fname.replace('.jpg', '_watershed_mask.png')
                                     .replace('.JPG', '_watershed_mask.png'))
            mask = read_image(mask_path)
            self.detection[fname] = mask

    def load_fuzzy_labels(self):
        fname_labels = {}
        labels = read_labels_xls(self.root_dir, concat=True)
        for fname in self.image_fnames:
            image_capture = list(filter(lambda x: fname.startswith(x), labels['Image Capture']))
            if len(image_capture) > 1:
                print('Warning: more that one matching for file')
            elif len(image_capture) == 0:
                print('Warning: image filename has no matching image capture entry')
                continue
            image_capture = image_capture[0]
            label = labels['Percentages of diagnoses'].loc[labels['Image Capture'] == image_capture]
            # Convert to text
            label = str(label.tolist()[0])
            try:
                values = {x.split('%')[1].strip(): float(x.split('%')[0]) for x in label.split(',')}
            except Exception as e:
                print(f'Warining: parsing label "{label}" string in xlsx table', e)
                continue
            fname_labels[fname] = values
        fname_labels = fix_target(fname_labels)
        self.image_fnames = list(fname_labels.keys())
        self.fuzzy_labels = list(fname_labels.values())

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.image_fnames[index])
        image = read_image(image_path)
        if self.fuzzy_labels is None:
            label = self.image_fnames[index].split("_")[0]
        else:
            label = self.fuzzy_labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class KFold:
    def __init__(self, root_dir, k=10):
        self.k = k
        self.splits_file = os.path.join(root_dir, f"splits_{k}.txt")
        self.image_fnames= [f for f in os.listdir(os.path.join(root_dir, "images")) if not '_mask.' in f]
        self.images_dir = os.path.join(root_dir, "images")
        if not os.path.exists(self.splits_file):
            self.create_splits_file()
        self.fold_indices = self.read_splits()

    def create_splits_file(self):
        N = len(self.image_fnames)
        fold_num = N // self.k
        fold_extra_samples = N % self.k

        fold_indices = [[i]*fold_num for i in range(self.k)]
        if fold_extra_samples > 0:
            fold_indices[0] = fold_indices[0] + [0]*fold_extra_samples
        fold_indices_flat = []
        for x in fold_indices:
            fold_indices_flat.extend(x)
        random.shuffle(fold_indices_flat)
        with open(self.splits_file, "w", newline='') as f:
            file_writer = csv.writer(f,  delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['fname', 'fold'])
            for (fname, index) in zip(self.image_fnames, fold_indices_flat):
                file_writer.writerow([fname, index])

    def read_splits(self):
        fold_indices = []
        with open(self.splits_file, mode='r', newline='') as f:
            file_reader = csv.DictReader(f, delimiter=',')
            for row in file_reader:
                # print('DEBUG row[fold] = ', row['fold'])
                fold_indices.append(int(row['fold']))
        return fold_indices

    def get_split(self, fold_iteration, test=False):
        assert fold_iteration>=0 and fold_iteration<=self.k-1, f"fold iteration must be between 0 and {self.k-1}"
        test_target_value = self.k - fold_iteration - 1
        print('DEBUG: test_target_value ', test_target_value)
        test_indices = list(filter(lambda x: self.fold_indices[x] == test_target_value, range(len(self.fold_indices))))
        train_indices = list(filter(lambda x: self.fold_indices[x] != test_target_value, range(len(self.fold_indices))))
        print('DEBuG lend of test indices: ', len(test_indices))
        print('DEBuG lend of train indices: ', len(train_indices))
        if test:
            return [self.image_fnames[i] for i in test_indices]
        else:
            return [self.image_fnames[i] for i in train_indices]

def get_dataloaders(config, batch, mode='all', fold_iteration=0, target='single'):
    assert mode in ['all', 'test', 'train'], 'valid options to mode are \'all\' \'test\' \'train\'.'
    assert target in ['onehot', 'single', 'string', 'fuzzy', 'multiple'], "valid options to target mode are 'onehot', 'number' or 'string, or 'fuzzy'"
    # TODO: FIX THIS TO new naems for target='fuzzylabel', multilabel, string and onehot
    root_dir = config['DATASET']['root']
    fuzzy_labels = False
    if target == 'onehot':
        target_transform = TargetOneHot()
    elif target == "single":
        target_transform = TargetValue()
    elif target == 'string':
        target_transform =None
    elif target == 'fuzzy' or target == 'multiple':
        fuzzy_labels = True
        target_transform = FuzzyTargetValue()
    else:
        raise ValueError(f"Invalid target {target}")

    
    # todo: from confing input_size must be generated of input from the get_model 
    if mode == "all":
        transform = Pretrained(test=True)
        wound_images = WoundImages(root_dir, fuzzy_labels=fuzzy_labels, transform=transform, target_transform=target_transform)
        dataloader = DataLoader(wound_images, batch_size=batch, shuffle=False)
    elif mode == 'test':
        transform = Pretrained(test=True)
        wound_images = WoundImages(root_dir, fold_iteration=fold_iteration, test=True, fuzzy_labels=fuzzy_labels,  transform=transform, target_transform=target_transform)

        # print('DEBUG: lenght of the dataset test: ', len(wound_images) )
        dataloader = DataLoader(wound_images, batch_size=batch, shuffle=False)
    elif mode == 'train':
        transform = Pretrained(test=False)
        wound_images = WoundImages(root_dir, fold_iteration=fold_iteration, test=False, fuzzy_labels=fuzzy_labels, transform=transform, target_transform=target_transform)
        # print('DEBUG: lenght of the dataset train: ', len(wound_images) )
        dataloader = DataLoader(wound_images, batch_size=batch, shuffle=True)
    return dataloader


def read_labels_xls(root_dir, concat=True):
    tables = {}
    labels_file = os.path.join(root_dir, 'labels', "labels.xlsx")
    xls = pd.ExcelFile(labels_file)
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        df = df.rename(columns=lambda x: x.strip())
        tables[sheet_name] = df[LABEL_COLUMN_NAMES]
    if not concat:
        return tables
    # concat all tables
    df = pd.concat(tables.values(), ignore_index=True)
    return df





    
