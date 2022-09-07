import os
import random
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from skinbot.transformers import TargetOneHot, TargetValue, Pretrained

class WoundImages(Dataset):
    def __init__(self, root_dir, fold_iteration=None, cross_validation_folds=10, test=False, transform=None, target_transform=None):
        if fold_iteration is None:
            self.image_fnames= [f for f in os.listdir(os.path.join(root_dir, "images")) if not '_mask.' in f]
        else:
            self.kfold = KFold(root_dir, k=cross_validation_folds)
            self.image_fnames = self.kfold.get_split(fold_iteration, test=test)
        self.images_dir = os.path.join(root_dir, "images")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.image_fnames[index])
        image = read_image(image_path)
        label = self.image_fnames[index].split("_")[0]
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
        # print('DEBUG: test_target_value ', test_target_value)
        test_indices = list(filter(lambda x: self.fold_indices[x] == test_target_value, range(len(self.fold_indices))))
        train_indices = list(filter(lambda x: self.fold_indices[x] != test_target_value, range(len(self.fold_indices))))
        # print('DEBuG lend of test indices: ', len(test_indices))
        # print('DEBuG lend of train indices: ', len(train_indices))
        if test:
            return [self.image_fnames[i] for i in test_indices]
        else:
            return [self.image_fnames[i] for i in train_indices]

def get_dataloaders(config, batch, mode='all', fold_iteration=0, target='number'):
    assert mode in ['all', 'test', 'train'], 'valid options to mode are \'all\' \'test\' \'train\'.'
    assert target in ['onehot', 'number', 'string'], "valid options to target mode are 'onehot', 'number' or 'string'"

    root_dir = config['DATASET']['root']
    if target == 'onehot':
        target_transform = TargetOneHot()
    elif target == "number":
        target_transform = TargetValue()
    elif target == 'string':
        target_transform =None 
    
    # todo: from confing input_size must be generated of input from the get_model 
    if mode == "all":
        wound_images = WoundImages(root_dir, target_transform=target_transform)
    elif mode == 'test':
        transform = Pretrained(test=True)
        wound_images = WoundImages(root_dir, fold_iteration=fold_iteration, test=True, transform=transform, target_transform=target_transform)

        # print('DEBUG: lenght of the dataset test: ', len(wound_images) )
        dataloader = DataLoader(wound_images, batch_size=batch, shuffle=False)
    elif mode == 'train':
        transform = Pretrained(test=False)
        wound_images = WoundImages(root_dir, fold_iteration=fold_iteration, test=False, transform=transform, target_transform=target_transform)
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





    
