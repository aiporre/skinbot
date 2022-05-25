import os
import random
import csv
from torch.utils.data import Dataset
from torchvision.io import read_image

class WoundImages(Dataset):
    def __init__(self, root_dir, fold_iteration=None, cross_validation_folds=10, test=False, transform=None, target_transform=None):
        if fold_iteration is None:
            self.image_fnames= os.listdir(os.path.join(root_dir, "images"))
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
        self.image_fnames= os.listdir(os.path.join(root_dir, "images"))
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
            file_writer = csv.writer(f,  delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['fname', 'fold'])
            for (fname, index) in zip(self.image_fnames, fold_indices_flat):
                file_writer.writerow([fname, index])

    def read_splits(self):
        fold_indices = []
        with open(self.splits_file, mode='r', newline='') as f:
            file_reader = csv.DictReader(f, delimiter=' ')
            for row in file_reader:
                print(row)
                fold_indices.append(row['fold'])
        return fold_indices

    def get_split(self, fold_iteration, test=False):
        assert fold_iteration>=0 and fold_iteration<=self.k-1, f"fold iteration must be between 0 and {self.k-1}"
        test_target_value = self.k - fold_iteration - 1
        test_indices = list(filter(lambda x: self.fold_indices[x] == test_target_value, range(len(self.fold_indices))))
        train_indices = list(filter(lambda x: self.fold_indices[x] != test_target_value, range(len(self.fold_indices))))
        if test:
            return [self.image_fnames[i] for i in test_indices]
        else:
            return [self.image_fnames[i] for i in train_indices]

