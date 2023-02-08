import json
import os
import random
import csv
import skinbot.skinlogging as logging
# try:
#     import cv2
# except ImportError:
#     print("OpenCV is not installed, please install it to use the dataset")
from skimage.measure import label as skimage_connected_components

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from skinbot.transformers import TargetOneHot, TargetValue, Pretrained, FuzzyTargetValue, \
    DetectionTarget, DetectionPretrained, PretrainedSegmentation

from skinbot.config import Config, LabelConstantsDetection

C = Config()

# TODO: move to a config file
# LESION_LABEL_ID = 1
# SCALE_LABEL_ID = 2
# BACKGROUND_ID = 0

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
        fuzzy_labels_fixed = {target_fix_names[k]: v for k, v in fuzzy_labels.items()}
        # setting zero to all the labels that are not present
        for k in C.labels.target_str_to_num.keys():
            if k not in fuzzy_labels_fixed:
                fuzzy_labels_fixed[k] = 0
        labels_fixed[fname] = fuzzy_labels_fixed
    return labels_fixed


def crop_lesion(img, boxes):
    if len(boxes) == 0:
        return img
    xmin, ymin, xmax, ymax = min(boxes[:, 0]), min(boxes[:, 1]), max(boxes[:, 2]), max(boxes[:, 3])
    img = img[:, ymin.int():ymax.int(), xmin.int():xmax.int()]
    return img


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class WoundImages(Dataset):
    def __init__(self, root_dir,
                 fold_iteration=None,
                 cross_validation_folds=5,
                 test=False,
                 crop_lesion=False,
                 fuzzy_labels=False,
                 detection=False,
                 transform=None,
                 target_transform=None):
        # or excusive assertion between detection and fuzzy_labels
        assert not (detection and fuzzy_labels), 'detection and fuzzy_labels are mutually exclusive'
        assert not (detection and crop_lesion), 'detection and crop_lesion are mutually exclusive'
        self.fold_iteration = fold_iteration
        self.test = test
        if fold_iteration is None:
            _files = os.listdir(os.path.join(root_dir, "images"))
            _files.sort()
            self.image_fnames = [f for f in _files
                                 if '_mask.' not in f and '_detection.' not in f]
        else:
            self.kfold = KFold(root_dir, k=cross_validation_folds)
            self.image_fnames = self.kfold.get_split(fold_iteration, test=test)
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.transform = transform
        self.target_transform = target_transform
        self.fuzzy_labels = None
        # fix missing files: removes files that are invalid
        self.__fix_missing_files()

        if fuzzy_labels:
            self.load_fuzzy_labels()

        self.create_detection = detection
        self.crop_lesion = crop_lesion
        self._crop_boxes = {}

    def __fix_missing_files(self):
        # Maintains list of files valid
        for f in self.image_fnames:
            # check image path exists
            if not os.path.exists(os.path.join(self.images_dir, f)):
                logging.info('File not found: %s' % f)
                self.image_fnames.remove(f)
                continue
            # check if watershed mask exists
            mask_path = os.path.join(self.images_dir,
                                     f.replace('.jpg', '_watershed_mask.png')
                                     .replace('.JPG', '_watershed_mask.png'))
            if not os.path.exists(mask_path):
                logging.info('File not found: %s' % mask_path)
                self.image_fnames.remove(f)
                continue
            if f == 'aux_files':
                logging.info('removing aux_files')
                self.image_fnames.remove(f)
                continue
    def clear_missing_boxes(self):
        files_to_remove = []
        logging.info('Removing detection without boxes...')
        train_or_test = 'test' if self.test else 'train'
        clear_missing_fname = os.path.join(self.root_dir, f"missing_boxes_train_fold_{self.fold_iteration}_{train_or_test}.csv")
        if os.path.exists(clear_missing_fname):
            with open(clear_missing_fname, "r") as f:
                files_to_remove = []
                for l in f.readlines():
                    files_to_remove.append(l.replace('\n','').strip())
        else:
            for index in range(len(self.image_fnames)):
                label = {}
                label = self._make_one_detection_label(label, index)
                f = self.image_fnames[index]
                if len(label['boxes']) == 0:
                    files_to_remove.append(f)
                if index % 50 == 0:
                    logging.info(f'finding ... {index}/{len(self.image_fnames)}')
            with open(clear_missing_fname, 'w') as f:
                for file_to_remove in files_to_remove:
                    f.write(file_to_remove + '\n')

        for f in files_to_remove:
            logging.info(f'Remove file {f} because it doesn\'t have boxes')
            self.image_fnames.remove(f)


    def load_fuzzy_labels(self):
        fname_labels = {}
        labels = read_labels_xls(self.root_dir, concat=True)
        for fname in self.image_fnames:
            image_capture = list(filter(lambda x: fname.startswith(x), labels['Image Capture']))
            if len(image_capture) > 1:
                logging.info('Warning: more that one matching for file')
            elif len(image_capture) == 0:
                logging.info('Warning: image filename has no matching image capture entry')
                continue
            image_capture = image_capture[0]
            label = labels['Percentages of diagnoses'].loc[labels['Image Capture'] == image_capture]
            # Convert to text
            label = str(label.tolist()[0])
            try:
                values = {x.split('%')[1].strip(): float(x.split('%')[0]) for x in label.split(',')}
            except Exception as e:
                logging.info(f'Warning: parsing label "{label}" string in xlsx table. Error: {e}')
                continue
            fname_labels[fname] = values
        fname_labels = fix_target(fname_labels)
        self.image_fnames = list(fname_labels.keys())
        self.fuzzy_labels = list(fname_labels.values())

    def _read_one_detection_mask(self, fname):
        mask_path = os.path.join(self.images_dir,
                                 fname.replace('.jpg', '_watershed_mask.png')
                                 .replace('.JPG', '_watershed_mask.png'))
        mask = read_image(mask_path)
        return mask

    def _make_one_detection_label(self, label, index):
        boxes, labels, masks, areas, iscrowd = [], [], [], [], []
        # append boxes for lesions
        # first checks if ther is a json and npy lesion file
        aux_path = os.path.join(self.images_dir, "aux_files")
        # create aux_files folder if don't exists
        if not os.path.exists(aux_path):
            os.makedirs(aux_path)
        image_path = os.path.join(aux_path, self.image_fnames[index])

        detection_json_path = image_path.replace('.jpg', '_detection.json').replace('.JPG', '_detection.json')
        detection_npy_path = image_path.replace('.jpg', '_detection.npy').replace('.JPG', '_detection.npy')
        # if there is a json file and npy masks detections files then use them
        if os.path.exists(detection_json_path) and os.path.exists(detection_npy_path):
            with open(detection_json_path, 'r') as f:
                detection_json = json.load(f)
            masks = [_ for _ in np.load(detection_npy_path)]
            boxes, labels, areas, iscrowd = detection_json['boxes'], detection_json['labels'], \
                                            detection_json['areas'], detection_json['iscrowd']
        else:
            mask = self._read_one_detection_mask(self.image_fnames[index])
            lesion_ids = get_ids_by_categorie(self.root_dir, 'lesion')

            def analyse_mask(mask, obj_ids, label_id):
                _boxes, _labels, _masks, _areas = get_boxes(mask, obj_ids, label_id)
                boxes.extend(_boxes)
                labels.extend(_labels)
                masks.extend(_masks)
                areas.extend(_areas)
                iscrowd.extend([0] * len(_boxes))
            
            analyse_mask(mask, lesion_ids, C.labels.target_str_to_num['lesion'])
            skin_ids = get_ids_by_categorie(self.root_dir, 'skin')
            skin_ids.pop('blandSkin', None)
            analyse_mask(mask, skin_ids, C.labels.target_str_to_num['lesion'])

            if len(boxes) == 0:
                logging.warning(f'Warning: No lesions found in image {self.image_fnames[index]}')
            # getting the scale mask from the image
            scale_id = {"scale": 13}
            analyse_mask(mask, scale_id, C.labels.target_str_to_num['scale'])
            # save mask in npy file
            np.save(detection_npy_path, np.array(masks))
            # other info in json file
            detection_json = {'boxes': boxes,
                              'labels': labels,
                              'areas': areas,
                              'iscrowd': iscrowd}
            logging.debug(f"Resulting detection json: {detection_json}.")
            with open(detection_json_path, 'w') as f:
                json.dump(detection_json, f, cls=NpEncoder)
        # convert to np.array to speed up conversion to tensor
        boxes, labels, masks, areas, iscrowd = np.array(boxes), np.array(labels), np.array(masks), np.array(areas), \
                                               np.array(iscrowd)
        # transform to tensor
        label['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        label['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        label['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
        label['area'] = torch.as_tensor(areas, dtype=torch.float32)
        label['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int64)
        label['image_id'] = torch.as_tensor(hash(self.image_fnames[index]), dtype=torch.int64)
        return label

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.image_fnames[index])
        try:
            image = read_image(image_path)/1.0
        except Exception as e:
            logging.error(f'Cannot read image: {image_path}, check file. Error message: {e}')
            raise e
        if self.fuzzy_labels is None:
            label = self.image_fnames[index].split("_")[0]
        else:
            label = self.fuzzy_labels[index]
        if self.create_detection:
            label = self._make_one_detection_label({}, index)
        if self.crop_lesion:
            # It uses the detection label if it was created before, otherwise it creates a new one
            if not self.image_fnames[index] in self._crop_boxes:
                _label = label if self.create_detection else self._make_one_detection_label({}, index)
                boxes = _label['boxes'][_label['labels'] == LabelConstantsDetection.target_str_to_num['lesion']]
                self._crop_boxes[self.image_fnames[index]] = boxes
            else:
                boxes = self._crop_boxes[self.image_fnames[index]]
            image = crop_lesion(image, boxes)
        if self.create_detection:
            # transforms uses two x and y
            if self.transform:
                image, target = self.transform(image, label)
        else:
            if self.transform and not self.create_detection:
                image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class WoundSegmentationImages(WoundImages):
    def __init__(self, root_dir,
                 fold_iteration=None,
                 cross_validation_folds=5,
                 test=False,
                 crop_lesion=False,
                 fuzzy_labels=False,
                 detection=False,
                 transform=None,
                 target_transform=None):
        super(WoundSegmentationImages, self).__init__(
            root_dir,
            fold_iteration=fold_iteration,
            cross_validation_folds=cross_validation_folds,
            test=test,
            crop_lesion=crop_lesion,
            fuzzy_labels=fuzzy_labels,
            detection=detection,
            transform=transform,
            target_transform=target_transform)

    def __make_segmentation_label(self, index):
        mask = super(WoundSegmentationImages, self)._read_one_detection_mask(self.image_fnames[index])[0]

        # objects and backgrounds
        ids = [14,15,16,17, 255]
        for _id in ids:
            mask[mask==_id] = 0

        # hypertrophic tissue
        ids = [18, 19, 20, 21, 22, 23]
        for _id in ids:
            mask[mask==_id] = 13

        # I guess bland skin??
        ids = [28, 33]
        for _id in ids:
            mask[mask == _id] = 1

        # label scale
        mask[mask == 13] = 14

        return mask

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.image_fnames[index])
        try:
            image = read_image(image_path)/1.0
        except Exception as e:
            logging.error(f'Cannot read image: {image_path}, check file. Error message: {e}')
            raise e
        target = self.__make_segmentation_label(index)

        # It uses the detection label if it was created before, otherwise it creates a new one
        if not self.image_fnames[index] in self._crop_boxes:
            _label = self._make_one_detection_label({}, index=index)
            boxes = _label['boxes'][_label['labels'] == LabelConstantsDetection.target_str_to_num['lesion']]
            self._crop_boxes[self.image_fnames[index]] = boxes
        else:
            boxes = self._crop_boxes[self.image_fnames[index]]
        image = crop_lesion(image, boxes)
        target = crop_lesion(torch.unsqueeze(target, dim=0), boxes)
        target = target[0].long()
        # print(torch.unique(target))
        # print('before T()')
        # print('before image shape', image.shape)
        # print('before taget shape', target.shape)
        if self.transform:
            image, target = self.transform(image, target)
        if self.target_transform:
            target = self.target_transform(target)
        # print('image shape', image.shape)
        # print('taget shape', target.shape)
        # print('target unique after T: ', torch.unique(target))
        return image, target

class KFold:
    def __init__(self, root_dir, k=5):
        self.k = k
        self.splits_file = os.path.join(root_dir, f"splits_{k}.txt")
        self.image_fnames = [f for f in os.listdir(os.path.join(root_dir, "images"))
                             if '_mask.' not in f and '_detection.' not in f]
        self.images_dir = os.path.join(root_dir, "images")
        if not os.path.exists(self.splits_file):
            self.create_splits_file()
        self.fold_indices = self.read_splits()

    def create_splits_file(self):
        N = len(self.image_fnames)
        fold_num = N // self.k
        fold_extra_samples = N % self.k

        fold_indices = [[i] * fold_num for i in range(self.k)]
        if fold_extra_samples > 0:
            fold_indices[0] = fold_indices[0] + [0] * fold_extra_samples
        fold_indices_flat = []
        for x in fold_indices:
            fold_indices_flat.extend(x)
        random.shuffle(fold_indices_flat)
        with open(self.splits_file, "w", newline='') as f:
            file_writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['fname', 'fold'])
            for (fname, index) in zip(self.image_fnames, fold_indices_flat):
                file_writer.writerow([fname, index])

    def read_splits(self):
        fold_indices = {}
        with open(self.splits_file, mode='r', newline='') as f:
            file_reader = csv.DictReader(f, delimiter=',')
            for row in file_reader:
                logging.debug(f"row[fold] = {row['fold']}")
                fold_indices[row['fname']] = int(row['fold'])
        return fold_indices

    def get_split(self, fold_iteration, test=False):
        assert fold_iteration >= 0 and fold_iteration <= self.k - 1, f"fold iteration must be between 0 and {self.k - 1}"
        test_target_value = self.k - fold_iteration - 1
        logging.info(f'Fold label used as testing set is: {test_target_value}')
        test_indices = list(filter(lambda x: self.fold_indices[x] == test_target_value, self.image_fnames))
        train_indices = list(filter(lambda x: self.fold_indices[x] != test_target_value, self.image_fnames))
        logging.info('Number of indices test : %d' % len(test_indices))
        logging.info('Number of indices training: %d' % len(train_indices))
        if test:
            return test_indices  # [self.image_fnames[i] for i in test_indices]
        else:
            return train_indices  # [self.image_fnames[i] for i in train_indices]


def get_dataloaders_segmentation(config, batch, mode='all', fold_iteration=0, target='segmentation'):
    root_dir = config['DATASET']['root']
    target_transform = None

    input_size = C.segmentation.patch_size
    if mode == "all":
        transform = PretrainedSegmentation(test=True, input_size=input_size)
        wound_images = WoundSegmentationImages(root_dir, transform=transform,target_transform=target_transform)
        shuffle_dataset = False
    elif mode == 'test':
        transform = PretrainedSegmentation(test=True, input_size=input_size)
        wound_images = WoundSegmentationImages(root_dir, transform=transform,target_transform=target_transform, test=True, fold_iteration=fold_iteration)
        batch = 1
        shuffle_dataset = False
    elif mode == 'train':
        transform = PretrainedSegmentation(test=False, input_size=input_size)
        wound_images = WoundSegmentationImages(root_dir, transform=transform,target_transform=target_transform, fold_iteration=fold_iteration)
        shuffle_dataset = True

    wound_images.clear_missing_boxes() # only labels with boxes are considered for training and evaluation of detection models
    dataloader = DataLoader(wound_images, batch_size=batch, shuffle=shuffle_dataset)

    return dataloader


def get_dataloaders(config, batch, mode='all', fold_iteration=0, target='single'):
    assert mode in ['all', 'test', 'train'], 'valid options to mode are \'all\' \'test\' \'train\'.'
    valid_targets =  ['onehot', 'single', 'string', 'fuzzy', 'multiple',
                      'detection',
                      'cropFuzzy', 'cropOnehot', 'cropSingle', 'cropString', 'segmentation']
    assert target in valid_targets, f"valid options to target mode are {valid_targets}"
    # TODO: FIX THIS TO new naems for target='fuzzylabel', multilabel, string and onehot
    root_dir = config['DATASET']['root']
    fuzzy_labels = False
    _crop_lesion = False
    detection = False
    if target == 'onehot':
        target_transform = TargetOneHot()
    elif target == "single":
        target_transform = TargetValue()
    elif target == 'string':
        target_transform = None
    elif target == 'detection':
        target_transform = None
        detection = True
    elif target == 'cropOnehot':
        target_transform = TargetOneHot()
        _crop_lesion = True
    elif target == "cropSingle":
        target_transform = TargetValue()
        _crop_lesion = True
    elif target == 'cropString':
        target_transform = None
        _crop_lesion = True
    elif target == 'fuzzy' or target == 'multiple':
        fuzzy_labels = True
        target_transform = FuzzyTargetValue()
    elif target.lower() == 'cropfuzzy' or target.lower() == 'cropmultiple':
        fuzzy_labels = True
        target_transform = FuzzyTargetValue()
        _crop_lesion = True
    elif target.lower() == 'segmentation':
        return get_dataloaders_segmentation(config, batch, mode=mode, fold_iteration=fold_iteration, target=target)
    else:
        raise ValueError(f"Invalid target {target}")


    # todo: from confing input_size must be generated of input from the get_model 
    if mode == "all":
        transform = Pretrained(test=True) if not detection else DetectionPretrained(test=True)
        wound_images = WoundImages(root_dir, crop_lesion=_crop_lesion, fuzzy_labels=fuzzy_labels, transform=transform,
                                   target_transform=target_transform, detection=detection)
        shuffle_dataset = False
    elif mode == 'test':
        transform = Pretrained(test=True) if not detection else DetectionPretrained(test=True)
        wound_images = WoundImages(root_dir, fold_iteration=fold_iteration, test=True, crop_lesion=_crop_lesion,
                                   fuzzy_labels=fuzzy_labels, transform=transform, target_transform=target_transform,
                                   detection=detection)

        shuffle_dataset = False
    elif mode == 'train':
        transform = Pretrained(test=False) if not detection else DetectionPretrained(test=False)
        wound_images = WoundImages(root_dir, fold_iteration=fold_iteration, test=False, crop_lesion=_crop_lesion,
                                   fuzzy_labels=fuzzy_labels, transform=transform, target_transform=target_transform,
                                   detection=detection)
        shuffle_dataset = True
    def detection_collate(batch):
        return tuple(zip(*batch))

    if 'detection' in target:
        wound_images.clear_missing_boxes() # only labels with boxes are considered for training and evaluation of detection models
        dataloader = DataLoader(wound_images, batch_size=batch, shuffle=shuffle_dataset, collate_fn=detection_collate)
    else:
        dataloader = DataLoader(wound_images, batch_size=batch, shuffle=shuffle_dataset)

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


def read_segmentation_json(root_dir):
    segmentation_file = os.path.join(root_dir, 'labels', "segmentation.json")
    with open(segmentation_file, 'r') as f:
        data = json.load(f)
    return data['labels']


def get_ids_by_categorie(root_dir, categorie):
    segmentation_data = read_segmentation_json(root_dir)
    colors = {}
    # get the color if the category is lesion
    for name, specs in segmentation_data.items():
        if specs['categorie'] == categorie:
            colors[name] = specs['id']
    return colors


def lesion_mask_rgb_to_binary(mask, id):
    # convert the color to a binary mask
    mask_binary = (mask[0] == id).int()
    return mask_binary


def get_boxes(mask, obj_ids, obj_label_id):
    # get the bounding boxes of the objects
    boxes = []
    masks = []
    areas = []
    labels = []
    # combine the masks transformed into binary
    mask_binary = torch.zeros_like(mask[0])
    for name, id in obj_ids.items():
        mask_binary += lesion_mask_rgb_to_binary(mask, id)
    # get the bounding boxes
    mask_binary = (mask_binary > 0.5).numpy().astype(np.uint8)
    # get the boxes
    # num_components, components_joint= cv2.connectedComponents(mask_binary)
    components_joint, num_components = skimage_connected_components(mask_binary, return_num=True)
    component_ids = list(range(1, num_components))
    component_masks = components_joint == np.array(component_ids)[:, None, None]
    for ii, id in enumerate(component_ids):
        pos = np.where(component_masks[ii])
        xmin, xmax = (np.min(pos[1]), np.max(pos[1]))
        ymin, ymax = (np.min(pos[0]), np.max(pos[0]))
        area = (xmax - xmin) * (ymax - ymin)
        if area < 224 * 224:
            continue
        areas.append(area)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(obj_label_id)
        masks.append(component_masks[ii])
    return boxes, labels, masks, areas
