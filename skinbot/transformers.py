import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from collections.abc import Iterable
from skinbot.torchvisionrefs import transforms as detection_transforms
from skinbot.config import Config
C = Config()

class TargetOneHot:
    def __init__(self):
        self.target_set_num = len(C.labels.target_str_to_num)
    def __call__(self, x):
        x = x.strip().lower()
        # string is fixed if it is in the fixed_error_labels dictionary
        if x in C.labels.fixed_error_labels:
            x = C.labels.fixed_error_labels[x]
        x_transform = np.zeros(self.target_set_num)
        x_transform[C.labels.target_str_to_num[x]] = 1.
        return x_transform

class TargetValue:
    def __call__(self, x):
        x = x.strip().lower()
        # string is fixed if it is in the fixed_error_labels dictionary
        if x in C.labels.fixed_error_labels:
            x = C.labels.fixed_error_labels[x]
        return C.labels.target_str_to_num[x]

class TargeOneHotfromNum:
    def __init__(self):
        self.target_set_num = len(C.labels.target_str_to_num)
        self.to_float = ToFloat()
        self.totesnor = transforms.ToTensor()
    def __call__(self, y):
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=int)
        y =  torch.nn.functional.one_hot(y, num_classes=C.labels.num_classes)
        return y

class FuzzyTargetValue:
    def __call__(self, x):
        values = np.array([x[k] for k in C.labels.target_str_to_num.keys()])
        values = values / 100  # np.max(values)
        return values

class ToFloat:
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x.float()
        elif isinstance(x, np.ndarray):
            return x.astype(np.float32)
        elif isinstance(x, Image.Image):
            return np.array(x).astype(np.float32)
        else:
            raise ValueError(f'Input x is type {type(x)} not supported. Valid inputs are torch.Tensor and numpy.ndarray')

class Pretrained:
    def __init__(self, test=False, input_size=224):
        self.test = test
        self.T = {
            'train': transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90),
                    #transforms.ToTensor(),
                    ToFloat(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'val': transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    # transforms.ToTensor(),
                    ToFloat(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        }
    def __call__(self, x):
        if self.test:
            return self.T['val'](x)
        else:
            return self.T['train'](x)

class PretrainedMNIST:
    def __init__(self, test=False, input_size=28):
        self.test = test
        self.T = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                #transforms.RandomResizedCrop(input_size),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                #transforms.RandomRotation(90),
                ToFloat(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                    # transforms.ToTensor(),
                    ToFloat(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
    def __call__(self, x):
        if self.test:
            return self.T['val'](x)
        else:
            return self.T['train'](x)
class PadIfLess:
    def __init__(self, input_size):
        self.input_size = max(input_size) if isinstance(input_size, Iterable) else input_size
        self.pad = transforms.CenterCrop(input_size)
    def __call__(self, x):
        H, W = x.shape[-2], x.shape[-1]
        if H < self.input_size or W < self.input_size:
            return self.pad(x)
        else:
            return x


class PretrainedSegmentation:
    def __init__(self, test=False, input_size=224):
        print('inputsize ', type(input_size))
        self.test = test
        self.T = {
            'train': transforms.Compose([
                transforms.RandomCrop(input_size, pad_if_needed=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                #transforms.ToTensor(),
                ToFloat(),
                transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
            ]),
            'val': transforms.Compose([
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                # transforms.ToTensor(),
                ToFloat(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    def __call__(self, x, y):
        if self.test:
            return self.T['val'](x), y
        else:
            y = torch.unsqueeze(y, dim=0).float()
            xx = torch.cat([x, y], 0)
            xx = self.T['train'](xx)
            return xx[:3], xx[-1].long()

class DetectionPretrained:
    def __init__(self, test=False):
        self.test = test
        T = []
        T.append(detection_transforms.ConvertImageDtype(torch.float))
        if not test:
            T.append(detection_transforms.RandomHorizontalFlip(0.5))
        self.T = detection_transforms.Compose(T)

    def __call__(self, x, y=None):
        return self.T(x,y)

class DetectionTarget:
    # Applies the transform to the target "image_label" if Target is a "detection" kind of label
    def __init__(self, target_transform):
        self.target_transform = target_transform
    def __call__(self, x):
        x['image_label'] = torch.as_tensor(self.target_transform(x['image_label']))
        return x

