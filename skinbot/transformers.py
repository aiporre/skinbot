import numpy as np
import torch
from torchvision import transforms

target_str_to_num = {
    'contact' :  0,
    'vasculitis' :  1,
    'necrosis' :  2,
    'malignant' :  3,
    'pyoderma': 4, 
    'dermatitis' :  5,
    'infection' :  6,
    'bland' : 7
}
fixed_error_labels = {
    'vaskulitis': 'vasculitis',
}

num_classes = 8

class TargetOneHot:
    def __init__(self):
        self.target_set_num = len(target_str_to_num)
    def __call__(self, x):
        x = x.strip().lower()
        # string is fixed if it is in the fixed_error_labels dictionary
        if x in fixed_error_labels:
            x = fixed_error_labels[x]
        x_transform = np.zeros(self.target_set_num)
        x_transform[target_str_to_num[x]] = 1.
        return x_transform

class TargetValue:
    def __call__(self, x):
        x = x.strip().lower()
        # string is fixed if it is in the fixed_error_labels dictionary
        if x in fixed_error_labels:
            x = fixed_error_labels[x]
        return target_str_to_num[x]

class FuzzyTargetValue:
    def __call__(self, x):
        values = np.array( [x[k] for k in target_str_to_num.keys()])
        values = values / 100  # np.max(values)
        return values

class ToFloat:
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x.float()
        elif isinstance(x, np.ndarray):
            return x.astype(np.float32)
        else:
            raise ValueError(f'Input x is type {type(x)} not supported. Valids are torch.Tensor and numpy.ndarray')

class Pretrained:
    def __init__(self, test=False, input_size=224):
        self.test = test
        self.T = {
            'train': transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
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


class DetectionTarget:
    # Applies the transform to the target "image_label" if Target is a "detection" kind of label
    def __init__(self, target_transform):
        self.target_transform = target_transform
    def __call__(self, x):
        x['image_label'] = self.target_transform(x['image_label'])
        return x

