import numpy as np

target_str_to_num = {
    'contact' :  0,
    'vasculitis' :  1,
    'necrosis' :  2,
    'malignant' :  3,
    'pyoderma': 4, 
    'vaskulitis' :  5,
    'dermatitis' :  6,
    'infection' :  7,
    'bland' : 8
}

class TargetOneHot:
    def __init__(self):
        self.target_set_num = len(target_str_to_num)
    def __call__(self, x):
        x = x.strip().lower()
        x_transform = np.zeros(self.target_set_num)
        x_transform[target_str_to_num[x]] = 1.
        return x_transform

class TargetValue:
    def __call__(self, x):
        x = x.strip().lower()
        return target_str_to_num[x]


