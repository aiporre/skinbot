import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from skinbot.transformers import num_classes

def pretrained_model(model_name, num_outputs, frezze=False, pretrained=True):
    backbone = None
    input_size = 224
    def freeze_model(model):
        for p in model.parameters():
            p.requires_grad = False

    if model_name == 'resnet101':
        backbone = models.resnet101(pretrained=pretrained)
        if frezze:
            frezze_model(backbone)
            num_features = backbone.fc.in_features
            backbone.fc = nn.Linear(num_features, num_outputs)
    else:
        raise Exception(f'model name {model_name} is not defined')
    return backbone

def make_model(model_name):
    if model_name.startswith('resnet'):
        return pretrained_model(model_name, num_outputs=num_classes)
    else:
        raise ValueException(f"Model name {model_name} is not defined.")


