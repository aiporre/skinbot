import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from skinbot.transformers import num_classes

def pretrained_model(model_name, num_outputs, freeze=False, pretrained=True):
    backbone = None
    input_size = 224
    def freeze_model(model):
        for p in model.parameters():
            p.requires_grad = False

    if model_name == 'resnet101':
        backbone = models.resnet101(pretrained=pretrained)
        if freeze:
            freeze_model(backbone)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Linear(num_features, num_outputs)
    else:
        raise Exception(f'model name {model_name} is not defined')
    return backbone

def get_model(model_name, optimizer=None, lr=0.001, momentum=0.8, frezze=False):
    if model_name.startswith('resnet'):
        model = pretrained_model(model_name, num_outputs=num_classes, freeze=frezze)
    else:
        raise Exception(f"Model name {model_name} is not defined.")
    if optimizer is not None:
        if frezze:
            model_parameters = []
            for n, p in model.named_parameters():
                if p.requires_grad:
                    model_parameters.append(p)
        else:
            model_parameters = model.parameters()

        if optimizer == 'SGD':
            model_optimizer = torch.optim.SGD(model_parameters, lr=lr, momentum=momentum)
        else:
            raise Exception(f"optimizer name {optimizer} not defined")
        return model, model_optimizer
    else:
        return model



