import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from skinbot.transformers import num_classes

def classification_model(model_name, num_outputs, freeze=False, pretrained=True):
    backbone = None
    input_size = 224
    def freeze_model(model):
        for p in model.parameters():
            p.requires_grad = False
    def get_mlp(num_inputs, num_outputs, layers=None, dropout=0.5):
        layers = [1024] if layers is None else layers
        instances = []
        for l in layers:
            instances.append(nn.Linear(num_inputs, l))
            instances.append(nn.ReLU())
            if dropout > 0:
                instances.append(nn.Dropout(dropout))
            num_inputs = l
        instances.append(nn.Linear(num_inputs, num_outputs))
        return nn.Sequential(*instances)

    if model_name == 'resnet101':
        weights = models.ResNet101_Weights.DEFAULT # if pretrained else None
        T = weights.transforms()
        backbone = models.resnet101(weights=weights)
        if freeze:
            freeze_model(backbone)
        num_features = backbone.fc.in_features
        backbone.fc = get_mlp(num_features, num_outputs) #nn.Linear(num_features, num_outputs)
    else:
        raise Exception(f'model name {model_name} is not defined')
    return backbone

def detection_model(model_name, num_classes, pretrained=True):
    # load an object detection model pre-trained on COCO
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_model(model_name, optimizer=None, lr=0.001, momentum=0.8, freeze=False):
    if model_name.startswith('resnet'):
        model = classification_model(model_name, num_outputs=num_classes, freeze=freeze)
    elif model_name == 'faster_rcnn_resnet50_fpn':
        model = detection_model(model_name, num_classes)
    else:
        raise Exception(f"Model name {model_name} is not defined.")
    if optimizer is not None:
        if freeze:
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



