import torch
from torch import nn as nn
from torchvision import models
import skinbot.skinlogging as logging

def get_mlp(num_inputs, num_outputs, layers=None, dropout=0.5):
    layers = [1024] if layers is None else layers
    instances = []
    for l in layers:
        instances.append(nn.Linear(num_inputs, l))
        instances.append(nn.ReLU())
        if dropout > -1:
            instances.append(nn.Dropout(dropout))
        num_inputs = l
    instances.append(nn.Linear(num_inputs, num_outputs))
    return nn.Sequential(*instances)

class SmallCNN(nn.Module):
    # four layers convolutiona network with input 224x224x3
    def __init__(self, num_classes=2):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding='same')
        self.conv2 = nn.Conv2d(16, 32, 3, padding='same')
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv4 = nn.Conv2d(64, 128, 3, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.num_middle = 128 * 14 * 14
        self.fc = get_mlp(self.num_middle, num_classes, layers=[1024], dropout=0.5)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = self.pool(nn.ReLU()(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
class PlainLayer(nn.Module):
    @staticmethod
    def forward(x):
        return x

def print_trainable_parameters(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            logging.info(n)


def get_backbone(model_name, num_outputs, freeze='No', pretrained=True, conv_only=False):
    def freeze_model(model):
        for p in model.parameters():
            p.requires_grad = False

    def freeze_before_conv(model, last_conv):
        for n, p in model.named_parameters():
            if n.startswith(last_conv):
                break
            p.requires_grad = False

    if model_name == 'resnet101':
        weights = models.ResNet101_Weights.DEFAULT  # if pretrained else None
        T = weights.transforms()
        backbone = models.resnet101(weights=weights)
        if freeze == 'yes':
            freeze_model(backbone)
        elif freeze != 'no':
            freeze_before_conv(backbone, last_conv=freeze)
            logging.info(f"Freezing all layers before {freeze}")
            print_trainable_parameters(backbone)
        num_features = backbone.fc.in_features
        backbone.num_features = num_features
        backbone.fc = PlainLayer() if conv_only else get_mlp(num_features, num_outputs)
    elif model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT  # if pretrained else None
        T = weights.transforms()
        backbone = models.resnet50(weights=weights)
        if freeze == 'yes':
            freeze_model(backbone)
        elif freeze != 'no':
            freeze_before_conv(backbone, last_conv=freeze)
            logging.info(f"Freezing all layers before {freeze}")
            print_trainable_parameters(backbone)
        num_features = backbone.fc.in_features
        backbone.fc = PlainLayer() if conv_only else get_mlp(num_features, num_outputs)  # nn.Linear(num_features, num_outputs)
    elif model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT  # if pretrained else None
        T = weights.transforms()
        backbone = models.resnet18(weights=weights)
        if freeze == 'yes':
            freeze_model(backbone)
        elif freeze != 'no':
            freeze_before_conv(backbone, last_conv=freeze)
            logging.info(f"Freezing all layers before {freeze}")
            print_trainable_parameters(backbone)
        num_features = backbone.fc.in_features
        backbone.num_features = num_features
        backbone.fc = PlainLayer() if conv_only else get_mlp(num_features, num_outputs)  # nn.Linear(num_features, num_outputs)
    elif model_name == 'vgg19':
        weights = models.VGG19_Weights.DEFAULT  # if pretrained else None
        T = weights.transforms()
        backbone = models.vgg19(weights=weights)
        if freeze == 'yes':
            freeze_model(backbone)
        elif freeze != 'no':
            freeze_before_conv(backbone, last_conv=freeze)
            logging.info(f"Freezing all layers before {freeze}")
            print_trainable_parameters(backbone)
        num_features = 512 * 7 * 7  # backbone.classifier.in_features
        backbone.num_features = num_features
        backbone.classifier = PlainLayer() if conv_only else get_mlp(num_features, num_outputs)  # nn.Linear(num_features, num_outputs)
    elif model_name == 'vgg16':
        weights = models.VGG16_Weights.DEFAULT  # if pretrained else None
        T = weights.transforms()
        backbone = models.vgg16(weights=weights)
        if freeze == 'yes':
            freeze_model(backbone)
        elif freeze != 'no':
            freeze_before_conv(backbone, last_conv=freeze)
            logging.info(f"Freezing all layers before {freeze}")
            print_trainable_parameters(backbone)
        num_features = 512 * 7 * 7  # backbone.classifier.in_features
        backbone.num_features = num_features
        backbone.classifier = PlainLayer() if conv_only else get_mlp(num_features, num_outputs)  # nn.Linear(num_features, num_outputs)
    elif model_name.lower() == 'smallcnn':
        backbone = SmallCNN(num_classes=num_outputs)
        backbone.num_features = backbone.num_middle
        if conv_only:
            backbone.fc = PlainLayer()
    else:
        raise Exception(f'model name {model_name} is not defined')
    return backbone