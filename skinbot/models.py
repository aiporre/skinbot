import skinbot.skinlogging as logging

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from skinbot.config import Config
from skinbot.segmentation import UNet

C = Config()


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
        x= self.fc(x)
        return x

class PlainLayer(nn.Module):
    @staticmethod
    def forward(x):
        return x

def print_trainable_parameters(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            logging.info(n)

def classification_model(model_name, num_outputs, freeze='No', pretrained=True):
    freeze = freeze.lower()
    backbone = None
    input_size = 224
    def freeze_model(model):
        for p in model.parameters():
            p.requires_grad = False
    def freeze_before_conv(model, last_conv):
        for n, p in model.named_parameters():
            if n.startswith(last_conv):
                break
            p.requires_grad = False
    if model_name == 'resnet101':
        weights = models.ResNet101_Weights.DEFAULT # if pretrained else None
        T = weights.transforms()
        backbone = models.resnet101(weights=weights)
        if freeze == 'yes':
            freeze_model(backbone)
        elif freeze != 'no':
            freeze_before_conv(backbone, last_conv=freeze)
            logging.info(f"Freezing all layers before {freeze}")
            print_trainable_parameters(backbone)
        num_features = backbone.fc.in_features
        backbone.fc = get_mlp(num_features, num_outputs) #nn.Linear(num_features, num_outputs)
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
        backbone.fc = get_mlp(num_features, num_outputs)  # nn.Linear(num_features, num_outputs)
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
        backbone.fc = get_mlp(num_features, num_outputs)  # nn.Linear(num_features, num_outputs)
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
        backbone.classifier = get_mlp(num_features, num_outputs)  # nn.Linear(num_features, num_outputs)
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
        backbone.classifier = get_mlp(num_features, num_outputs)  # nn.Linear(num_features, num_outputs)
    elif model_name.lower() == 'smallcnn':
        backbone = SmallCNN(num_classes=num_outputs)
    else:
        raise Exception(f'model name {model_name} is not defined')
    return backbone

def detection_model(model_name, num_classes, pretrained=True):
    # load an object detection model pre-trained on COCO
    weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def segmentation_model(model_name, num_classes, freeze='No', learnable_upsample=True):
    model = UNet(in_channels=3, num_classes=num_classes, learnable_upsample=learnable_upsample)
    return model

def get_model(model_name, optimizer=None, lr=0.001, momentum=0.8, freeze='No', **kwargs):
    model_name = model_name.lower()
    if model_name.startswith('resnet') or model_name.startswith('vgg') or model_name == 'smallcnn':
        model = classification_model(model_name, num_outputs=C.labels.num_classes, freeze=freeze)
    elif model_name == 'faster_rcnn_resnet50_fpn':
        model = detection_model(model_name, C.labels.num_classes)
    elif model_name == 'unet':
        model = segmentation_model(model_name, num_classes=C.labels.num_classes, freeze=freeze, **kwargs)
    else:
        raise Exception(f"Model name {model_name} is not defined.")
    if optimizer is not None:
        if freeze != 'No':
            model_parameters = []
            for n, p in model.named_parameters():
                if p.requires_grad:
                    model_parameters.append(p)
        else:
            model_parameters = model.parameters()

        if optimizer == 'SGD':
            model_optimizer = torch.optim.SGD(model_parameters, lr=lr, momentum=momentum)
        elif optimizer == 'ADAM':
            model_optimizer = torch.optim.Adam(model_parameters, lr=lr, betas=(0.9, 0.999), eps=1e-08)
        else:
            raise Exception(f"optimizer name {optimizer} not defined")
        return model, model_optimizer
    else:
        return model



