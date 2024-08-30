from collections import OrderedDict

import torch
from torch import nn as nn
from torchvision import models
import skinbot.skinlogging as logging
from skinbot.config import Config
import math

from skinbot.segmentation import pad_to_match

C = Config()


def get_mlp(num_inputs, num_outputs, layers=None, dropout=0.1):
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

def get_output_size(model, input_size):
    B = 1 # batch_size
    if isinstance(input_size, int):
        input_size_1 = (B , 3, input_size, input_size)
    else:
        input_size_1 = [B] + list(input_size)
    x = torch.randn(input_size_1)
    y = model(x)
    return tuple(y.shape)

def get_conv_size(model, input_size):
    B = 1 # batch_size
    if isinstance(input_size, int):
        input_size = (B , 1, input_size, input_size)
    x = torch.randn(input_size)
    from collections import OrderedDict
    activation = OrderedDict()

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach().shape

        return hook
    for name, module in model.named_modules():
        if hasattr(module, 'forward'):
            module.register_forward_hook(get_activation(name))
    # make a batch
    x = x.unsqueeze(0)
    y = model(x)
    last_conv_output = None
    last_conv_output_name = None
    for a_name in reversed(activation):
        a = activation[a_name]
        if len(a) == 4 and a[-1]>1 and a[-2]>1:
            last_conv_output = a
            last_conv_output_name = a_name
            break
    if last_conv_output is None:
        raise RuntimeError(f'Cannot find shape of 4 dimension (batch, channels, H, W).')
    print(activation)
    return last_conv_output_name, tuple(last_conv_output)

class SmallCNN(nn.Module):
    # four layers convolutiona network with input 224x224x3
    def __init__(self, num_classes=2):
        super(SmallCNN, self).__init__()
        conv_layers_dims = eval(C.config['MODELS']['conv_layers'])
        input_size = eval(C.config['MODELS']['input_size'])
        self.use_global_pool = eval(C.config['MODELS']['use_global_pool'])

        if isinstance(input_size, tuple):
            in_channels = input_size[0]
        else:
            in_channels = 3
        modules = []
        for conv_dim in conv_layers_dims:
            componets = OrderedDict()
            componets['conv'] = nn.Conv2d(in_channels, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)
            componets['batchnorm'] = nn.BatchNorm2d(conv_dim)
            componets['relu'] = nn.LeakyReLU()
            componets['maxpool'] = nn.MaxPool2d(2, 2)
            modules.append(
                nn.Sequential(componets
                    )
            )
            in_channels = conv_dim
        backbone_componets = OrderedDict()
        for i, m in enumerate(modules):
            backbone_componets[f'layer{i}'] = m
        self.backbone = nn.Sequential(backbone_componets)
        if not self.use_global_pool:
            self.num_middle = math.prod(get_output_size(self.backbone, input_size))
        else:
            self.num_middle = conv_layers_dims[-1]

        fc_layers_dims = eval(C.config['MODELS']['fc_layers'])
        if len(fc_layers_dims) > 0:
            self.fc = get_mlp(self.num_middle, num_classes, layers=fc_layers_dims, dropout=0.2)
        else:
            self.fc = PlainLayer()


    def forward(self, x):
        x = self.backbone(x)
        if not self.use_global_pool:
            x = torch.flatten(x, 1)
        else:
            x = nn.functional.adaptive_max_pool2d(x, output_size=1)
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


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


def freeze_before_conv(model, last_conv):
    for n, p in model.named_parameters():
        if n.startswith(last_conv):
            break
        p.requires_grad = False
    return model


def get_backbone(model_name, num_outputs, freeze='No', pretrained=True, conv_only=False, layers=None):

    if model_name == 'resnet101':
        weights = models.ResNet101_Weights.DEFAULT  # if pretrained else None
        T = weights.transforms()
        backbone = models.resnet101(weights=weights)
        if freeze == 'yes':
            freeze_model(backbone)
        elif freeze != 'no':
            backbone = freeze_before_conv(backbone, last_conv=freeze)
            logging.info(f"Freezing all layers before {freeze}")
            print_trainable_parameters(backbone)
        num_features = backbone.fc.in_features
        backbone.num_features = num_features
        backbone.fc = PlainLayer() if conv_only else get_mlp(num_features, num_outputs, layers=layers)
    elif model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT  # if pretrained else None
        T = weights.transforms()
        backbone = models.resnet50(weights=weights)
        if freeze == 'yes':
            backbone = freeze_model(backbone)
        elif freeze != 'no':
            backbone = freeze_before_conv(backbone, last_conv=freeze)
            logging.info(f"Freezing all layers before {freeze}")
            print_trainable_parameters(backbone)
        num_features = backbone.fc.in_features
        backbone.num_features = num_features
        backbone.fc = PlainLayer() if conv_only else get_mlp(num_features, num_outputs, layers=layers)  # nn.Linear(num_features, num_outputs)
    elif model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT  # if pretrained else None
        T = weights.transforms()
        backbone = models.resnet18(weights=weights)
        if freeze == 'yes':
            backbone = freeze_model(backbone)
        elif freeze != 'no':
            backbone = freeze_before_conv(backbone, last_conv=freeze)
            logging.info(f"Freezing all layers before {freeze}")
            print_trainable_parameters(backbone)
        num_features = backbone.fc.in_features
        backbone.num_features = num_features
        backbone.fc = PlainLayer() if conv_only else get_mlp(num_features, num_outputs, layers=layers)  # nn.Linear(num_features, num_outputs)
    elif model_name == 'vgg19':
        weights = models.VGG19_Weights.DEFAULT  # if pretrained else None
        T = weights.transforms()
        backbone = models.vgg19(weights=weights)
        if freeze == 'yes':
            backbone = freeze_model(backbone)
        elif freeze != 'no':
            backbone = freeze_before_conv(backbone, last_conv=freeze)
            logging.info(f"Freezing all layers before {freeze}")
            print_trainable_parameters(backbone)
        num_features = 512 * 7 * 7  # backbone.classifier.in_features
        backbone.num_features = num_features
        backbone.classifier = PlainLayer() if conv_only else get_mlp(num_features, num_outputs)  # nn.Linear(num_features, num_outputs)
    elif model_name == 'efficientnet-b0':
        weights = models.EfficientNetB0_Weights.DEFAULT
        T = weights.transforms()
        backbone = models.efficientnet_b0(weights=weights)
        if freeze == 'yes':
            backbone = freeze_model(backbone)
        elif freeze != 'no':
            backbone = freeze_before_conv(backbone, last_conv=freeze)
            logging.info(f"Freezing all layers before {freeze}")
            print_trainable_parameters(backbone)
        num_features = backbone._fc.in_features
        backbone.num_features = num_features
        backbone._fc = PlainLayer() if conv_only else get_mlp(num_features, num_outputs)  # nn.Linear(num_features, num_outputs)
    elif model_name == 'efficientnet-b1':
        weights = models.EfficientNetB1_Weights.DEFAULT
        T = weights.transforms()
        backbone = models.efficientnet_b1(weights=weights)
        if freeze == 'yes':
            backbone = freeze_model(backbone)
        elif freeze != 'no':
            backbone = freeze_before_conv(backbone, last_conv=freeze)
            logging.info(f"Freezing all layers before {freeze}")
            print_trainable_parameters(backbone)
        num_features = backbone._fc.in_features
        backbone.num_features = num_features
        backbone._fc = PlainLayer() if conv_only else get_mlp(num_features, num_outputs)
    elif model_name == 'efficientnet-b2':
        weights = models.EfficientNetB2_Weights.DEFAULT
        T = weights.transforms()
        backbone = models.efficientnet_b2(weights=weights)
        if freeze == 'yes':
            backbone = freeze_model(backbone)
        elif freeze != 'no':
            backbone = freeze_before_conv(backbone, last_conv=freeze)
            logging.info(f"Freezing all layers before {freeze}")
            print_trainable_parameters(backbone)
        num_features = backbone._fc.in_features
        backbone.num_features = num_features
        backbone._fc = PlainLayer() if conv_only else get_mlp(num_features, num_outputs)

    elif model_name == 'vgg16':
        weights = models.VGG16_Weights.DEFAULT  # if pretrained else None
        T = weights.transforms()
        backbone = models.vgg16(weights=weights)
        if freeze == 'yes':
            backbone = freeze_model(backbone)
        elif freeze != 'no':
            backbone = freeze_before_conv(backbone, last_conv=freeze)
            logging.info(f"Freezing all layers before {freeze}")
            print_trainable_parameters(backbone)
        num_features = 512 * 7 * 7  # backbone.classifier.in_features
        backbone.num_features = num_features
        backbone.classifier = PlainLayer() if conv_only else get_mlp(num_features, num_outputs, layers=layers)  # nn.Linear(num_features, num_outputs)
    elif model_name.lower() == 'smallcnn':
        backbone = SmallCNN(num_classes=num_outputs)
        backbone.num_features = backbone.num_middle
        if conv_only:
            backbone.fc = PlainLayer()
    else:
        raise Exception(f'model name {model_name} is not defined')
    return backbone


class Interpolation(nn.Module):
    def __init__(self, size, mode):
        super(Interpolation, self).__init__()
        self.size = size
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(x, self.size, mode=self.mode)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsampling_step):
        super(DeconvBlock, self).__init__()
        # Hout = (Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        # Hout = (Hin−1)×stride[0]+output_padding[0]+1
        # if isinstance(input_size, int):
        #     H, W = input_size, input_size
        # else:
        #     H, W = input_size
        # max_iters = 10
        # iters = 0

        # def error(s,a,b):
        #     return (-s - 2*a + b + 1) != 0 or a<0 or b<0

        # epsilon = 0
        # padding = 0
        # output_padding = 0
        # while error(upsampling_step, padding, output_padding) and iters<max_iters:
        #     iters +=1
        #     padding = epsilon + 1 - upsampling_step
        #     output_padding = 2*epsilon + 1 - upsampling_step
        #     epsilon +=1
        # if iters>=max_iters and not error(upsampling_step, padding, output_padding):
        #     raise RuntimeError('Cannot find padding and output_padding combinations')

        # self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=upsampling_step,
        #                               padding=padding, output_padding=output_padding)
        self.upsample = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=upsampling_step)
        self.upsample_step = upsampling_step

    def forward(self, x):
        H_expected, W_expected = x.shape[-2]*self.upsample_step, x.shape[-1]*self.upsample_step
        return pad_to_match(self.upsample(x), H_expected, W_expected)

def implements_flatten(model):
    if isinstance(model, models.ResNet):
        return False
    elif isinstance(model, SmallCNN):
        return not model.use_global_pool
    elif isinstance(model, models.VGG):
        return True
    else:
        raise ValueError(f"model {type(model)} is not recognized.")


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view( [-1] + list(self.shape))
class RecoverH1W1C1(nn.Module):
    def __init__(self, num_input_features, H1, W1, C1, from_flatten):
        super(RecoverH1W1C1, self).__init__()
        if not from_flatten:
            modules = OrderedDict()
            modules['fc_h1w1_recover'] = nn.Linear(num_input_features, H1*W1)
            modules['reshape'] = Reshape((1,H1,W1))
            modules['conv_c1_recover'] = nn.Conv2d(1, C1, 1)
            self.recover = nn.Sequential(modules)
        else:
            self.recover = Reshape((C1,H1,W1))

    def forward(self, x):
        return self.recover(x)
