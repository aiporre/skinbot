from functools import reduce
import os
import skinbot.skinlogging as logging

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from skinbot.config import Config
from skinbot.segmentation import UNet
from skinbot.autoencoders import VariationalAutoEncoder, AutoEncoder, ConvolutionalAutoEncoder, AutoEncoderClassifier
from skinbot.utils_models import get_backbone, freeze_model, freeze_before_conv

C = Config()


def classification_model(model_name, num_outputs, freeze='No', pretrained=True, ae_model_path=None):
    freeze = freeze.lower()
    backbone = None
    input_size = 224
    if model_name == 'aec':
        # load an autoencoder model
        assert ae_model_path is not None, 'You need to provide a path to the autoencoder model'
        # get model name thrid element of ae_model_path separated by _ e.g. 'best_fold=0_convae_reconstruction_all_model_negval=-0.0379.pt
        convae_model_name = os.path.basename(ae_model_path).split('_')[2]
        logging.info(f"the autoencoder backbonoe detected is: {convae_model_name}:")
        backbone_autoencoder = autoencoder_model(convae_model_name, num_outputs, freeze='No')
        # load the autoencoder model from ae_model_path
        backbone_autoencoder.load_state_dict(torch.load(ae_model_path))
        # freeze yes, no or before
        if freeze == 'yes':
            backbone_autoencoder = freeze_model(backbone_autoencoder)
        elif freeze == 'no':
            pass
        else:
            backbone_autoencoder = freeze_before_conv(backbone_autoencoder, freeze)
        model = AutoEncoderClassifier(backbone_autoencoder, num_outputs)
        return model
    else:
        return get_backbone(model_name, num_outputs, freeze='No', pretrained=True)


def detection_model(model_name, num_classes, pretrained=True):
    if model_name == 'fasterresnet50':
        # load an object detection model Vj
        weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif model_name == 'fastermobilenet':
        # load an object detection model Vj
        weights = models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights, box_score_thresh=0.9)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif model_name == 'maskrcnn':
        # load an object detection model Vj
        weights = models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        dim_reduced_mask = model.roi_heads.mask_predictor.conv5_mask.out_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced_mask, num_classes)
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        raise ValueError(
            f'model_name= {model_name} is not a valid option. Try fastermobilenet, fasterresnet50 or maskrcnn')
    return model


def segmentation_model(model_name, num_classes, freeze='No', learnable_upsample=True):
    model = UNet(in_channels=3, num_classes=num_classes, learnable_upsample=learnable_upsample)
    return model


def autoencoder_model(model_name, num_classes, freeze='No'):
    num_inputs = eval(C.config['AUTOENCODER']['num_inputs'])
    num_outputs = eval(C.config['AUTOENCODER']['num_outputs'])
    latent_dims = int(C.config['AUTOENCODER']['latent_dims'])
    layers = eval(C.config['AUTOENCODER']['layers'])
    preserve_shape = bool(C.config['AUTOENCODER']['preserve_shape'])
    backbone_name = C.config['AUTOENCODER'].get('backbone', None)
    reconstruct_image = False # initialization changes based on the num_output being a list or an int.
    if isinstance(num_inputs, (list, tuple)) and isinstance(num_outputs, (list, tuple)):
        assert len(num_inputs) == len(num_outputs) and len(num_inputs) < 4, \
            f'Wrong config.ini. Num_inputs must be the same a num_outputs.' \
            f' And max only three. Given  inputs {num_inputs},' \
            f'and outputs {num_outputs}'
        convolutional = True
    elif isinstance(num_inputs, int) and isinstance(num_outputs, int):
        assert num_inputs == num_outputs, \
            f'Wrong config.ini. Num_inputs must be the same a num_outputs. ' \
            f'And max only three. Given inputs {num_inputs}' \
            f'and outputs {num_outputs}'
        convolutional = False
        reconstruct_image = True
    elif isinstance(num_inputs, (list, tuple)) and isinstance(num_outputs, int):
        assert model_name.startswith(
            'conv'), " if num input has more than one dimension you should use a conv version of the autoencoder"
        if reduce((lambda x, y: x * y), num_inputs) == num_outputs:
            logging.warn(
                'Wrong config.ini. Num of input must be a same number of elements as output? you are trying to '
                'generate a feature space. That is fine.')
        convolutional = True
        reconstruct_image = False
    else:
        raise ValueError(
            f'Wrong inp[uts in the config.ini num_inputs na num_outs, try same int int or tuple tuple or list list')
        # infers if convolutional activativate

    if convolutional:

        if model_name == 'convae':
            varational=False
            _num_classes =None
        elif model_name == 'convvae':
            varational = True
            _num_classes =None
        elif model_name == 'convcvae':
            varational = True
            _num_classes = num_classes
        else:
            raise ValueError(f"model_name = {model_name} is not defined. Options: convAE, convVAE or convCVAE.")
        model = ConvolutionalAutoEncoder(
            num_inputs=num_inputs, num_outputs=num_outputs, latent_dims=latent_dims, num_classes=num_classes,
            layers=layers, backbone_name=backbone_name,
            preserve_shape=False, varational=varational, reconstruct_image_features=reconstruct_image
        )
    else:
        if model_name == 'ae':
            model = AutoEncoder(num_inputs=num_inputs, num_outputs=num_outputs, latent_dims=latent_dims, layers=layers,
                                preserve_shape=preserve_shape)
        elif model_name == 'vae':
            model = VariationalAutoEncoder(num_inputs=num_inputs, num_outputs=num_outputs, latent_dims=latent_dims,
                                           layers=layers, preserve_shape=preserve_shape)
        elif model_name == 'cvae':
            model = AutoEncoder(num_inputs=num_inputs, num_outputs=num_outputs, num_classes=num_classes,
                                latent_dims=latent_dims, layers=layers, preserve_shape=preserve_shape)
        else:
            # model_name is then CVAE
            # model = VariationalAutoEncoder(num_inputs=num_inputs, num_outputs=num_outputs, num_classes=num_classes,
            #                                latent_dims=latent_dims, layers=layers, preserve_shape=preserve_shape)
            raise ValueError(f"model_name = {model_name} is not defined. Options: ae, vae cae.")
    return model


def get_model(model_name, optimizer=None, lr=0.001, momentum=0.8, freeze='No', **kwargs):
    model_name = model_name.lower()
    if model_name.startswith('resnet') or model_name.startswith('vgg') or model_name == 'smallcnn' or model_name == 'aec':
        model = classification_model(model_name, num_outputs=C.labels.num_classes, freeze=freeze, **kwargs)
    elif 'faster' in model_name or 'mask' in model_name:
        model = detection_model(model_name, C.labels.num_classes)
    elif model_name == 'unet':
        model = segmentation_model(model_name, num_classes=C.labels.num_classes, freeze=freeze, **kwargs)
    elif model_name in ['vae', 'ae', 'cvae', 'convvae', 'convae', 'convcvae']:
        model = autoencoder_model(model_name, num_classes=C.labels.num_classes, freeze=freeze, **kwargs)
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
