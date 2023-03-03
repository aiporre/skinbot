import torch
import torch.nn as nn

import skinbot.skinlogging as logging
from skinbot.utils_models import get_backbone, PlainLayer


def get_mlp(num_inputs, num_outputs, layers=None, dropout=0.5):
    layers = [1024] if layers is None else layers
    instances = []
    for l in layers:
        instances.append(nn.Linear(num_inputs, l))
        instances.append(nn.ReLU())
        if dropout is not None and dropout>0:
            instances.append(nn.Dropout(dropout))
        num_inputs = l
    instances.append(nn.Linear(num_inputs, num_outputs))
    return nn.Sequential(*instances)


class Encoder(nn.Module):
    def __init__(self, num_inputs, latent_dims, layers=None):
        super(Encoder, self).__init__()
        self.encoder_mlp = get_mlp(num_inputs, latent_dims, dropout=0, layers=layers)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.encoder_mlp(x)


class Decoder(nn.Module):
    def __init__(self, latent_dims, num_outputs, layers=None):
        super(Decoder, self).__init__()
        self.decoder_mlp = get_mlp(latent_dims, num_outputs, dropout=0, layers=layers)

    def forward(self, z, shape=None):
        if shape is None:
            return self.decoder_mlp(z)
        else:
            return torch.reshape(self.decoder_mlp(z), shape)
class AutoEncoder(nn.Module):
    def __init__(self, num_inputs, num_outputs, latent_dims, num_classes=None, layers=None, preserve_shape=False):
        super(AutoEncoder, self).__init__()
        if num_classes is None:
            self.encoder = Encoder(num_inputs, latent_dims, layers=layers)
            self.decoder = Decoder(latent_dims, num_outputs, layers=layers)
            self.conditional = False
        else:
            self.encoder = Encoder(num_inputs, latent_dims, layers=layers)
            self.decoder = Decoder(latent_dims+num_classes, num_outputs, layers=layers)
            self.conditional = True
        self.preserve_shape = preserve_shape

    def forward(self, x, y=None):
        shape = x.shape if self.preserve_shape else None
        z = self.encoder(x)
        if y is not None:
            z = torch.concat([z, y], dim=1)
        return self.decoder(z, shape=shape)



class VariationalEncoder(nn.Module):
    def __init__(self, num_inputs, latent_dims, layers=None):
        super(VariationalEncoder, self).__init__()
        if layers is not None:
            self.encoder_mlp = get_mlp(num_inputs, layers[-1], dropout=0, layers=layers[:-1])
        else:
            self.encoder_mlp = PlainLayer()
        # TODO: 512 is hardcoded
        self.mean_mlp = get_mlp(layers[-1], latent_dims, layers=[], dropout=0)
        self.var_mlp = get_mlp(layers[-1], latent_dims, layers=[], dropout=0)
        self.N = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder_mlp(x)
        mu = self.mean_mlp(x)
        log_var = self.var_mlp(x)
        sigma = torch.exp(0.5 * log_var)
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = 0.5 * (sigma**2 + mu ** 2 - log_var - 1).sum(dim=1).mean(dim=0)
        return z, mu, log_var


class VariationalAutoEncoder(nn.Module):
    def __init__(self, num_inputs, num_outputs, latent_dims, num_classes=None, layers=None, preserve_shape=False):
        super(VariationalAutoEncoder, self).__init__()
        assert len(layers) > 0, 'Variational autoencoder need layers to have at least one dimension'

        if num_classes is None:
            self.encoder = VariationalEncoder(num_inputs, latent_dims, layers=layers)
            layers.reverse()
            self.decoder = Decoder(latent_dims, num_outputs, layers=layers)
            self.conditional = False
        else:
            self.encoder = VariationalEncoder(num_inputs, latent_dims, layers=layers)
            layers.reverse()
            self.decoder = Decoder(latent_dims+num_classes, num_outputs, layers=layers)
            self.conditional = True
        self.preserve_shape = preserve_shape

    def compute_kl(self):
        return self.encoder.kl
    def forward(self, x, y=None):
        shape = x.shape if self.preserve_shape else None
        (z, _, _) = self.encoder(x)
        if y is not None:
            z = torch.concat([z, y], dim=1)
        return self.decoder(z, shape=shape)

class ConvolutionalAutoEncoder(nn.Module):

    # def __init__(self, num_inputs, num_outputs, latent_dims, num_classes=None, layers=None, preserve_shape=False):
    def __init__(self, input_size,
                 num_inputs,
                 num_outputs,
                 latent_dims,
                 num_classes=None,
                 layers=None,
                 preserve_shape=False,
                 varational=False,
                 reconstruct_image_features=False):

        self.backbone = get_backbone('resnet50', None, freeze='Yes', conv_only=True)
        num_features_backbone = self.backbone.num_features
        if varational:
            self.autoencoder = AutoEncoder(num_inputs=num_features_backbone,
                                           num_outputs=num_outputs,
                                           latent_dims=latent_dims,
                                           num_classes=num_classes,
                                           layers=layers,
                                           preserve_shape=preserve_shape)
        else:
            self.autoencoder = VariationalAutoEncoder(
                                       num_inputs=num_features_backbone,
                                       num_outputs=num_outputs,
                                       latent_dims=latent_dims,
                                       num_classes=num_classes,
                                       layers=layers,
                                       preserve_shape=preserve_shape)

        if not reconstruct_image_features:
            # needs a reconstruction of original input image.
            # self.deconvolution = Deconvolution(num_inputs, output_size=input_size)
            raise Exception("Not implemente decovntion to reconsvtruio image")
