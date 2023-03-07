import math

import torch
import torch.nn as nn

import skinbot.skinlogging as logging
from skinbot.utils_models import get_backbone, PlainLayer, get_conv_size, Interpolation, DeconvBlock, \
    implements_flatten, RecoverH1W1C1


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

class ConditionalGaussians(nn.Module):
    def __init__(self, num_classes, latent_dims):
        super(ConditionalGaussians, self).__init__()
        _means = torch.empty(num_classes, latent_dims)
        _means = nn.init.xavier_uniform_(_means, gain=10*num_classes)
        # _means = nn.init.zeros_(_means)
        self.means = nn.Parameter(_means, requires_grad=True)

    def forward(self, z):
        return z.unsqueeze(dim=1)-self.means.unsqueeze(dim=0)
    def __str__(self):
        with torch.no_grad():
            means = self.means.detach().cpu().numpy()
            return f"cond. means = {means}"


class VariationalEncoderConditional(nn.Module):
    def __init__(self, num_inputs, num_classes, latent_dims, layers=None):
        super(VariationalEncoderConditional, self).__init__()
        if layers is not None:
            self.encoder_mlp = nn.Sequential(get_mlp(num_inputs, layers[-1], dropout=0, layers=layers[:-1]), nn.ReLU())
        else:
            self.encoder_mlp = PlainLayer()
        # TODO: 512 is hardcoded
        self.mean_mlp = get_mlp(layers[-1], latent_dims, layers=[], dropout=0)
        self.var_mlp = get_mlp(layers[-1], latent_dims, layers=[], dropout=0)
        self.N = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = torch.tensor(0)
        self.means_y = ConditionalGaussians(num_classes, latent_dims)

    def forward(self, x, y=None):
        x_flat = torch.flatten(x, start_dim=1)
        h = self.encoder_mlp(x_flat)
        mu = self.mean_mlp(h)
        log_var = self.var_mlp(h)
        sigma = torch.exp(0.5 * log_var)
        z = mu + sigma * self.N.sample(mu.shape)
        if y is not None:
            z_prior = self.means_y(z)
            self.kl = 0.5*(z_prior**2 + sigma.unsqueeze(dim=1)**2 - log_var.unsqueeze(dim=1)-1)
            self.kl = torch.bmm(y.unsqueeze(dim=1).float(), self.kl).mean() #.sum(dim=1).mean(dim=0) # .mean(dim=1)
        else:
            self.kl = torch.tensor(0)
        # self.kl = 0.5 * (sigma**2 + mu ** 2 - log_var - 1).sum(dim=1).mean(dim=0)
        return z, mu, log_var


class VariationalEncoder(nn.Module):
    def __init__(self, num_inputs, latent_dims, layers=None):
        super(VariationalEncoder, self).__init__()
        if layers is not None:
            self.encoder_mlp = get_mlp(num_inputs, layers[-1], dropout=0, layers=layers[:-1])
        else:
            self.encoder_mlp = PlainLayer()
        self.mean_mlp = get_mlp(layers[-1], latent_dims, layers=[], dropout=0)
        self.var_mlp = get_mlp(layers[-1], latent_dims, layers=[], dropout=0)
        self.N = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x_flat = torch.flatten(x, start_dim=1)
        h = self.encoder_mlp(x_flat)
        mu = self.mean_mlp(h)
        log_var = self.var_mlp(h)
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
            self.encoder = VariationalEncoderConditional(num_inputs, num_classes, latent_dims, layers=layers)
            layers.reverse()
            self.decoder = Decoder(latent_dims, num_outputs, layers=layers)
            self.conditional = True
        self.preserve_shape = preserve_shape

    def compute_kl(self):
        return self.encoder.kl
    def forward(self, x, y=None):
        shape = x.shape if self.preserve_shape else None
        (z, _, _) = self.encoder(x, y)
        # if y is not None:
        #    z = torch.concat([z, y], dim=1)
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
                 reconstruct_image_features=False,
                 backbone_name='resnet50'):

        self.backbone = get_backbone(backbone_name, None, freeze='Yes', conv_only=True)
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

        self.reconstruct_image_features =   reconstruct_image_features

        if not reconstruct_image_features:
            # needs a reconstruction of original input image.
            # self.global_avg_pool = nn.AdaptiveAvgPool2ddaptive_max_pool2d(output_size=1)

            num_deconv_steps = 3 # todo hardcoded int(C.config['AUTONECODES....
            m0C, m0h, m0w = tuple([3] + list(input_size)) if isinstance(input_size, (list, tuple)) else (3, input_size, input_size)
            B, m1C, m1h, m1w = get_conv_size(self.backbone, input_size)


            # recover layer the H1, W1 C1 from the backbone output that can be flatten or using avg max pool
            self.recover_H1W1C1 = RecoverH1W1C1(num_input_features=num_features_backbone,
                                                 H1=m1h, W1=m1w, C1=m1C,
                                                 from_flatten=implements_flatten(self.backbone))
            m0 = min(m0h, m0w)
            m1 = min(m1h, m1w)
            upsampling_step = math.floor(m0/m1^(1/num_deconv_steps))
            upsampling_modules = []
            num_channels = m1C
            output_size = (m1h,m1w)
            for i in range(num_deconv_steps):
                upsampling_modules.append(DeconvBlock(num_channels, num_channels//2, upsampling_step))
                # update the output calculation
                output_size = (m1h*upsampling_step, m1w*upsampling_step)
                num_channels = num_channels//2
                if num_channels//num_channels < 3:
                    break
            self.deconv = nn.Sequential(*upsampling_modules)
            self.interpolation = Interpolation((m0h, m0w), mode='linear')
            self.output_layer = nn.Conv2d(num_channels, 3, 1)

    def forward(self, x):
        h = self.backbone(x)
        h_hat = self.autoencoder(h)
        if not self.reconstruct_image_features:
           h_hat = self.recover_H1W1C1(h_hat)
           h_hat = self.deconv(h_hat)
           h_hat = self.interpolation(h_hat)
           h_hat = self.output_layer(h_hat)

        return h_hat

