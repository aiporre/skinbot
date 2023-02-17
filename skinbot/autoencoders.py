import torch
import torch.nn as nn

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
    def __init__(self, num_inputs, latent_dims):
        super(Encoder, self).__init__()
        self.encoder_mlp = get_mlp(num_inputs, latent_dims, dropout=0)

    def forward(self, x, y=None):
        x = torch.flatten(x, start_dim=1)
        if y is not None:
            x = torch.concat([x, y], dim=1)
        return self.encoder_mlp(x)


class Decoder(nn.Module):
    def __init__(self, latent_dims, num_outputs):
        super(Decoder, self).__init__()
        self.decoder_mlp = get_mlp(latent_dims, num_outputs, dropout=0)

    def forward(self, z):
        return self.decoder_mlp(z)
class AutoEncoder(nn.Module):
    def __init__(self, num_inputs, num_outputs, latent_dims, num_classes=None):
        super(AutoEncoder, self).__init__()
        if num_classes is not None:
            self.encoder = Encoder(num_inputs, latent_dims)
            self.decoder = Decoder(latent_dims+num_classes, num_outputs)
            self.conditional = False
        else:
            self.encoder = Encoder(num_inputs, latent_dims)
            self.decoder = Decoder(latent_dims, num_outputs)
            self.conditional = True

    def forward(self, x, y=None):
        z = self.encoder(x, y=y)
        return self.decoder(z)



class VariationalEncoder(nn.Module):
    def __init__(self, num_inputs, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.encoder_mlp = get_mlp(num_inputs, 512, dropout=0)
        # TODO: 512 is hardcoded
        self.mean_mlp = get_mlp(512, latent_dims, layers=[], dropout=0)
        self.std_mlp = get_mlp(512, latent_dims, layers=[], dropout=0)
        self.N = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder_mlp(x)
        mu = self.mean_mlp(x)
        sigma = torch.exp(self.std_mlp(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class VariationalAutoEncoder(nn.Module):
    def __init__(self, num_inputs, num_outputs, latent_dims, num_classes=None):
        super(VariationalAutoEncoder, self).__init__()
        if num_classes is not None:
            self.encoder = Encoder(num_inputs, latent_dims)
            self.decoder = Decoder(latent_dims+num_classes, num_outputs)
            self.conditional = False
        else:
            self.encoder = Encoder(num_inputs, latent_dims)
            self.decoder = Decoder(latent_dims+num_classes, num_outputs)
            self.conditional = True

    def compute_kl(self):
        return self.encoder.kl
    def forward(self, x, y=None):
        z = self.encoder(x, y=y)
        return self.decoder(z)
