import os
import pickle

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from FrEIA import framework as fr
from math import exp
from torch.nn.utils.parametrizations import spectral_norm

class DLRom(nn.Module):
    def __init__(self, z_dim, mu_dim, time_dim=1, h_dim=256):
        super(DLRom, self).__init__()

        self.fc1 = nn.Linear(mu_dim + time_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        z = self.fc3(x)
        return z

class StepRom(nn.Module):
    def __init__(self, z_dim, mu_dim, h_dim=256):
        super(StepRom, self).__init__()

        self.fc1 = nn.Linear(mu_dim + z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        z = self.fc3(x)
        return z

class AEIdentity(nn.Module):
    def forward(self, x):
        return x, x

    def encode(self, x):
        return x

    def decode(self, z):
        return z


class ConvAE(nn.Module):
    def __init__(self,  x_dim=256, z_dim=50, mu_dim=2, h_dim=256):
        super(ConvAE, self).__init__()

        self.encoder = ConvEncoder(x_dim=x_dim, z_dim=z_dim, h_dim=h_dim)
        self.decoder = ConvDecoder(x_dim=x_dim, z_dim=z_dim)

        #self.df = DF(mu_dim=mu_dim, z_dim=z_dim)

        self.x_dim = x_dim
        self.x_dim_reshaped = int(np.sqrt(x_dim // 2))
        self.z_dim = z_dim
        self.mu_dim = mu_dim
        self.h_dim = h_dim

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        #z_n = self.df(mu_t)
        #x_recon_n = self.decode(z_n)
        return x_recon, z, #x_recon_n, z_n

    def encode(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 2, self.x_dim_reshaped, self.x_dim_reshaped)
        return self.encoder(x)

    def decode(self, z):
        batch_size = z.shape[0]
        x_recon = self.decoder(z)
        x_recon = x_recon.reshape(batch_size, self.x_dim)
        return x_recon

    def save_config(self, path):
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        with open(path, 'wb') as handle:
            config = {'x_dim': self.x_dim, 'z_dim': self.z_dim, 'mu_dim': self.mu_dim, 'h_dim': self.h_dim}
            pickle.dump(config, handle)

    @classmethod
    def load_config(cls, path):
        with open(path, 'rb') as handle:
            config = pickle.load(handle)
            return cls(**config)



class ConvEncoder(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim=256):
        super(ConvEncoder, self).__init__()

        self.conv1 = nn.Conv2d(2, 32, (3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.batch1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.batch2 = nn.BatchNorm2d(32)
        out_dim = 8
        self.fc = nn.Linear(32 * out_dim * out_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, z_dim)

    def encoder(self, x):
        #x = x.unsqueeze(dim=1)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.batch1(x)
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.batch2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.elu(self.fc(x))
        x = self.fc1(x)
        return x

    def forward(self, x):
        return self.encoder(x)

class ConvDecoder(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(ConvDecoder, self).__init__()

        # decoder part
        out_dim = 8
        self.fcz = nn.Linear(z_dim, 32 * out_dim * out_dim)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, out_dim, out_dim))
        self.deconv1 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.batch3 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.deconv4 = nn.ConvTranspose2d(32, 2, (3, 3), stride=(1, 1))#, output_padding=(1, 1))
        self.batch4 = nn.BatchNorm2d(32)

    def decoder(self, z):
        z = F.elu(self.fcz(z))
        z = self.unflatten(z)
        z = self.batch3(z)
        z = F.elu(self.deconv1(z))
        z = F.elu(self.deconv2(z))
        z = self.batch4(z)
        z = F.elu(self.deconv3(z))
        x = self.deconv4(z)
       # x = x.squeeze(dim=1)
        return x

    def forward(self, x):
        return self.decoder(x)
class DF(nn.Module):
    def __init__(self, mu_dim=2, z_dim=50, h_dim=256):
        super(DF, self).__init__()

        self.fc1 = nn.Linear(mu_dim + 1, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        z = self.fc3(x)
        return z

class EncoderMLP(nn.Module):
    def __init__(self, x_dim=256, z_dim=50, h_dim=512):
        super(EncoderMLP, self).__init__()

        self.fc = nn.Linear(x_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, z_dim)

    def encoder(self, x):
        x = F.leaky_relu(self.fc(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        z = self.fc4(x)
        return z

    def forward(self, x):
        return self.encoder(x)

class DecoderMLP(nn.Module):
    def __init__(self, x_dim=256, z_dim=20, h_dim=512):
        super(DecoderMLP, self).__init__()

        # decoder part
        self.fc = nn.Linear(z_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, x_dim)

        torch.nn.init.zeros_(self.fc4.weight)

    def decoder(self, z):
        x = F.leaky_relu(self.fc(z))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x
    def forward(self, x):
        return self.decoder(x)

class AEMLP(nn.Module):
    def __init__(self, x_dim=256, z_dim=50, mu_dim=2, h_dim=256):
        super(AEMLP, self).__init__()

        self.encoder = EncoderMLP(x_dim=x_dim, z_dim=z_dim, h_dim=h_dim)
        self.decoder = DecoderMLP(x_dim=x_dim, z_dim=z_dim, h_dim=h_dim)
        #self.df = DF(mu_dim=mu_dim, z_dim=z_dim, h_dim=256)

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.mu_dim = mu_dim
        self.h_dim = h_dim

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        #z_n = self.df(mu_t)
        #x_recon_n = self.decode(z_n)
        return x_recon, z, #x_recon_n, z_n

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def save_config(self, path):
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        with open(path, 'wb') as handle:
            config = {'x_dim': self.x_dim, 'z_dim': self.z_dim, 'mu_dim': self.mu_dim, 'h_dim': self.h_dim}
            pickle.dump(config, handle)

    @classmethod
    def load_config(cls, path):
        with open(path, 'rb') as handle:
            config = pickle.load(handle)
            return cls(**config)

class AE(nn.Module):
    def __init__(self, x_dim=256, channels=1, z_dim=50, mu_dim=2, h_dim=256, kernel_size='large', nonlinearity='elu'):
        super(AE, self).__init__()

        self.encoder = Encoder(x_dim=x_dim, channels=channels, z_dim=z_dim, h_dim=h_dim, kernel_size=kernel_size, nonlinearity=nonlinearity)
        self.decoder = Decoder(x_dim=x_dim, channels=channels, z_dim=z_dim, h_dim=h_dim, kernel_size=kernel_size, nonlinearity=nonlinearity)
        self.df = DF(mu_dim=mu_dim, z_dim=z_dim)

    def forward(self, x, mu_t):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        z_n = self.df(mu_t)
        x_recon_n = self.decoder(z_n)
        return x_recon, z, x_recon_n, z_n

class Encoder(nn.Module):
    def __init__(self, x_dim=256, channels=1, z_dim=50, h_dim=128, kernel_size='large', nonlinearity='elu'):
        super(Encoder, self).__init__()

        self.ch = channels

        if kernel_size == 'large':
            self.conv1 = nn.Conv1d(channels, 8, 25, stride=2, padding=1)
            self.conv2 = nn.Conv1d(8, 16, 25, stride=2, padding=1)
            self.conv3 = nn.Conv1d(16, 32, 25, stride=1, padding=1)
            self.conv4 = nn.Conv1d(32, 64, 25, stride=2, padding=1)
        if kernel_size == 'small':
            self.conv1 = nn.Conv1d(channels, 8, 3, stride=2, padding=1)
            self.conv2 = nn.Conv1d(8, 16, 3, stride=2, padding=1)
            self.conv3 = nn.Conv1d(16, 32, 3, stride=1, padding=1)
            self.conv4 = nn.Conv1d(32, 64, 3, stride=2, padding=1)

        self.fc = nn.Linear(h_dim, z_dim)

        self.x_dim = x_dim
        self.nonlinearity = nonlinearity

    def encoder(self, x):
        pad_dim = (256 - x.shape[1]) // 2
        pad = torch.zeros(x.shape[0], pad_dim).to(x.device)
        x = torch.cat([pad, x], dim=1)
        x = torch.cat([x, pad], dim=1)
        x = x.reshape(x.shape[0], self.ch, int(x.shape[1]/self.ch))
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, x_dim=256, channels=1, z_dim=20, h_dim=128,  kernel_size='large', nonlinearity='elu'):
        super(Decoder, self).__init__()

        self.ch = 1
        # decoder part
        self.x_dim = x_dim
        self.fc = nn.Linear(z_dim, h_dim)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, int(h_dim / 64)))

        if kernel_size == 'large':
            self.deconv1 = nn.ConvTranspose1d(64, 32, 25, stride=1, padding=0)
            self.deconv2 = nn.ConvTranspose1d(32, 16, 25, stride=1, padding=1)
            self.deconv3 = nn.ConvTranspose1d(16, 8, 25, stride=2, padding=0)
            self.deconv4 = nn.ConvTranspose1d(8, self.ch, 25, stride=2, padding=0)
        if kernel_size == 'small':
            self.fc = nn.Linear(z_dim, 2048)
            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, int(2048/64)))
            self.deconv1 = nn.ConvTranspose1d(64, 32, 3, stride=1, padding=0)
            self.deconv2 = nn.ConvTranspose1d(32, 16, 3, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose1d(16, 8, 3, stride=2, padding=1)
            self.deconv4 = nn.ConvTranspose1d(8, self.ch, 3, stride=2, padding=1)

        self.nonlinearity = nonlinearity

    def decoder(self, z):
        z = F.elu(self.fc(z))
        z = self.unflatten(z)
        z = F.elu(self.deconv1(z))
        z = F.elu(self.deconv2(z))
        z = F.elu(self.deconv3(z))
        x = self.deconv4(z)
        x = x[:, :, :self.x_dim]
        x = x.reshape(x.shape[0], int(self.ch*x.shape[2]))
        return x

    def forward(self, x):
        return self.decoder(x)


class AE_INN(fr.ReversibleGraphNet):
    def __init__(self, x_dim=64, z_dim=1, h_dim=128, activation='relu', inn_layer_type='rev_multiplicative_layer',
           use_spectral_norm=False, f_class='fcnn'):
        """
            Return an autoencoder.

            :param input_size: size of the input. Default: 64
            :return:
            """
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.activation = activation
        self.inn_layer_type = inn_layer_type
        self.use_spectral_norm = use_spectral_norm
        self.f_class = f_class

        if activation == 'relu':
            activation = torch.nn.ReLU()
        if activation == 'elu':
            activation = torch.nn.ELU()
        if activation == 'tanh':
            activation = torch.nn.Tanh()
        if activation == 'sigmoid':
            activation = torch.nn.Sigmoid()
        if activation == 'gelu':
            activation = torch.nn.GELU()

        if inn_layer_type == 'rev_multiplicative_layer':
            layer_type = rev_multiplicative_layer
        if inn_layer_type == 'rev_layer':
            layer_type = rev_layer

        if f_class == 'fcnn':
            function_class = Fully_Connected_Network

        inp = fr.InputNode(x_dim, name='input')

        fc = fr.Node([inp.out0], layer_type, {'F_class': function_class, 'F_args': {'input_dim': x_dim,
                                                                                    'output_dim': x_dim,
                                                                                    'z_dim': z_dim,
                                                                                    'h_dim': h_dim,
                                                                                    'activation': activation,
                                                                                    'is_last_layer': False,
                                                                                    'use_spectral_norm': use_spectral_norm}},
                     name='fc')
        fc1 = fr.Node([fc.out0], layer_type, {'F_class': function_class, 'F_args': {'input_dim': x_dim,
                                                                                    'output_dim': x_dim,
                                                                                    'z_dim': z_dim,
                                                                                    'h_dim': h_dim,
                                                                                    'activation': activation,
                                                                                    'is_last_layer': False,
                                                                                    'use_spectral_norm': use_spectral_norm}},
                      name='fc1')
        fc2 = fr.Node([fc1.out0], layer_type, {'F_class': function_class, 'F_args': {'input_dim': x_dim,
                                                                                     'output_dim': x_dim,
                                                                                     'z_dim': z_dim,
                                                                                     'h_dim': h_dim,
                                                                                     'activation': activation,
                                                                                     'is_last_layer': False,
                                                                                     'use_spectral_norm': use_spectral_norm}},
                      name='fc2')
        fc3 = fr.Node([fc2.out0], layer_type, {'F_class': function_class, 'F_args': {'input_dim': x_dim,
                                                                                     'output_dim': x_dim,
                                                                                     'z_dim': z_dim,
                                                                                     'h_dim': h_dim,
                                                                                     'activation': activation,
                                                                                     'is_last_layer': False,
                                                                                     'use_spectral_norm': use_spectral_norm}},
                      name='fc3')
        fc4 = fr.Node([fc3.out0], layer_type, {'F_class': function_class, 'F_args': {'input_dim': x_dim,
                                                                                     'output_dim': x_dim,
                                                                                     'z_dim': z_dim,
                                                                                     'h_dim': h_dim,
                                                                                     'activation': activation,
                                                                                     'is_last_layer': True,
                                                                                     'use_spectral_norm': use_spectral_norm}},
                      name='fc4')
        outp = fr.OutputNode([fc4.out0], name='output')
        nodes = [inp, outp, fc, fc1, fc2, fc3, fc4]
        super().__init__(nodes, 0, 1)

    def forward(self, x, z_dim=None):
        z_dim = self.z_dim if z_dim is None else z_dim
        z = super().forward(x, rev=False)
        z[...,z_dim:] = 0
        return super().forward(z, rev=True), z[...,:z_dim]

    def encode(self, x):
        z = super().forward(x, rev=False)
        z_trunc = z[..., :self.z_dim]
        return z_trunc

    def decode(self, z):
        z_padded = torch.zeros((*z.shape[:-1],self.x_dim)).to(z.device)
        z_padded[..., :self.z_dim] = z
        return super().forward(z_padded, rev=True)

    def save_config(self, path):
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        with open(path, 'wb') as handle:
            config = {
                'x_dim': self.x_dim,
                'z_dim': self.z_dim,
                'h_dim': self.h_dim,
                'activation': self.activation,
                'inn_layer_type': self.inn_layer_type,
                'use_spectral_norm': self.use_spectral_norm,
                'f_class': self.f_class
            }
            pickle.dump(config, handle)

    @classmethod
    def load_config(cls, path):
        with open(path, 'rb') as handle:
            config = pickle.load(handle)
            return cls(**config)


# def AE_INN(x_dim=64, z_dim=1, h_dim=128, activation='relu', inn_layer_type='rev_multiplicative_layer',
#            use_spectral_norm=False, f_class='fcnn'):
#     """
#     Return an autoencoder.
#
#     :param input_size: size of the input. Default: 64
#     :return:
#     """
#
#     if activation == 'relu':
#         activation = torch.nn.ReLU()
#     if activation == 'elu':
#         activation = torch.nn.ELU()
#     if activation == 'tanh':
#         activation = torch.nn.Tanh()
#     if activation == 'sigmoid':
#         activation = torch.nn.Sigmoid()
#     if activation == 'gelu':
#         activation = torch.nn.GELU()
#
#     if inn_layer_type == 'rev_multiplicative_layer':
#         layer_type = rev_multiplicative_layer
#     if inn_layer_type == 'rev_layer':
#         layer_type = rev_layer
#
#     if f_class == 'fcnn':
#         function_class = Fully_Connected_Network
#
#     inp = fr.InputNode(x_dim, name='input')
#
#     fc = fr.Node([inp.out0], layer_type, { 'F_class': function_class, 'F_args': {'input_dim': x_dim,
#                                                                                                 'output_dim': x_dim,
#                                                                                                 'z_dim': z_dim,
#                                                                                                 'h_dim': h_dim,
#                                                                                                 'activation': activation,
#                                                                                                 'is_last_layer': False,
#                                                                                                 'use_spectral_norm': use_spectral_norm}}, name='fc')
#     fc1 = fr.Node([fc.out0], layer_type, { 'F_class': function_class, 'F_args': {'input_dim': x_dim,
#                                                                                                 'output_dim': x_dim,
#                                                                                                 'z_dim': z_dim,
#                                                                                                 'h_dim': h_dim,
#                                                                                                 'activation': activation,
#                                                                                                 'is_last_layer': False,
#                                                                                                 'use_spectral_norm': use_spectral_norm}}, name='fc1')
#     fc2 = fr.Node([fc1.out0], layer_type, { 'F_class': function_class, 'F_args': {'input_dim': x_dim,
#                                                                                                 'output_dim': x_dim,
#                                                                                                 'z_dim': z_dim,
#                                                                                                 'h_dim': h_dim,
#                                                                                                 'activation': activation,
#                                                                                                 'is_last_layer': False,
#                                                                                                 'use_spectral_norm': use_spectral_norm}}, name='fc2')
#     fc3 = fr.Node([fc2.out0], layer_type, { 'F_class': function_class, 'F_args': {'input_dim': x_dim,
#                                                                                                 'output_dim': x_dim,
#                                                                                                 'z_dim': z_dim,
#                                                                                                 'h_dim': h_dim,
#                                                                                                 'activation': activation,
#                                                                                                 'is_last_layer': False,
#                                                                                                 'use_spectral_norm': use_spectral_norm}}, name='fc3')
#     fc4 = fr.Node([fc3.out0], layer_type, { 'F_class': function_class, 'F_args': {'input_dim': x_dim,
#                                                                                                 'output_dim': x_dim,
#                                                                                                 'z_dim': z_dim,
#                                                                                                 'h_dim': h_dim,
#                                                                                                 'activation': activation,
#                                                                                                 'is_last_layer': True,
#                                                                                                 'use_spectral_norm': use_spectral_norm}}, name='fc4')
#     outp = fr.OutputNode([fc4.out0], name='output')
#     nodes = [inp, outp, fc, fc1, fc2, fc3, fc4]
#     model = fr.ReversibleGraphNet(nodes, 0, 1)
#
#     return model


class Fully_Connected_Network(nn.Module):
    def __init__(self, input_dim=64, output_dim=64, z_dim=1, h_dim=512, activation=torch.nn.ReLU(), is_last_layer=False, use_spectral_norm=False):
        super(Fully_Connected_Network, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        #self.z_dim = z_dim
        self.act = activation

        if use_spectral_norm:
            print("spectral norm for each layer")
            self.fc1 = spectral_norm(nn.Linear(input_dim, h_dim), n_power_iterations=1)
            self.fc2 = spectral_norm(nn.Linear(h_dim, output_dim), n_power_iterations=1)
        else:
            self.fc1 = nn.Linear(input_dim, h_dim)
            self.fc2 = nn.Linear(h_dim, output_dim)

        if is_last_layer:
            torch.nn.init.zeros_(self.fc2.weight)

    def net(self, x):
        z = self.act(self.fc1(x))
        z = self.fc2(z)
        return z

    def forward(self, x):
        return self.net(x)


class rev_multiplicative_layer(nn.Module):
    '''The RevNet block is not a general function approximator. The reversible
    layer with a multiplicative term presented in the real-NVP paper is much
    more general. This class uses some non-reversible transformation F, but
    splits the channels up to make it revesible (see lifting scheme). F itself
    does not have to be revesible. See F_* classes above for examples.'''

    def __init__(self, dims_in, F_class, clamp=1., F_args={}):
        super(rev_multiplicative_layer, self).__init__()

        input_dim = F_args['input_dim']
        output_dim = F_args['output_dim']
        z_dim = F_args['z_dim']
        h_dim = F_args['h_dim']
        activation = F_args['activation']
        is_last_layer = F_args['is_last_layer']
        use_spectral_norm = F_args['use_spectral_norm']

        channels = input_dim

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        self.ndims = len(dims_in)

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.s1 = F_class(input_dim=self.split_len1, output_dim=self.split_len1, z_dim=z_dim, h_dim=h_dim, activation=activation, is_last_layer=is_last_layer, use_spectral_norm=use_spectral_norm)
        self.t1 = F_class(input_dim=self.split_len1, output_dim=self.split_len1, z_dim=z_dim, h_dim=h_dim, activation=activation, is_last_layer=is_last_layer, use_spectral_norm=use_spectral_norm)
        self.s2 = F_class(input_dim=self.split_len2, output_dim=self.split_len1, z_dim=z_dim, h_dim=h_dim, activation=activation, is_last_layer=is_last_layer, use_spectral_norm=use_spectral_norm)
        self.t2 = F_class(input_dim=self.split_len2, output_dim=self.split_len1, z_dim=z_dim, h_dim=h_dim, activation=activation, is_last_layer=is_last_layer, use_spectral_norm=use_spectral_norm)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s))

    def log_e(self, s):
        '''log of the nonlinear function e'''
        return self.clamp * 0.636 * torch.atan(s)

    def forward(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = self.e(self.s2(x2)) * x1 + self.t2(x2)
            y2 = self.e(self.s1(y1)) * x2 + self.t1(y1)
        else:  # names of x and y are swapped!
            y2 = (x2 - self.t1(x1)) / self.e(self.s1(x1))
            y1 = (x1 - self.t2(y2)) / self.e(self.s2(y2))
        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            s2 = self.s2(x2)
            y1 = self.e(s2) * x1 + self.t2(x2)
            jac = self.log_e(self.s1(y1)) + self.log_e(s2)
        else:
            s1 = self.s1(x1)
            y2 = (x2 - self.t1(x1)) / self.e(s1)
            jac = -self.log_e(s1) - self.log_e(self.s2(y2))

        return torch.sum(jac, dim=tuple(range(1, self.ndims+1)))

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class rev_layer(nn.Module):
    '''General reversible layer modeled after the lifting scheme. Uses some
    non-reversible transformation F, but splits the channels up to make it
    revesible (see lifting scheme). F itself does not have to be revesible. See
    F_* classes above for examples.'''

    def __init__(self, dims_in, F_class, F_args={}):
        super(rev_layer, self).__init__()

        input_dim = F_args['input_dim']
        output_dim = F_args['output_dim']
        z_dim = F_args['z_dim']
        h_dim = F_args['h_dim']
        activation = F_args['activation']
        is_last_layer = F_args['is_last_layer']

        channels = input_dim
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.F = F_class(input_dim=self.split_len1, output_dim=self.split_len1, z_dim=z_dim, h_dim=h_dim, activation=activation, is_last_layer=is_last_layer, is_G=False)
        self.G = F_class(input_dim=self.split_len1, output_dim=self.split_len1, z_dim=z_dim, h_dim=h_dim, activation=activation, is_last_layer=is_last_layer, is_G=True)
        
    def forward(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            y2 = x2 + self.G(y1)
        else:
            y2 = x2 - self.G(x1)
            y1 = x1 - self.F(y2)

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, rev=False):
        return torch.zeros(x.shape[0])

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims