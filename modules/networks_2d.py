from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import numpy as np
import copy
import utils
from modules.utils import pad_with_cls


def conv_weights_init_ones(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') == 0 or classname.find('Conv2d') == 0:
        m.weight.data.fill_(1 / np.prod(m.kernel_size))
        m.bias.data.fill_(0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_activation(act):
    activations = {
        "relu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(0.2),
        "elu": nn.ELU(alpha=1.0, inplace=True),
        "prelu": nn.PReLU(num_parameters=1, init=0.25),
        "selu": nn.SELU(inplace=True)
    }
    return activations[act]


def reparameterize(mu, logvar, training):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = torch.zeros_like(std).normal_()
        return eps.mul(std).add_(mu)
    else:
        return torch.zeros_like(mu).normal_()


def reparameterize_bern(x, training):
    if training:
        eps = torch.zeros_like(x).uniform_()
        return torch.log(x + 1e-20) - torch.log(-torch.log(eps + 1e-20) + 1e-20)
    else:
        return torch.zeros_like(x).bernoulli_()


class ConvBlock2D(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn=True, act='lrelu'):
        super(ConvBlock2D, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size,
                                          stride=stride, padding=padding))
        if bn:
            self.add_module('norm', nn.BatchNorm2d(out_channel))
        if act is not None:
            self.add_module(act, get_activation(act))


class ConvBlock2DSN(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn=True, act='lrelu', pooling=False):
        super(ConvBlock2DSN, self).__init__()
        if bn:
            self.add_module('conv', nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=ker_size,
                                                                     stride=stride, padding=padding)))
        else:
            self.add_module('conv',
                            nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padding,
                                      padding_mode='reflect'))
        if act is not None:
            self.add_module(act, get_activation(act))

        if pooling:
            self.add_module("avg_pool", nn.AvgPool2d(ker_size))


class FeatureExtractor(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, num_blocks=2, return_linear=False, pooling=True):
        super(FeatureExtractor, self).__init__()
        self.add_module('conv_block_0', ConvBlock2DSN(in_channel, out_channel, ker_size, padding, stride, pooling=pooling)),
        for i in range(num_blocks - 1):
            self.add_module('conv_block_{}'.format(i + 1),
                            ConvBlock2DSN(out_channel, out_channel, ker_size, padding, stride, pooling=pooling))
        if return_linear:
            self.add_module('conv_block_{}'.format(num_blocks),
                            ConvBlock2DSN(out_channel, out_channel, ker_size, padding, stride, bn=False, act=None))
        else:
            self.add_module('conv_block_{}'.format(num_blocks),
                            ConvBlock2DSN(out_channel, out_channel, ker_size, padding, stride))


class Encode2DAE(nn.Module):
    def __init__(self, opt, out_dim=None, num_blocks=2):
        super(Encode2DAE, self).__init__()

        if out_dim is None:
            output_dim = opt.nfc
        else:
            assert type(out_dim) is int
            output_dim = out_dim

        self.features = FeatureExtractor(opt.nc_im+1, opt.nfc, opt.ker_size, opt.ker_size // 2, 1, num_blocks=num_blocks)
        self.encoder = ConvBlock2D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1)

    def forward(self, x):
        features = self.features(x)
        encoder = self.encoder(features)

        return encoder


class Encode2DVAE(nn.Module):
    def __init__(self, opt, out_dim=None, num_blocks=2):
        super(Encode2DVAE, self).__init__()

        if out_dim is None:
            output_dim = opt.nfc
        else:
            assert type(out_dim) is int
            output_dim = out_dim

        self.features = FeatureExtractor(opt.nc_im+1, opt.nfc, opt.ker_size, opt.ker_size // 2, 1, num_blocks=num_blocks, pooling=False)
        self.mu = ConvBlock2D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None)
        self.logvar = ConvBlock2D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None)

    def forward(self, x):
        features = self.features(x)
        mu = self.mu(features)
        logvar = self.logvar(features)

        return mu, logvar


class Encode2DVAE_nb(nn.Module):
    def __init__(self, opt, out_dim=None, num_blocks=2):
        super(Encode2DVAE_nb, self).__init__()

        if out_dim is None:
            output_dim = opt.nfc
        else:
            assert type(out_dim) is int
            output_dim = out_dim

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, opt.ker_size, opt.ker_size // 2, 1, num_blocks=num_blocks)
        self.mu = nn.Sequential(
            ConvBlock2D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None),
            nn.AdaptiveAvgPool2d(1)
        )
        self.logvar = nn.Sequential(
            ConvBlock2D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None),
            nn.AdaptiveAvgPool2d(1)
        )
        self.bern = ConvBlock2D(opt.nfc, 1, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None)

    def forward(self, x):
        features = self.features(x)
        bern = torch.sigmoid(self.bern(features))
        features = bern * features
        mu = self.mu(features)
        logvar = self.logvar(features)

        return mu, logvar, bern


class Encode3DVAE1x1(nn.Module):
    def __init__(self, opt, out_dim=None):
        super(Encode3DVAE1x1, self).__init__()

        if out_dim is None:
            output_dim = opt.nfc
        else:
            assert type(out_dim) is int
            output_dim = out_dim

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, 1, 0, 1, num_blocks=2)
        self.mu = ConvBlock2D(opt.nfc, output_dim, 1, 0, 1, bn=False, act=None)
        self.logvar = ConvBlock2D(opt.nfc, output_dim, 1, 0, 1, bn=False, act=None)

    def forward(self, x):
        features = self.features(x)
        mu = self.mu(features)
        logvar = self.logvar(features)

        return mu, logvar


class WDiscriminator2D(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator2D, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        self.head = ConvBlock2DSN(opt.nc_im+1, N, opt.ker_size, opt.ker_size // 2, stride=1, bn=True, act='lrelu')
        self.body = nn.Sequential()
        for i in range(opt.num_layer):
            block = ConvBlock2DSN(N, N, opt.ker_size, opt.ker_size // 2, stride=1, bn=True, act='lrelu')
            self.body.add_module('block%d' % (i), block)
        self.tail = nn.Conv2d(N, 1, kernel_size=opt.ker_size, padding=1, stride=1)

    def forward(self, x):
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        return out


class WDiscriminator2DMulti(nn.Module):
    def __init__(self, opt, num_classes):
        super(WDiscriminator2DMulti, self).__init__()
        self.opt = opt
        N = int(opt.nfc)
        self.head = ConvBlock2DSN(opt.nc_im, N, opt.ker_size, opt.ker_size // 2, stride=1, bn=True, act='lrelu')
        self.body = nn.Sequential()
        for i in range(opt.num_layer):
            block = ConvBlock2DSN(N, N, opt.ker_size, opt.ker_size // 2, stride=1, bn=True, act='lrelu')
            self.body.add_module('block%d' % (i), block)
        self.tail = nn.Conv2d(N, num_classes, kernel_size=opt.ker_size, padding=opt.padd_size, stride=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class Conv2DTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, act='lrelu'):
        super(Conv2DTranspose, self).__init__()
        self.trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(act)

    def forward(self, x, output_size=None):
        x = self.trans(x, output_size=output_size)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Conv2DUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, **kwargs):
        super(Conv2DUpsample, self).__init__()
        self.conv = ConvBlock2D(in_channels, out_channels, kernel_size, padding, stride)
        self.up = nn.Upsample(**kwargs)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class GeneratorHPVAEGAN(nn.Module):
    def __init__(self, opt):
        super(GeneratorHPVAEGAN, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        self.N = N

        self.encode = Encode2DAE(opt, out_dim=opt.latent_dim, num_blocks=opt.enc_blocks)

        # AE decoder:
        self.decoder_head = Conv2DUpsample(opt.latent_dim, N, opt.ker_size, 1, opt.padd_size, scale_factor=2)
        self.decoder_base = Conv2DUpsample(N, N, opt.ker_size, 1, opt.padd_size, size=(22, 33))
        self.decoder_tail = nn.Conv2d(N, opt.nc_im, opt.ker_size, 1, opt.padd_size)

        self.body = torch.nn.ModuleList([])
        self.extra_body = torch.nn.ModuleList([])

    def init_next_stage(self):
        if len(self.body) == 0:
            _first_stage = nn.Sequential()
            _first_stage.add_module('head',
                                    ConvBlock2D(self.opt.nc_im+1, self.N, self.opt.ker_size, self.opt.padd_size,
                                                stride=1))
            for i in range(self.opt.num_layer):
                block = ConvBlock2D(self.N, self.N, self.opt.ker_size, self.opt.padd_size, stride=1)
                _first_stage.add_module('block%d' % (i), block)
            _first_stage.add_module('tail',
                                    nn.Conv2d(self.N, self.opt.nc_im, self.opt.ker_size, 1, self.opt.ker_size // 2))
            self.body.append(_first_stage)
        else:
            self.body.append(copy.deepcopy(self.body[-1]))

    def init_extra_layer(self):
        seq = nn.Sequential()
        seq.add_module('head', ConvBlock2D(self.opt.nc_im, self.N, self.opt.ker_size, self.opt.padd_size, stride=1))
        for i in range(self.opt.num_layer):
            block = ConvBlock2D(self.N, self.N, self.opt.ker_size, self.opt.padd_size, stride=1)
            seq.add_module('block%d' % (i), block)
        seq.add_module('tail', nn.Conv2d(self.N, self.opt.nc_im, self.opt.ker_size, 1, self.opt.ker_size // 2))
        self.extra_body.append(seq)

    def forward(self, video, class_maps_per_scale, noise_amp, noise_init=None, sample_init=None, mode='rand'):
        if sample_init is not None:
            assert len(self.body) > sample_init[0], "Strating index must be lower than # of body blocks"

        class_maps_for_encoder = class_maps_per_scale[0]
        z_ae = self.encode(torch.cat([video, class_maps_for_encoder], dim=1))

        if noise_init is not None:
            z_ae += noise_init

        # decode
        vae_out = self.decoder_head(z_ae)
        vae_out = self.decoder_base(vae_out)
        vae_out = self.decoder_tail(vae_out)
        vae_out = torch.tanh(vae_out)

        x_prev_out = self.refinement_layers(0, vae_out, noise_amp, mode, class_maps_per_scale)

        for block in self.extra_body:
            x_prev_out_residuals = block(x_prev_out)
            x_prev_out = torch.tanh(x_prev_out + x_prev_out_residuals)

        return x_prev_out, vae_out, z_ae

    def refinement_layers(self, start_idx, x_prev_out, noise_amp, mode, class_maps_per_scale):
        for idx, block in enumerate(self.body[start_idx:], start_idx):
            # todo: remove this so encoder will train at all scales?
            if self.opt.vae_levels == idx + 1 and not self.opt.train_all:
                x_prev_out.detach_()

            # Upscale
            x_prev_out_up = utils.upscale_2d(x_prev_out, idx + 1, self.opt)

            # Add noise if "random" sampling, else, add no noise is "reconstruction" mode
            if mode == 'rand':
                noise = utils.generate_noise(ref=x_prev_out_up)
                x_prev_forward = x_prev_out_up + noise * noise_amp[idx + 1]
            else:
                x_prev_forward = x_prev_out_up

            x_prev = block(torch.cat([x_prev_forward, class_maps_per_scale[idx+1]], dim=1))

            x_prev_out = torch.tanh(x_prev + x_prev_out_up)

        return x_prev_out


class GeneratorVAE_nb(nn.Module):
    def __init__(self, opt):
        super(GeneratorVAE_nb, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        self.N = N

        self.encode = Encode2DVAE_nb(opt, out_dim=opt.latent_dim, num_blocks=opt.enc_blocks)
        self.decoder = nn.Sequential()

        # Normal Decoder
        self.decoder.add_module('head', ConvBlock2D(opt.latent_dim, N, opt.ker_size, opt.padd_size, stride=1))
        for i in range(opt.num_layer):
            block = ConvBlock2D(N, N, opt.ker_size, opt.padd_size, stride=1)
            self.decoder.add_module('block%d' % (i), block)
        self.decoder.add_module('tail', nn.Conv2d(N, opt.nc_im, opt.ker_size, 1, opt.ker_size // 2))

        self.body = torch.nn.ModuleList([])

    def init_next_stage(self):
        if len(self.body) == 0:
            _first_stage = nn.Sequential()
            _first_stage.add_module('head',
                                    ConvBlock2D(self.opt.nc_im, self.N, self.opt.ker_size, self.opt.padd_size,
                                                stride=1))
            for i in range(self.opt.num_layer):
                block = ConvBlock2D(self.N, self.N, self.opt.ker_size, self.opt.padd_size, stride=1)
                _first_stage.add_module('block%d' % (i), block)
            _first_stage.add_module('tail',
                                    nn.Conv2d(self.N, self.opt.nc_im, self.opt.ker_size, 1, self.opt.ker_size // 2))
            self.body.append(_first_stage)
        else:
            self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, video, noise_amp, noise_init_norm=None, noise_init_bern=None, sample_init=None, mode='rand'):
        if sample_init is not None:
            assert len(self.body) > sample_init[0], "Strating index must be lower than # of body blocks"

        if noise_init_norm is None:
            mu, logvar, bern = self.encode(video)
            z_vae_norm = reparameterize(mu, logvar, self.training)
            z_vae_bern = reparameterize_bern(bern, self.training)
        else:
            z_vae_norm = noise_init_norm
            z_vae_bern = noise_init_bern

        vae_out = torch.tanh(self.decoder(z_vae_norm * z_vae_bern))

        if sample_init is not None:
            x_prev_out = self.refinement_layers(sample_init[0], sample_init[1], noise_amp, mode)
        else:
            x_prev_out = self.refinement_layers(0, vae_out, noise_amp, mode)

        if noise_init_norm is None:
            return x_prev_out, vae_out, (mu, logvar, bern)
        else:
            return x_prev_out, vae_out

    def refinement_layers(self, start_idx, x_prev_out, noise_amp, mode):
        for idx, block in enumerate(self.body[start_idx:], start_idx):
            if self.opt.vae_levels == idx + 1:
                x_prev_out.detach_()

            # Upscale
            x_prev_out_up = utils.upscale_2d(x_prev_out, idx + 1, self.opt)

            # Add noise if "random" sampling, else, add no noise is "reconstruction" mode
            if mode == 'rand':
                noise = utils.generate_noise(ref=x_prev_out_up)
                x_prev = block(x_prev_out_up + noise * noise_amp[idx + 1])
            else:
                x_prev = block(x_prev_out_up)

            x_prev_out = torch.tanh(x_prev + x_prev_out_up)

        return x_prev_out
