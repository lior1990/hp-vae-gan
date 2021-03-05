from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import numpy as np
import copy
import utils
from vqvae.quantizer import VectorQuantizer


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
        "lrelu": nn.LeakyReLU(0.2, inplace=True),
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
            self.add_module("pooling", torch.nn.AvgPool2d(2))


class FeatureExtractor(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, num_blocks=2, return_linear=False, pooling=False):
        super(FeatureExtractor, self).__init__()
        self.add_module('conv_block_0', ConvBlock2DSN(in_channel, out_channel, ker_size, padding, stride)),
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
    def __init__(self, opt, out_dim: int, latent_spatial_dimensions: "Tuple[int, int]", num_blocks=2):
        super(Encode2DAE, self).__init__()

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, opt.ker_size, opt.ker_size // 2, 1, num_blocks=num_blocks)
        self.max_pool2d = torch.nn.MaxPool2d(opt.ker_size)
        self.conv_pre_resize = ConvBlock2D(opt.nfc, out_dim, opt.ker_size, opt.ker_size // 2, 1, bn=False)
        self.conv_post_resize = ConvBlock2D(out_dim, out_dim, opt.ker_size, opt.ker_size // 2, 1, bn=False)

        self.latent_spatial_dim = latent_spatial_dimensions

    def forward(self, x):
        features = self.features(x)
        features = self.max_pool2d(features)
        z = self.conv_pre_resize(features)
        z = nn.functional.interpolate(z, size=self.latent_spatial_dim)
        z = self.conv_post_resize(z)
        return z


class Encode2DVQVAE(nn.Module):
    def __init__(self, opt, out_dim: int, num_blocks=2):
        super(Encode2DVQVAE, self).__init__()

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, opt.ker_size, opt.ker_size // 2, 1, num_blocks=num_blocks, pooling=True)
        self.conv = ConvBlock2D(opt.nfc, out_dim, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None)

    def forward(self, x):
        features = self.features(x)
        return self.conv(features)


class Encode2DVAE(nn.Module):
    def __init__(self, opt, out_dim=None, num_blocks=2):
        super(Encode2DVAE, self).__init__()

        if out_dim is None:
            output_dim = opt.nfc
        else:
            assert type(out_dim) is int
            output_dim = out_dim

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, opt.ker_size, opt.ker_size // 2, 1, num_blocks=num_blocks)
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
        self.head = ConvBlock2DSN(opt.nc_im, N, opt.ker_size, opt.ker_size // 2, stride=1, bn=True, act='lrelu')
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


class UpsampleConvBlock2D(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn=True, act='lrelu'):
        super(UpsampleConvBlock2D, self).__init__()
        self.add_module("upsample", torch.nn.Upsample(scale_factor=2))
        self.add_module("conv", ConvBlock2D(in_channel, out_channel, ker_size, padding, stride,bn=bn, act=act))


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential()
        N = int(opt.nfc)
        self.decoder.add_module('head', ConvBlock2D(opt.embedding_dim, N, opt.ker_size, opt.padd_size, stride=1))
        for i in range(opt.enc_blocks-1):
            block = UpsampleConvBlock2D(N, N, opt.ker_size, opt.padd_size, stride=1)
            self.decoder.add_module('block%d' % (i), block)
        self.decoder.add_module('tail', nn.Conv2d(N, opt.nc_im, opt.ker_size, 1, opt.ker_size // 2))

    def forward(self, z):
        return self.decoder(z)


class GeneratorHPVAEGAN(nn.Module):
    def __init__(self, opt):
        super(GeneratorHPVAEGAN, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        self.N = N

        self.vqvae_encode = Encode2DVQVAE(opt, out_dim=opt.embedding_dim, num_blocks=opt.enc_blocks)
        self.vector_quantization = VectorQuantizer(opt.n_embeddings, opt.embedding_dim, opt.vqvae_beta)
        self.decoder = Decoder(opt)

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

    def forward(self, img, noise_amp, noise_init=None, mode='rand'):
        z_e = self.vqvae_encode(img)
        embedding_loss, z_q, _, _, _ = self.vector_quantization(z_e, mode)
        vqvae_out = torch.tanh(self.decoder(z_q))

        x_prev_out = self.refinement_layers(0, vqvae_out, noise_amp, mode)

        return x_prev_out, embedding_loss

    def refinement_layers(self, start_idx, x_prev_out, noise_amp, mode):
        for idx, block in enumerate(self.body[start_idx:], start_idx):
            # Upscale
            x_prev_out_up = utils.upscale_2d(x_prev_out, idx + 1, self.opt)

            # Add noise if "random" sampling, else, add no noise is "reconstruction" mode
            if mode in ['rand', 'noise_rand']:
                noise = utils.generate_noise(ref=x_prev_out_up)
                x_prev = block(x_prev_out_up + noise * noise_amp[idx + 1])
            else:
                x_prev = block(x_prev_out_up)

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
