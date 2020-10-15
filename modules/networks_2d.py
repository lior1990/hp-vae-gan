from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
import torchvision

import utils
from modules.spade_block import SPADEResnetBlock
from modules.spade_normalization import SPADE, NormLayer

VGG = torchvision.models.vgg19(pretrained=True).features
VGG_CACHE = {}
for param in VGG.parameters():
    param.requires_grad = False


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
        "relu": nn.ReLU(inplace=True),
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


class SPADEBlock2D(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, norm_layer: NormLayer, act='lrelu'):
        super(SPADEBlock2D, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size,
                                          stride=stride, padding=padding))
        self.add_module('norm', SPADE(norm_layer, ker_size, out_channel, 3))
        self.add_module(act, get_activation(act))

    def forward(self, tup):
        x, source_img = tup
        modules = iter(self)

        conv = modules.__next__()
        x = conv(x)

        norm = modules.__next__()
        x = norm((x, source_img))

        actv = modules.__next__()
        x = actv(x)

        return x


class ConvBlock2DSN(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn=True, act='lrelu'):
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


class FeatureExtractor(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, num_blocks=2, return_linear=False):
        super(FeatureExtractor, self).__init__()
        self.add_module('conv_block_0', ConvBlock2DSN(in_channel, out_channel, ker_size, padding, stride)),
        for i in range(num_blocks - 1):
            self.add_module('conv_block_{}'.format(i + 1),
                            ConvBlock2DSN(out_channel, out_channel, ker_size, padding, stride))
        if return_linear:
            self.add_module('conv_block_{}'.format(num_blocks),
                            ConvBlock2DSN(out_channel, out_channel, ker_size, padding, stride, bn=False, act=None))
        else:
            self.add_module('conv_block_{}'.format(num_blocks),
                            ConvBlock2DSN(out_channel, out_channel, ker_size, padding, stride))


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


class SPADESequential(nn.Sequential):
    def forward(self, tup):
        x, source_img = tup
        for module in self:
            x = module((x, source_img))
        return x


class GeneratorHPVAEGAN(nn.Module):
    def __init__(self, opt):
        super(GeneratorHPVAEGAN, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        self.N = N

        self.encode = Encode2DVAE(opt, out_dim=opt.latent_dim, num_blocks=opt.enc_blocks)
        self.decoder = nn.Sequential()

        # Normal Decoder
        self.decoder.add_module('head', ConvBlock2D(opt.latent_dim, N, opt.ker_size, opt.padd_size, stride=1))
        for i in range(opt.num_layer):
            block = ConvBlock2D(N, N, opt.ker_size, opt.padd_size, stride=1)
            self.decoder.add_module('block%d' % (i), block)
        self.decoder.add_module('tail', nn.Conv2d(N, opt.nc_im, opt.ker_size, 1, opt.ker_size // 2))

        self.body = torch.nn.ModuleList([])
        self.max_pool_2d = nn.MaxPool2d(opt.ker_size, stride=1, padding=1)
        self.l1_loss = nn.L1Loss()
        global VGG
        VGG = VGG.to(opt.device)

    def init_next_stage(self):
        def create_spade_seq():
            _stage = SPADESequential()
            _stage.add_module('head', SPADEBlock2D(self.opt.nc_im, self.N, self.opt.ker_size, 1, 1, self.opt.norm_layer))
            for i in range(self.opt.num_layer):
                block = SPADEBlock2D(self.N, self.N, self.opt.ker_size, 1, 1, self.opt.norm_layer)
                _stage.add_module('block%d' % (i), block)
            _stage.add_module('tail', SPADEBlock2D(self.N, self.opt.nc_im, self.opt.ker_size, 1, 1, self.opt.norm_layer))
            return _stage

        if len(self.body) == 0:
            first_stage = create_spade_seq()
            self.body.append(first_stage)
        else:
            new_stage = create_spade_seq()
            new_stage.load_state_dict(copy.deepcopy(self.body[-1].state_dict()))
            self.body.append(new_stage)

            global VGG_CACHE
            VGG_CACHE.clear()  # reduce memory consumption between scales

    @staticmethod
    def _calc_mean_std(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, real_zero, noise_amp, noise_init=None, sample_init=None, mode='rand'):
        if sample_init is not None:
            assert len(self.body) > sample_init[0], "Strating index must be lower than # of body blocks"

        size = real_zero.size()
        real_mean, real_std = self._calc_mean_std(real_zero)
        real_mean = real_mean.expand(size)
        real_std = real_std.expand(size)

        normalized_real_zero = (real_zero - real_mean) / real_std

        if noise_init is None:
            mu, logvar = self.encode(normalized_real_zero)
            z_vae = reparameterize(mu, logvar, self.training)
        else:
            z_vae = noise_init

        z_vae = self.decoder(z_vae)
        un_normalized_z_vae = z_vae * real_std + real_mean
        vae_out = torch.tanh(un_normalized_z_vae)

        vae_out_features = VGG(utils.upscale_2d(vae_out, 2, self.opt))
        real_zero_features = VGG(utils.upscale_2d(real_zero, 2, self.opt))
        features_loss = self.l1_loss(vae_out_features, real_zero_features)

        if sample_init is not None:
            x_prev_out, gan_features_loss = self.refinement_layers(real_zero, sample_init[0], sample_init[1], noise_amp, mode)
        else:
            x_prev_out, gan_features_loss = self.refinement_layers(real_zero, 0, vae_out, noise_amp, mode)

        features_loss += gan_features_loss

        if noise_init is None:
            return x_prev_out, vae_out, features_loss, (mu, logvar)
        else:
            return x_prev_out, vae_out, features_loss

    def _calc_vgg_features(self, tensor, scale_idx):
        if scale_idx == 0:
            tensor = utils.upscale_2d(tensor, scale_idx + 2, self.opt)
        features = VGG(tensor)
        for i in range(scale_idx):
            features = self.max_pool_2d(features)

        features = features.flatten(start_dim=1)
        features = F.normalize(features, p=2, dim=1)
        return features

    def refinement_layers(self, real_zero, start_idx, x_prev_out, noise_amp, mode):
        global VGG_CACHE

        features_loss = torch.tensor(0, dtype=torch.float32, device=self.opt.device)

        real_hash = real_zero.__hash__()

        for idx, block in enumerate(self.body[start_idx:], start_idx):
            if self.opt.vae_levels == idx + 1 and not self.opt.train_all:
                x_prev_out.detach_()

            # Upscale
            x_prev_out_up = utils.upscale_2d(x_prev_out, idx + 1, self.opt)
            real_up = utils.upscale_2d(real_zero, idx + 1, self.opt)

            real_cache_key = (idx, real_hash)

            if real_cache_key in VGG_CACHE:
                real_features = VGG_CACHE[real_cache_key]
            else:
                real_features = self._calc_vgg_features(real_up, idx)
                VGG_CACHE[real_cache_key] = real_features

            x_prev_features = self._calc_vgg_features(x_prev_out_up, idx)

            features_loss += (self.l1_loss(x_prev_features, real_features))

            noise = utils.generate_noise(ref=x_prev_out_up)
            x_prev = block((noise * noise_amp[idx + 1], x_prev_out_up))

            x_prev_out = torch.tanh(x_prev)

        return x_prev_out, features_loss


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
