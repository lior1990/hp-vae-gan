from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import numpy as np
import copy
import utils
from modules.spade_block import SPADEResnetBlock
from vqvae.new_quantizer import Codebook
from vqvae.quantizer import VectorQuantizer
from vqvae.vqvae2 import VQVAE2


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
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn="batch", act='lrelu', padding_mode="zeros"):
        super(ConvBlock2D, self).__init__()

        if bn =="spectral":
            self.add_module('conv', nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=ker_size,
                                          stride=stride, padding=padding, padding_mode=padding_mode)))
        else:
            self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size,
                                              stride=stride, padding=padding, padding_mode=padding_mode))
            if bn == "batch":
                self.add_module('norm', nn.BatchNorm2d(out_channel))
            elif bn == "instance":
                self.add_module('norm', nn.InstanceNorm2d(out_channel))
            elif bn == "group":
                self.add_module('norm', nn.GroupNorm(32, out_channel))
            elif bn in [False, "none", None]:
                pass
            else:
                raise NotImplementedError

        if act is not None:
            self.add_module(act, get_activation(act))


class ConvBlock2DSN(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn="spectral", act='lrelu', pooling=False, padding_mode="zeros"):
        super(ConvBlock2DSN, self).__init__()
        if bn == "spectral":
            self.add_module('conv', nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=ker_size,
                                                                     stride=stride, padding=padding, padding_mode=padding_mode)))
        elif bn == "instance":
            self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size,
                                                                     stride=stride, padding=padding, padding_mode=padding_mode))
            self.add_module("norm", nn.InstanceNorm2d(out_channel))
        elif bn == "group":
            self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size,
                                                                     stride=stride, padding=padding, padding_mode=padding_mode))
            self.add_module('norm', nn.GroupNorm(32, out_channel))
        elif bn == "batch":
            self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size,
                                                                     stride=stride, padding=padding, padding_mode=padding_mode))
            self.add_module("norm", nn.BatchNorm2d(out_channel))
        else:
            raise NotImplementedError

        if act is not None:
            self.add_module(act, get_activation(act))

        if pooling:
            self.add_module("pooling", torch.nn.AvgPool2d(2))


class FeatureExtractor(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, padding_mode, normalization_method, num_blocks=2, return_linear=False, pooling=False):
        super(FeatureExtractor, self).__init__()
        self.add_module('conv_block_0', ConvBlock2DSN(in_channel, out_channel, ker_size, padding, stride, padding_mode=padding_mode, bn=normalization_method))
        for i in range(num_blocks - 1):
            self.add_module('conv_block_{}'.format(i + 1),
                            ConvBlock2DSN(out_channel, out_channel, ker_size, padding, stride, pooling=pooling, padding_mode=padding_mode, bn=normalization_method))
        if return_linear:
            self.add_module('conv_block_{}'.format(num_blocks),
                            ConvBlock2DSN(out_channel, out_channel, ker_size, padding, stride, bn=False, act=None, padding_mode=padding_mode))
        else:
            self.add_module('conv_block_{}'.format(num_blocks),
                            ConvBlock2DSN(out_channel, out_channel, ker_size, padding, stride, padding_mode=padding_mode, bn=normalization_method))


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

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, opt.ker_size, opt.ker_size // 2, 1, opt.padding_mode, opt.encoder_normalization_method, num_blocks=num_blocks, pooling=opt.pooling)
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
        self.head = ConvBlock2DSN(opt.nc_im, N, opt.ker_size, opt.ker_size // 2, stride=1, bn=opt.d_normalization_method, act='lrelu',
                                  padding_mode=opt.padding_mode)
        self.body = nn.Sequential()
        for i in range(opt.num_layer):
            block = ConvBlock2DSN(N, N, opt.ker_size, opt.ker_size // 2, stride=1, bn=opt.d_normalization_method, act='lrelu', padding_mode=opt.padding_mode)
            self.body.add_module('block%d' % (i), block)
        self.tail = nn.Conv2d(N, 1, kernel_size=opt.ker_size, padding=1, stride=1, padding_mode=opt.padding_mode)

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
        self.num_layer = opt.num_layer

        vqvae_embedding_dim = opt.embedding_dim + 2*opt.positional_encoding_weight  # 2 for positional encoding
        self.vqvae_encode = Encode2DVQVAE(opt, out_dim=opt.embedding_dim, num_blocks=opt.enc_blocks)

        if opt.vqvae_version == "new":
            vqvae_class = Codebook
        elif opt.vqvae_version == "old":
            vqvae_class = VectorQuantizer
        elif opt.vqvae_version == "vqvae2":
            vqvae_class = VQVAE2
            assert opt.fake_mode == "rec", "Other modes are not supported yet"
        else:
            raise NotImplementedError

        self.vector_quantization = vqvae_class(opt.n_embeddings, vqvae_embedding_dim)
        self.decoder = nn.Sequential()

        # Normal Decoder
        self.decoder.add_module('head', ConvBlock2D(vqvae_embedding_dim, N, opt.ker_size, opt.padd_size, stride=1, padding_mode=opt.padding_mode, bn=opt.decoder_normalization_method))
        for i in range(opt.enc_blocks-1):
            if opt.pooling:
                block = UpsampleConvBlock2D(N, N, opt.ker_size, opt.padd_size, stride=1, bn=opt.decoder_normalization_method)
            else:
                block = ConvBlock2D(N, N, opt.ker_size, opt.padd_size, stride=1, padding_mode=opt.padding_mode, bn=opt.decoder_normalization_method)
            self.decoder.add_module('block%d' % (i), block)
        self.decoder.add_module('tail', nn.Conv2d(N, opt.nc_im, opt.ker_size, 1, opt.ker_size // 2, padding_mode=self.opt.padding_mode))

        self.body = torch.nn.ModuleList([])

    def init_next_stage(self):
        def create_regular_block():
            _stage = nn.Sequential()
            _stage.add_module('head',
                                    ConvBlock2D(self.opt.nc_im, self.N, self.opt.ker_size, self.opt.padd_size,
                                                stride=1, padding_mode=self.opt.padding_mode,
                                                bn=self.opt.g_normalization_method))
            for i in range(self.num_layer):
                block = ConvBlock2D(self.N, self.N, self.opt.ker_size, self.opt.padd_size, stride=1,
                                    padding_mode=self.opt.padding_mode, bn=self.opt.g_normalization_method)
                _stage.add_module('block%d' % (i), block)
            _stage.add_module('tail',
                                    nn.Conv2d(self.N, self.opt.nc_im, self.opt.ker_size, 1, self.opt.ker_size // 2,
                                              padding_mode=self.opt.padding_mode))
            return _stage

        def create_spade_seq():
            _stage = SPADESequential()

            _stage.add_module('head', SPADEResnetBlock(self.opt.n_embeddings, self.opt.nc_im, self.N, self.opt.ker_size, self.opt.g_normalization_method, use_spectral_norm=self.opt.spade_use_spectral))
            for i in range(self.opt.num_layer):
                block = SPADEResnetBlock(self.opt.n_embeddings, self.N, self.N, self.opt.ker_size, self.opt.g_normalization_method, use_spectral_norm=self.opt.spade_use_spectral)
                _stage.add_module('block%d' % (i), block)
            _stage.add_module('tail', SPADEResnetBlock(self.opt.n_embeddings, self.N, self.opt.nc_im, self.opt.ker_size, self.opt.g_normalization_method, use_spectral_norm=self.opt.spade_use_spectral))
            return _stage

        if self.opt.scale_idx in self.opt.spade_scales:
            self.body.append(create_spade_seq())
        else:
            self.body.append(create_regular_block())

    def forward(self, img, noise_amp, mode='rand', reference_img=None):
        img_to_encode = img if reference_img is None else reference_img
        # z_e = self.encode(img_to_encode)
        vqvae_out, embedding_loss, encoding_indices = self.vector_quantization(img, mode)
        # vqvae_out = torch.tanh(self.decoder(z_q))

        x_prev_out, last_residual_tuple = self.refinement_layers(0, vqvae_out, noise_amp, mode, encoding_indices)

        return x_prev_out, embedding_loss, encoding_indices, last_residual_tuple, None

    def forward_w_interpolation(self, img_all_scales, interpolation_indices):
        if interpolation_indices is None:
            interpolation_indices = {}

        mode = "rec"
        z_e = self.encode(img_all_scales[0])
        embedding_loss, z_q, _, _, _ = self.vector_quantization(z_e, mode)
        vqvae_out = torch.tanh(self.decoder(z_q))

        x_prev_out = self.refinement_layers_w_interpolation(vqvae_out, img_all_scales, interpolation_indices)

        return x_prev_out, embedding_loss

    def refinement_layers_w_interpolation(self, x_prev_out, img_all_scales, interpolation_indices: "Dict[int, float]"):
        for idx, block in enumerate(self.body):
            # Upscale
            x_prev_out_up = utils.upscale_2d(x_prev_out, idx + 1, self.opt)

            if idx in interpolation_indices:
                interpolation_value = interpolation_indices[idx]
                x_prev_out_up = img_all_scales[idx+1] * interpolation_value + x_prev_out_up * (1-interpolation_value)

            x_prev = block(x_prev_out_up)

            x_prev_out = torch.tanh(x_prev + x_prev_out_up)

        return x_prev_out

    def refinement_interpolation_at_scale(self, img_all_scales, start_scale):
        for idx, block in enumerate(self.body[start_scale:], start_scale):
            # Upscale
            if idx == start_scale:
                x_prev_out_up = img_all_scales[start_scale]
            else:
                x_prev_out_up = utils.upscale_2d(x_prev_out, idx + 1, self.opt)

            x_prev = block(x_prev_out_up)

            x_prev_out = torch.tanh(x_prev + x_prev_out_up)

        return x_prev_out

    def encode(self, img):
        z_e = self.vqvae_encode(img)
        positional_encoding = utils.convert_to_coord_format(z_e.shape[0], z_e.shape[-2], z_e.shape[-1], device=self.opt.device)
        z_e = torch.cat([z_e, positional_encoding.repeat(1, self.opt.positional_encoding_weight, 1, 1)], dim=1)
        return z_e

    def refinement_layers(self, start_idx, x_prev_out, noise_amp, mode, spade_img):
        last_residual_tuple = None  # the last pair of x after residual block and its prev

        bs, h, w = spade_img.size()
        nc = self.vector_quantization.n_codes
        input_label = torch.zeros(bs, nc, h, w, device=self.opt.device)
        input_label.scatter_(1, spade_img.unsqueeze(dim=1), 1.0)

        for idx, block in enumerate(self.body, 1):
            # Upscale
            x_prev_out_up = utils.upscale_2d(x_prev_out, idx, self.opt)

            # Add noise if "random" sampling, else, add no noise is "reconstruction" mode
            if mode == "rec_noise":
                noise = utils.generate_noise(ref=x_prev_out_up)
                if idx in self.opt.spade_scales:
                    x_prev = block((x_prev_out_up + noise * noise_amp[idx], input_label))
                else:
                    x_prev = block(x_prev_out_up + noise * noise_amp[idx])
            else:
                if idx in self.opt.spade_scales:
                    x_prev = block((x_prev_out_up, input_label))
                else:
                    x_prev = block(x_prev_out_up)
            x_prev_out = torch.tanh(x_prev + x_prev_out_up)

            last_residual_tuple = (x_prev, x_prev_out_up)

        return x_prev_out, last_residual_tuple


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_scales, padding_mode, scale_factor=2, base_channels=128, extra_conv_layers=0):
        super(MultiScaleDiscriminator, self).__init__()
        self.padding_mode = padding_mode
        self.base_channels = base_channels
        self.scale_factor = scale_factor
        self.extra_conv_layers = extra_conv_layers

        self.max_n_scales = n_scales
        # Prepare a list of all the networks for all the wanted scales
        self.nets = nn.ModuleList()

        # Create a network for each scale
        for _ in range(self.max_n_scales):
            self.nets.append(self.make_net())

        self.scales_weight = [1 / self.max_n_scales for _ in range(self.max_n_scales)]

    def make_net(self):
        base_channels = self.base_channels
        net = []

        # Entry block
        net += [nn.utils.spectral_norm(nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding_mode=self.padding_mode)),
                nn.BatchNorm2d(base_channels),
                nn.LeakyReLU(0.2, True)]

        # Downscaling blocks
        # A sequence of strided conv-blocks. Image dims shrink by 2, channels dim expands by 2 at each block
        net += [nn.utils.spectral_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding_mode=self.padding_mode)),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, True)]

        # Regular conv-block
        net += [nn.utils.spectral_norm(nn.Conv2d(in_channels=base_channels * 2,
                                                 out_channels=base_channels * 2,
                                                 kernel_size=3,
                                                 bias=True,
                                                 padding_mode=self.padding_mode)),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, True)]

        # Additional 1x1 conv-blocks
        for _ in range(self.extra_conv_layers):
            net += [nn.utils.spectral_norm(nn.Conv2d(in_channels=base_channels * 2,
                                                     out_channels=base_channels * 2,
                                                     kernel_size=3,
                                                     bias=True,
                                                     padding_mode=self.padding_mode)),
                    nn.BatchNorm2d(base_channels * 2),
                    nn.LeakyReLU(0.2, True)]

        # Final conv-block
        # Ends with a Sigmoid to get a range of 0-1
        net += nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, 1, kernel_size=1, padding_mode=self.padding_mode)),
                             nn.Sigmoid())

        # Make it a valid layers sequence and return
        return nn.Sequential(*net)

    def forward(self, input_tensor):
        scale_weights = self.scales_weight  # todo: set this as parameter and make it dynamic
        aggregated_result_maps_from_all_scales = self.nets[0](input_tensor) * scale_weights[0]
        map_size = aggregated_result_maps_from_all_scales.shape[2:]

        # Run all nets over all scales and aggregate the interpolated results
        size_delta = 10
        h, w = input_tensor.shape[-2:]
        for net, scale_weight, i in zip(self.nets[1:], scale_weights[1:], range(1, len(scale_weights))):
            h -= size_delta
            w -= size_delta
            downscaled_image = nn.functional.interpolate(input_tensor, size=(h, w), mode='bilinear')
            result_map_for_current_scale = net(downscaled_image)
            upscaled_result_map_for_current_scale = nn.functional.interpolate(result_map_for_current_scale,
                                                                              size=map_size,
                                                                              mode='bilinear')
            aggregated_result_maps_from_all_scales += upscaled_result_map_for_current_scale * scale_weight

        return aggregated_result_maps_from_all_scales


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_scales, scale_factor=2, base_channels=128, extra_conv_layers=0):
        super(MultiScaleDiscriminator, self).__init__()
        self.base_channels = base_channels
        self.scale_factor = scale_factor
        self.extra_conv_layers = extra_conv_layers

        self.max_n_scales = n_scales
        # Prepare a list of all the networks for all the wanted scales
        self.nets = nn.ModuleList()

        # Create a network for each scale
        for _ in range(self.max_n_scales):
            self.nets.append(self.make_net())

        self.scales_weight = [1 / self.max_n_scales for _ in range(self.max_n_scales)]

    def make_net(self):
        base_channels = self.base_channels
        net = []

        # Entry block
        net += [nn.utils.spectral_norm(nn.Conv2d(3, base_channels, kernel_size=3, stride=1)),
                nn.BatchNorm2d(base_channels),
                nn.LeakyReLU(0.2, True)]

        # Downscaling blocks
        # A sequence of strided conv-blocks. Image dims shrink by 2, channels dim expands by 2 at each block
        net += [nn.utils.spectral_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2)),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, True)]

        # Regular conv-block
        net += [nn.utils.spectral_norm(nn.Conv2d(in_channels=base_channels * 2,
                                                 out_channels=base_channels * 2,
                                                 kernel_size=3,
                                                 bias=True)),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, True)]

        # Additional 1x1 conv-blocks
        for _ in range(self.extra_conv_layers):
            net += [nn.utils.spectral_norm(nn.Conv2d(in_channels=base_channels * 2,
                                                     out_channels=base_channels * 2,
                                                     kernel_size=3,
                                                     bias=True)),
                    nn.BatchNorm2d(base_channels * 2),
                    nn.LeakyReLU(0.2, True)]

        # Final conv-block
        # Ends with a Sigmoid to get a range of 0-1
        net += nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, 1, kernel_size=1)),
                             nn.Sigmoid())

        # Make it a valid layers sequence and return
        return nn.Sequential(*net)

    def forward(self, input_tensor):
        scale_weights = self.scales_weight  # todo: set this as parameter and make it dynamic
        aggregated_result_maps_from_all_scales = self.nets[0](input_tensor) * scale_weights[0]
        map_size = aggregated_result_maps_from_all_scales.shape[2:]

        # Run all nets over all scales and aggregate the interpolated results
        size_delta = 10
        h, w = input_tensor.shape[-2:]
        for net, scale_weight, i in zip(self.nets[1:], scale_weights[1:], range(1, len(scale_weights))):
            h -= size_delta
            w -= size_delta
            downscaled_image = nn.functional.interpolate(input_tensor, size=(h, w), mode='bilinear')
            result_map_for_current_scale = net(downscaled_image)
            upscaled_result_map_for_current_scale = nn.functional.interpolate(result_map_for_current_scale,
                                                                              size=map_size,
                                                                              mode='bilinear')
            aggregated_result_maps_from_all_scales += upscaled_result_map_for_current_scale * scale_weight

        return aggregated_result_maps_from_all_scales


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
