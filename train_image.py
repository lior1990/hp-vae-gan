import matplotlib

from modules.sr import SRGenerator
from train_sr import train_sr, eval_sr
from utils.ema import ExponentialMovingAverage
from modules.networks_2d import MultiScaleDiscriminator

matplotlib.use("Agg")

import argparse
import utils
import random
import os

from utils import logger, tools
import logging
import colorama

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from modules import networks_2d
from modules.losses import kl_criterion
from modules.utils import calc_gradient_penalty
from datasets.image import SingleImageDataset, MultipleImageDataset

clear = colorama.Style.RESET_ALL
blue = colorama.Fore.CYAN + colorama.Style.BRIGHT
green = colorama.Fore.GREEN + colorama.Style.BRIGHT
magenta = colorama.Fore.MAGENTA + colorama.Style.BRIGHT


def train(opt, netG):
    if opt.vae_levels < opt.scale_idx + 1:
        print(f"Starting GAN training at scale {opt.scale_idx}")
        if opt.scale_idx > opt.ingan_disc_start_scale:
            print("InGAN Disc")
            D_curr = MultiScaleDiscriminator(opt.ingan_disc_n_scales, opt.padding_mode).to(opt.device)
        else:
            D_curr = getattr(networks_2d, opt.discriminator)(opt).to(opt.device)
            if opt.vae_levels < opt.scale_idx:
                if (opt.netG != '') and opt.resumed_idx != -1:
                    D_curr.load_state_dict(
                        torch.load('{}/netD_{}.pth'.format(opt.resume_dir, opt.scale_idx - 1))['state_dict'])
                else:
                    missing_keys, unexpected_keys = D_curr.load_state_dict(
                        torch.load('{}/netD_{}.pth'.format(opt.saver.experiment_dir, opt.scale_idx - 1))['state_dict'],
                        strict=False)
                    if unexpected_keys:
                        raise ValueError(unexpected_keys)

        # Current optimizers
        optimizerD = optim.Adam(D_curr.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

    if opt.residual_loss_start_scale <= opt.scale_idx:
        residual_loss_weight = opt.residual_loss_weight

        if opt.residual_loss_scale_factor > 0:
            residual_loss_weight = round(opt.residual_loss_weight ** (opt.residual_loss_scale_factor * opt.scale_idx), 2)

        print(f"residual_loss_weight: {residual_loss_weight}")

    parameter_list = []
    # Generator Adversary

    if not opt.train_all:
        if opt.scale_idx > 0:
            train_depth = opt.train_depth
            for idx, block in enumerate(netG.body[-train_depth:]):
                lr = max(opt.lr_g * (opt.lr_scale ** (len(netG.body) - 1 - idx)), opt.min_lr_g)
                parameter_list.append(
                    {
                        "params": block.parameters(),
                        "lr": lr,
                    }
                )
                print(f"Learning rate for block {idx} at scale {opt.scale_idx} is {lr}")
        else:
            # VQVAE
            parameter_list += [{"params": netG.vqvae_encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.vector_quantization.parameters(),
                                "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.decoder.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}]
    else:
        if len(netG.body) < opt.train_depth:
            parameter_list += [{"params": netG.vqvae_encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.vector_quantization.parameters(),
                                "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.decoder.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}]
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body) - 1 - idx))}
                for idx, block in enumerate(netG.body)]
        else:
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-opt.train_depth:])]

    optimizerG = optim.Adam(parameter_list, lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # Parallel
    is_parallel = torch.cuda.device_count() > 1 and dynamic_batch_size > 1
    if is_parallel:
        print("Working with data parallel")
        G_curr = torch.nn.DataParallel(netG)
        if opt.vae_levels < opt.scale_idx + 1:
            D_curr = torch.nn.DataParallel(D_curr)
    else:
        G_curr = netG

    progressbar_args = {
        "iterable": range(opt.niter),
        "desc": "Training scale [{}/{}]".format(opt.scale_idx + 1, opt.stop_scale + 1),
        "train": True,
        "offset": 0,
        "logging_on_update": False,
        "logging_on_close": True,
        "postfix": True
    }
    epoch_iterator = tools.create_progressbar(**progressbar_args)

    iterator = iter(opt.data_loader)
    reference_iterator = iter(ref_data_loader)

    for iteration in epoch_iterator:
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(opt.data_loader)
            data = next(iterator)

        if opt.scale_idx > 0:
            real, real_zero = data
            real = real.to(opt.device)
            real_zero = real_zero.to(opt.device)
        else:
            real = data
            real = real.to(opt.device)
            real_zero = real

        try:
            ref_data = next(reference_iterator)
        except StopIteration:
            reference_iterator = iter(ref_data_loader)
            ref_data = next(reference_iterator)

        if opt.scale_idx > 0:
            ref_real, ref_real_zero = ref_data
            ref_real_zero = ref_real_zero.to(opt.device)
        else:
            ref_real = ref_data
            ref_real_zero = ref_data
            ref_real_zero = ref_real_zero.to(opt.device)

        ############################
        # calculate noise_amp
        ###########################
        if iteration == 0:
            if opt.const_amp > 0:
                opt.Noise_Amps.append(opt.const_amp)
            else:
                with torch.no_grad():
                    if opt.scale_idx == 0:
                        opt.noise_amp = 1
                        opt.Noise_Amps.append(opt.noise_amp)
                    else:
                        opt.Noise_Amps.append(0)
                        z_reconstruction = G_curr(real_zero, opt.Noise_Amps, mode="rec")[0]

                        RMSE = torch.sqrt(F.mse_loss(real, z_reconstruction))
                        opt.noise_amp = opt.noise_amp_init * RMSE.item() / opt.batch_size
                        opt.Noise_Amps[-1] = opt.noise_amp

        ############################
        # (1) Update VAE network
        ###########################
        total_loss = 0

        generated, embedding_loss, encoding_indices, _, _ = G_curr(real_zero, [], mode="rec")

        if opt.vae_levels >= opt.scale_idx + 1:
            rec_vae_loss = opt.rec_loss(generated, real)
            vqvae_loss = opt.rec_weight * rec_vae_loss + embedding_loss

            total_loss += vqvae_loss
        else:
            ############################
            # (2) Update D network: maximize D(x) + D(G(z))
            ###########################
            # train with real
            #################

            # Train 3D Discriminator
            for _ in range(opt.d_steps):
                D_curr.zero_grad()
                real_discrimination_map = D_curr(real)
                errD_real = -real_discrimination_map.mean()

                # train with fake
                #################
                fake, fake_embedding_loss, _, last_residual_tuple, fake_z_e = G_curr(None, opt.Noise_Amps, mode=opt.fake_mode, reference_img=ref_real_zero)

                # Train 3D Discriminator
                output = D_curr(fake.detach())
                errD_fake = output.mean()

                gradient_penalty = calc_gradient_penalty(D_curr, real, fake, opt.lambda_grad, opt.device)
                errD_total = errD_real + errD_fake + gradient_penalty
                errD_total.backward()
                optimizerD.step()

            ############################
            # (3) Update G network: maximize D(G(z))
            ###########################
            errG_total = 0

            if opt.scale_idx == 0:
                # support GAN training in scale 0
                errG_total += embedding_loss + fake_embedding_loss

            if opt.residual_loss_start_scale <= opt.scale_idx:
                residual_result, prev_result = last_residual_tuple
                # minimize changes in between residual blocks
                residual_blocks_diff_loss = opt.rec_loss(residual_result, prev_result)
                errG_total += residual_blocks_diff_loss * residual_loss_weight

            if opt.indices_cycle_loss:
                fake_zero_scale = F.interpolate(fake, size=real_zero.shape[2:])
                fake_zero_scale_z_e = G_curr.encode(fake_zero_scale)
                indices_cycle_loss = opt.rec_loss(fake_z_e.detach(), fake_zero_scale_z_e)
                errG_total += indices_cycle_loss

            rec_loss = opt.rec_loss(generated, real)  # todo: remove this in G? add perceptual loss?
            errG_total += opt.rec_weight * rec_loss

            # Train with 3D Discriminator
            fake_discrimination_map = D_curr(fake)
            errG = -fake_discrimination_map.mean() * opt.disc_loss_weight
            errG_total += errG

            total_loss += errG_total

        G_curr.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(G_curr.parameters(), opt.grad_clip)
        optimizerG.step()

        # Update progress bar
        epoch_iterator.set_description('Scale [{}/{}], Iteration [{}/{}]'.format(
            opt.scale_idx + 1, opt.stop_scale + 1,
            iteration + 1, opt.niter,
        ))

        if opt.visualize:
            # Tensorboard
            opt.summary.add_scalar('Video/Scale {}/noise_amp'.format(opt.scale_idx), opt.noise_amp, iteration)
            if opt.vae_levels >= opt.scale_idx + 1:
                opt.summary.add_scalar('Video/Scale {}/embedding loss'.format(opt.scale_idx), embedding_loss.item(), iteration)
            else:
                opt.summary.add_scalar('Video/Scale {}/rec loss'.format(opt.scale_idx), rec_loss.item(), iteration)
            if opt.vae_levels < opt.scale_idx + 1:
                opt.summary.add_scalar('Video/Scale {}/errG'.format(opt.scale_idx), errG.item(), iteration)
                opt.summary.add_scalar('Video/Scale {}/errD_fake'.format(opt.scale_idx), errD_fake.item(), iteration)
                opt.summary.add_scalar('Video/Scale {}/errD_real'.format(opt.scale_idx), errD_real.item(), iteration)
                if opt.residual_loss_start_scale <= opt.scale_idx:
                    opt.summary.add_scalar('Video/Scale {}/Residual diff loss'.format(opt.scale_idx),
                                           residual_blocks_diff_loss.item(), iteration)
            else:
                opt.summary.add_scalar('Video/Scale {}/Rec VAE'.format(opt.scale_idx), rec_vae_loss.item(), iteration)

            if opt.indices_cycle_loss:
                opt.summary.add_scalar('Video/Scale {}/Indices cycle loss'.format(opt.scale_idx), indices_cycle_loss.item(), iteration)

            if iteration % opt.print_interval == 0:
                with torch.no_grad():
                    fake_var = []

                    G_curr.eval()
                    for _ in range(3):
                        fake, _, ref_encoding_indices, _, _ = G_curr(None, opt.Noise_Amps, mode=opt.fake_mode, reference_img=ref_real_zero)
                        fake_var.append(fake)
                    fake_var = torch.cat(fake_var, dim=0)
                    G_curr.train()

                opt.summary.visualize_image(opt, iteration, real, 'Real')
                opt.summary.visualize_image(opt, iteration, generated, 'Generated')
                opt.summary.visualize_image(opt, iteration, encoding_indices, 'Generated Indices', dim=3)
                opt.summary.visualize_image(opt, iteration, fake_var, 'Fake var')
                opt.summary.visualize_image(opt, iteration, ref_encoding_indices, 'Ref Indices', dim=3)
                opt.summary.visualize_image(opt, iteration, ref_real, 'Ref real')
                if opt.vae_levels < opt.scale_idx + 1:
                    opt.summary.visualize_image(opt, iteration, fake, 'Fake')
                    opt.summary.visualize_image(opt, iteration, real_discrimination_map.squeeze(dim=1), 'Real D map', dim=3)
                    opt.summary.visualize_image(opt, iteration, fake_discrimination_map.squeeze(dim=1), 'Fake D map', dim=3)

    epoch_iterator.close()

    opt.saver.save_checkpoint({'data': opt.Noise_Amps}, 'Noise_Amps.pth')
    # Save data
    opt.saver.save_checkpoint({
        'scale': opt.scale_idx,
        'state_dict': netG.state_dict(),
        'optimizer': optimizerG.state_dict(),
    }, 'netG.pth')
    if opt.vae_levels < opt.scale_idx + 1:
        opt.saver.save_checkpoint({
            'scale': opt.scale_idx,
            'state_dict': D_curr.module.state_dict() if is_parallel else D_curr.state_dict(),
            'optimizer': optimizerD.state_dict(),
        }, 'netD_{}.pth'.format(opt.scale_idx))

    samples_folder = os.path.join(opt.saver.experiment_dir, f"generated_images_{opt.scale_idx}")
    os.makedirs(samples_folder, exist_ok=True)
    eval_netG(opt.eval_dataset, samples_folder, opt, netG)


def eval_netG(image_path, save_dir, opt, netG):
    import matplotlib.pyplot as plt

    original_image_path = opt.image_path
    original_rep = opt.data_rep
    original_hflip = opt.hflip
    opt.hflip = False  # disable hflip in eval
    opt.image_path = image_path
    opt.data_rep = 1
    dataset = MultipleImageDataset(opt)
    test_data_loader = DataLoader(dataset, batch_size=1, num_workers=0)

    fakes_folder = os.path.join(save_dir, "fakes")
    reals_folder = os.path.join(save_dir, "reals")
    os.makedirs(fakes_folder, exist_ok=True)
    os.makedirs(reals_folder, exist_ok=True)

    netG.eval()
    with torch.no_grad():
        def norm(t):
            def norm_ip(img, min, max):
                img.clamp_(min=min, max=max)
                img.add_(-min).div_(max - min + 1e-5)

            norm_ip(t, float(t.min()), float(t.max()))

        def tensor_to_plot(t):
            norm(t)
            return t.squeeze().cpu().permute((1, 2, 0)).numpy()

        for idx, img_tup in enumerate(test_data_loader):
            fig, axes = plt.subplots(1, 2, figsize=(20, 5))

            if opt.scale_idx > 0:
                real, real_zero = img_tup
            else:
                real_zero = img_tup
                real = real_zero

            for plot_idx in range(2):
                axes[plot_idx].set_xticks([])
                axes[plot_idx].set_yticks([])

            real_zero = real_zero.to(opt.device)
            rec_output = netG(real_zero, opt.Noise_Amps, mode="rec")[0]

            real_tensor_to_plot = tensor_to_plot(real)
            rec_tensor_to_plot = tensor_to_plot(rec_output)
            axes[0].imshow(real_tensor_to_plot)
            axes[1].imshow(rec_tensor_to_plot)
            fig.savefig(os.path.join(save_dir, f"{idx}.png"))  # save the figure to file
            plt.close(fig)
            plt.imsave(os.path.join(reals_folder, f"real_{idx}.png"), real_tensor_to_plot)
            plt.imsave(os.path.join(fakes_folder, f"reconstruction_{idx}.png"), rec_tensor_to_plot)

    netG.train()

    opt.image_path = original_image_path
    opt.data_rep = original_rep
    opt.hflip = original_hflip


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # load, input, save configurations:
    parser.add_argument('--netG', default='', help='path to netG (to continue training)')
    parser.add_argument('--netD', default='', help='path to netD (to continue training)')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    # networks hyper parameters:
    parser.add_argument('--nc-im', type=int, default=3, help='# channels')
    parser.add_argument('--nfc', type=int, default=64, help='model basic # channels')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dim size')
    parser.add_argument('--vae-levels', type=int, default=3, help='# VAE levels')
    parser.add_argument('--enc-blocks', type=int, default=2, help='# encoder blocks')
    parser.add_argument('--ker-size', type=int, default=3, help='kernel size')
    parser.add_argument('--num-layer', type=int, default=5, help='number of layers')
    parser.add_argument('--stride', default=1, help='stride')
    parser.add_argument('--padd-size', type=int, default=1, help='net pad size')
    parser.add_argument('--padding-mode', type=str, default="zeros", help='net padding mode')
    parser.add_argument('--encoder-normalization-method', type=str, default="spectral", help='encoder normalization method')
    parser.add_argument('--decoder-normalization-method', type=str, default="batch", help='decoder normalization method')
    parser.add_argument('--g-normalization-method', type=str, default="batch", help='generator normalization method')
    parser.add_argument('--generator', type=str, default='GeneratorHPVAEGAN', help='generator model')
    parser.add_argument('--discriminator', type=str, default='WDiscriminator2D', help='discriminator model')

    # pyramid parameters:
    parser.add_argument('--scale-factor', type=float, default=0.75, help='pyramid scale factor')
    parser.add_argument('--noise_amp', type=float, default=0.1, help='addative noise cont weight')
    parser.add_argument('--min-size', type=int, default=32, help='image minimal size at the coarser scale')
    parser.add_argument('--max-size', type=int, default=256, help='image minimal size at the coarser scale')
    parser.add_argument('--pooling', action='store_true', default=False, help='pooling in encoder&decoder')
    parser.add_argument('--interpolation-method', type=str, default="bilinear", help="upscale interpolation method")
    parser.add_argument('--fixed-scales', action='store_true', default=True, help='use hard-coded scales')
    parser.add_argument('--fake-mode', type=str, default="rec", help='fake mode (rec/rec_noise)')

    # optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=50000, help='number of iterations to train per scale')
    parser.add_argument('--lr-g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--min-lr-g', type=float, default=0.00001, help='min learning rate for G')
    parser.add_argument('--lr-d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lambda-grad', type=float, default=0.1, help='gradient penelty weight')
    parser.add_argument('--rec-weight', type=float, default=10., help='reconstruction loss weight')
    parser.add_argument('--kl-weight', type=float, default=1., help='reconstruction loss weight')
    parser.add_argument('--disc-loss-weight', type=float, default=1.0, help='discriminator weight')
    parser.add_argument('--lr-scale', type=float, default=0.8, help='scaling of learning rate for lower stages')
    parser.add_argument('--train-depth', type=int, default=1, help='how many layers are trained if growing')
    parser.add_argument('--grad-clip', type=float, default=5, help='gradient clip')
    parser.add_argument('--const-amp', type=float, default=0, help='constant noise amplitude')
    parser.add_argument('--train-all', action='store_true', default=False, help='train all levels w.r.t. train-depth')
    parser.add_argument('--ingan-disc-n-scales', type=int, default=4)
    parser.add_argument('--ingan-disc-start-scale', type=int, default=100)
    parser.add_argument('--d-steps', type=int, default=1, help='D steps before G')
    parser.add_argument('--residual-loss-start-scale', type=int, default=1)
    parser.add_argument('--residual-loss-weight', type=float, default=1.1)
    parser.add_argument('--residual-loss-scale-factor', type=float, default=1.1)
    parser.add_argument('--indices-cycle-loss', action='store_true', default=False)
    parser.add_argument('--sr-start-scale', type=int, default=6)

    # Dataset
    parser.add_argument('--image-path', required=True, help="image path")
    parser.add_argument('--ref-image-path', required=False, help="image path", default="data/imgs/mountains")
    parser.add_argument('--hflip', action='store_true', default=False, help='horizontal flip')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--stop-scale-time', type=int, default=-1)
    parser.add_argument('--data-rep', type=int, default=1000, help='data repetition')

    # main arguments
    parser.add_argument('--checkname', type=str, default='DEBUG', help='check name')
    parser.add_argument('--mode', default='train', help='task to be done')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--print-interval', type=int, default=100, help='print interva')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize using tensorboard')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda')

    # vq vae arguments
    parser.add_argument('--n_embeddings', type=int, default=100, help='number of embeddings (keys of the vqvae dict)')
    parser.add_argument('--embedding_dim', type=int, default=64, help='embedding dimension (values of the vqvae dict)')
    parser.add_argument('--vqvae_beta', type=float, default=.25)
    parser.add_argument('--positional_encoding_weight', type=int, default=1)
    parser.add_argument('--eval_dataset', type=str, default="data/imgs/misc")
    parser.add_argument('--reduce-batch-interval', type=int, default=15)

    parser.set_defaults(hflip=False)
    opt = parser.parse_args()

    assert opt.disc_loss_weight > 0
    assert opt.residual_loss_start_scale > 0

    if opt.data_rep < opt.batch_size:
        opt.data_rep = opt.batch_size

    # Define Saver
    opt.saver = utils.ImageSaver(opt)

    opt.summary = utils.TensorboardSummary(opt.saver.experiment_dir)

    logger.configure_logging(os.path.abspath(os.path.join(opt.saver.experiment_dir, 'logbook.txt')))

    # CUDA
    device = 'cuda' if torch.cuda.is_available() and not opt.no_cuda else 'cpu'
    opt.device = device
    if torch.cuda.is_available() and device == 'cpu':
        logging.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Initial config
    opt.noise_amp_init = opt.noise_amp
    opt.scale_factor_init = opt.scale_factor

    # Adjust scales
    utils.adjust_scales2image(opt.img_size, opt)

    # Manual seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    logging.info("Random Seed: {}".format(opt.manualSeed))
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # Reconstruction loss
    opt.rec_loss = torch.nn.MSELoss()

    # Initial parameters
    opt.scale_idx = 0
    opt.nfc_prev = 0
    opt.Noise_Amps = []

    # Date
    if os.path.isdir(opt.image_path):
        dataset = MultipleImageDataset(opt)
    elif os.path.isfile(opt.image_path):
        dataset = SingleImageDataset(opt)
    else:
        raise NotImplementedError

    data_loader = DataLoader(dataset,
                             shuffle=True,
                             drop_last=True,
                             batch_size=opt.batch_size,
                             num_workers=0)

    ref_dataset = MultipleImageDataset(opt, load_ref_image=True)
    ref_data_loader = DataLoader(ref_dataset,
                                 shuffle=True,
                                 drop_last=False,
                                 batch_size=opt.batch_size,
                                 num_workers=0)

    if opt.stop_scale_time == -1:
        opt.stop_scale_time = opt.stop_scale

    opt.dataset = dataset
    opt.data_loader = data_loader

    with logger.LoggingBlock("Commandline Arguments", emph=True):
        for argument, value in sorted(vars(opt).items()):
            if type(value) in (str, int, float, tuple, list, bool):
                logging.info('{}: {}'.format(argument, value))

    with logger.LoggingBlock("Experiment Summary", emph=True):
        video_file_name, checkname, experiment = os.path.normpath(opt.saver.experiment_dir).split(os.path.sep)[-3:]
        logging.info("{}Checkname  :{} {}{}".format(magenta, clear, checkname, clear))
        logging.info("{}Experiment :{} {}{}".format(magenta, clear, experiment, clear))

        with logger.LoggingBlock("Commandline Summary", emph=True):
            logging.info("{}Generator      :{} {}{}".format(blue, clear, opt.generator, clear))
            logging.info("{}Iterations     :{} {}{}".format(blue, clear, opt.niter, clear))
            logging.info("{}Rec. Weight    :{} {}{}".format(blue, clear, opt.rec_weight, clear))

    # Current networks
    assert hasattr(networks_2d, opt.generator)
    netG = getattr(networks_2d, opt.generator)(opt).to(opt.device)

    if opt.netG != '':
        if not os.path.isfile(opt.netG):
            raise RuntimeError("=> no <G> checkpoint found at '{}'".format(opt.netG))
        checkpoint = torch.load(opt.netG)
        opt.scale_idx = checkpoint['scale']
        opt.resumed_idx = checkpoint['scale']
        opt.resume_dir = os.sep.join(opt.netG.split(os.sep)[:-1])
        for _ in range(opt.scale_idx):
            netG.init_next_stage()
        netG.load_state_dict(checkpoint['state_dict'])
        opt.scale_idx += 1
        # NoiseAmp
        opt.Noise_Amps = torch.load(os.path.join(opt.resume_dir, 'Noise_Amps.pth'))['data']
    else:
        opt.resumed_idx = -1

    dynamic_batch_size = opt.batch_size

    while opt.scale_idx < opt.stop_scale + 1:
        if opt.scale_idx >= opt.sr_start_scale:
            print("Starting SR training")
            sr_generator = SRGenerator()
            sr_generator.to(opt.device)
            train_sr(opt, sr_generator)
            eval_sr(opt, sr_generator, netG)
        else:
            if opt.scale_idx > 0:
                netG.init_next_stage()
            netG.to(opt.device)

            if opt.scale_idx > 0 and opt.scale_idx % opt.reduce_batch_interval == 0 and opt.batch_size > 1:
                # memory limitations
                new_batch_size = max(dynamic_batch_size // 2, 1)
                print(f"Reducing batch size from {dynamic_batch_size} to {new_batch_size}")
                dynamic_batch_size = new_batch_size
                opt.data_loader = DataLoader(dataset, batch_size=dynamic_batch_size, num_workers=0)
                ref_data_loader = DataLoader(ref_dataset, batch_size=dynamic_batch_size, num_workers=0)

            train(opt, netG)

        # Increase scale
        opt.scale_idx += 1
        if opt.resumed_idx != -1:
            opt.resumed_idx = -1
