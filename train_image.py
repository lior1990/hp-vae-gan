import matplotlib

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


def orig_train(opt, netG, data_loader, noise_amps):
    if opt.vae_levels < opt.scale_idx + 1:
        D_curr = getattr(networks_2d, opt.discriminator)(opt).to(opt.device)

        if (opt.netG != '') and (opt.resumed_idx == opt.scale_idx):
            D_curr.load_state_dict(
                torch.load('{}/netD_{}.pth'.format(opt.resume_dir, opt.scale_idx - 1))['state_dict'])
        elif opt.vae_levels < opt.scale_idx:
            D_curr.load_state_dict(
                torch.load('{}/netD_{}.pth'.format(opt.saver.experiment_dir, opt.scale_idx - 1))['state_dict'])

        # Current optimizers
        optimizerD = optim.Adam(D_curr.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

    parameter_list = []
    # Generator Adversary

    if not opt.train_all:
        if opt.vae_levels < opt.scale_idx + 1:
            train_depth = min(opt.train_depth, len(netG.body) - opt.vae_levels + 1)
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-train_depth:])]
        else:
            # VAE
            parameter_list += [{"params": netG.encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.decoder.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}]
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-opt.train_depth:])]
    else:
        if len(netG.body) < opt.train_depth:
            parameter_list += [{"params": netG.encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
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
    if opt.device == 'cuda':
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

    iterator = iter(data_loader)

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
            real = data.to(opt.device)
            real_zero = real

        initial_size = utils.get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size)
        initial_size = [int(initial_size * opt.ar), initial_size]
        opt.Z_init_size = [opt.batch_size, opt.latent_dim, *initial_size]

        noise_init = utils.generate_noise(size=opt.Z_init_size, device=opt.device)

        ############################
        # calculate noise_amp
        ###########################
        if iteration == 0:
            if opt.const_amp:
                opt.Noise_Amps.append(1)
            else:
                with torch.no_grad():
                    if opt.scale_idx == 0:
                        opt.noise_amp = 1
                        noise_amps.append(opt.noise_amp)
                    else:
                        noise_amps.append(0)
                        z_reconstruction, _, _ = G_curr(real_zero, noise_amps, mode="rec")

                        RMSE = torch.sqrt(F.mse_loss(real, z_reconstruction))
                        opt.noise_amp = opt.noise_amp_init * RMSE.item() / opt.batch_size
                        noise_amps[-1] = opt.noise_amp

        ############################
        # (1) Update VAE network
        ###########################
        total_loss = 0

        generated, generated_vae, (mu, logvar) = G_curr(real_zero, noise_amps, mode="rec")

        if opt.vae_levels >= opt.scale_idx + 1:
            rec_vae_loss = opt.rec_loss(generated, real) + opt.rec_loss(generated_vae, real_zero)
            kl_loss = kl_criterion(mu, logvar)
            vae_loss = opt.rec_weight * rec_vae_loss + opt.kl_weight * kl_loss

            total_loss += vae_loss
        else:
            ############################
            # (2) Update D network: maximize D(x) + D(G(z))
            ###########################
            # train with real
            #################

            # Train 3D Discriminator
            D_curr.zero_grad()
            output = D_curr(real)
            errD_real = -output.mean()

            # train with fake
            #################
            fake, _ = G_curr(noise_init, noise_amps, noise_init=noise_init, mode="rand")

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
            rec_loss = opt.rec_loss(generated, real)
            errG_total += opt.rec_weight * rec_loss

            # Train with 3D Discriminator
            output = D_curr(fake)
            errG = -output.mean() * opt.disc_loss_weight
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
                opt.summary.add_scalar('Video/Scale {}/KLD'.format(opt.scale_idx), kl_loss.item(), iteration)
            else:
                opt.summary.add_scalar('Video/Scale {}/rec loss'.format(opt.scale_idx), rec_loss.item(), iteration)
            opt.summary.add_scalar('Video/Scale {}/noise_amp'.format(opt.scale_idx), opt.noise_amp, iteration)
            if opt.vae_levels < opt.scale_idx + 1:
                opt.summary.add_scalar('Video/Scale {}/errG'.format(opt.scale_idx), errG.item(), iteration)
                opt.summary.add_scalar('Video/Scale {}/errD_fake'.format(opt.scale_idx), errD_fake.item(), iteration)
                opt.summary.add_scalar('Video/Scale {}/errD_real'.format(opt.scale_idx), errD_real.item(), iteration)
            else:
                opt.summary.add_scalar('Video/Scale {}/Rec VAE'.format(opt.scale_idx), rec_vae_loss.item(), iteration)

            if iteration % opt.print_interval == 0:
                with torch.no_grad():
                    fake_var = []
                    fake_vae_var = []
                    for _ in range(3):
                        noise_init = utils.generate_noise(ref=noise_init)
                        fake, fake_vae = G_curr(noise_init, noise_amps, noise_init=noise_init, mode="rand")
                        fake_var.append(fake)
                        fake_vae_var.append(fake_vae)
                    fake_var = torch.cat(fake_var, dim=0)
                    fake_vae_var = torch.cat(fake_vae_var, dim=0)

                opt.summary.visualize_image(opt, iteration, real, 'Real')
                opt.summary.visualize_image(opt, iteration, generated, 'Generated')
                opt.summary.visualize_image(opt, iteration, generated_vae, 'Generated VAE')
                opt.summary.visualize_image(opt, iteration, fake_var, 'Fake var')
                opt.summary.visualize_image(opt, iteration, fake_vae_var, 'Fake VAE var')

    epoch_iterator.close()
    return optimizerG, D_curr, optimizerD


def train(opt, data_loader, outer_iteration, noise_amps, generator_optimizer_tuple,
          discriminator_optimizer_tuple=None, validate=False, task_index: int = None):
    G_curr, optimizerG = generator_optimizer_tuple

    if discriminator_optimizer_tuple is not None:
        D_curr, optimizerD = discriminator_optimizer_tuple

    iterator = iter(data_loader)

    current_iteration_base = outer_iteration * opt.inner_iterations

    for iteration in range(opt.inner_iterations):
        current_iteration = current_iteration_base + iteration

        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            data = next(iterator)

        if opt.scale_idx > 0:
            real, real_zero = data
            real = real.to(opt.device)
            real_zero = real_zero.to(opt.device)
        else:
            real = data.to(opt.device)
            real_zero = real

        initial_size = utils.get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size)
        initial_size = [int(initial_size * opt.ar), initial_size]
        opt.Z_init_size = [opt.batch_size, opt.latent_dim, *initial_size]

        noise_init = utils.generate_noise(size=opt.Z_init_size, device=opt.device)

        ############################
        # calculate noise_amp
        ###########################
        if iteration == 0 and len(noise_amps) == opt.scale_idx:
            if opt.const_amp:
                noise_amps.append(1)
            else:
                with torch.no_grad():
                    if opt.scale_idx == 0:
                        opt.noise_amp = 1
                        noise_amps.append(opt.noise_amp)
                    else:
                        noise_amps.append(0)
                        z_reconstruction, _, _ = G_curr(real_zero, noise_amps, mode="rec")

                        RMSE = torch.sqrt(F.mse_loss(real, z_reconstruction))
                        opt.noise_amp = opt.noise_amp_init * RMSE.item() / opt.batch_size
                        noise_amps[-1] = opt.noise_amp

        ############################
        # (1) Update VAE network
        ###########################
        total_loss = 0

        generated, generated_vae, (mu, logvar) = G_curr(real_zero, noise_amps, mode="rec")

        if opt.vae_levels >= opt.scale_idx + 1:
            rec_vae_loss = opt.rec_loss(generated, real) + opt.rec_loss(generated_vae, real_zero)
            kl_loss = kl_criterion(mu, logvar)
            vae_loss = opt.rec_weight * rec_vae_loss + opt.kl_weight * kl_loss

            total_loss += vae_loss
        else:
            ############################
            # (2) Update D network: maximize D(x) + D(G(z))
            ###########################
            # train with real
            #################

            # Train 3D Discriminator
            D_curr.zero_grad()
            output = D_curr(real)
            errD_real = -output.mean()

            # train with fake
            #################
            fake, _ = G_curr(noise_init, noise_amps, noise_init=noise_init, mode="rand")

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
            rec_loss = opt.rec_loss(generated, real)
            errG_total += opt.rec_weight * rec_loss

            # Train with 3D Discriminator
            output = D_curr(fake)
            errG = -output.mean() * opt.disc_loss_weight
            errG_total += errG

            total_loss += errG_total

        G_curr.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(G_curr.parameters(), opt.grad_clip)
        optimizerG.step()

        if opt.visualize:
            # Tensorboard

            if current_iteration % opt.scalar_interval == 0:
                base_scalar_str = f"Image/Scale {opt.scale_idx}"
                if validate:
                    base_scalar_str = f"{base_scalar_str}/Validation/"

                # update scalars on scalar interval due to low disk space
                opt.summary.add_scalar(f'{base_scalar_str}/noise_amp', opt.noise_amp, current_iteration)
                if opt.vae_levels >= opt.scale_idx + 1:
                    opt.summary.add_scalar(f'{base_scalar_str}/KLD', kl_loss.item(), current_iteration)
                else:
                    opt.summary.add_scalar(f'{base_scalar_str}/rec loss', rec_loss.item(), current_iteration)

                if opt.vae_levels < opt.scale_idx + 1:
                    opt.summary.add_scalar(f'{base_scalar_str}/errG', errG.item(), current_iteration)
                    opt.summary.add_scalar(f'{base_scalar_str}/errD_fake', errD_fake.item(), current_iteration)
                    opt.summary.add_scalar(f'{base_scalar_str}/errD_real', errD_real.item(), current_iteration)
                else:
                    opt.summary.add_scalar(f'{base_scalar_str}/Rec VAE', rec_vae_loss.item(), current_iteration)

            if current_iteration % opt.print_interval == 0:
                if validate:
                    G_curr.eval()

                with torch.no_grad():
                    fake_var = []
                    fake_vae_var = []
                    for _ in range(3):
                        noise_init = utils.generate_noise(ref=noise_init)
                        fake, fake_vae = G_curr(noise_init, noise_amps, noise_init=noise_init, mode="rand")
                        fake_var.append(fake)
                        fake_vae_var.append(fake_vae)
                    fake_var = torch.cat(fake_var, dim=0)
                    fake_vae_var = torch.cat(fake_vae_var, dim=0)

                if validate:
                    G_curr.train()

                assert task_index is not None or validate
                task_index = f"{task_index}" if task_index is not None else "Validation"

                opt.summary.visualize_image(opt, current_iteration, real, f'{task_index}/Real')
                opt.summary.visualize_image(opt, current_iteration, generated, f'{task_index}/Reconstruction GAN')
                opt.summary.visualize_image(opt, current_iteration, generated_vae, f'{task_index}/Reconstruction VAE')
                opt.summary.visualize_image(opt, current_iteration, fake_var, f'{task_index}/Fake GAN')
                opt.summary.visualize_image(opt, current_iteration, fake_vae_var, f'{task_index}/Fake VAE')


def get_G_parameters_list(opt, netG):
    parameter_list = []
    # Generator Adversary
    if not opt.train_all:
        if opt.vae_levels < opt.scale_idx + 1:
            train_depth = min(opt.train_depth, len(netG.body) - opt.vae_levels + 1)
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-train_depth:])]
        else:
            # VAE
            parameter_list += [
                {"params": netG.encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                {"params": netG.decoder.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}]
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-opt.train_depth:])]
    else:
        if len(netG.body) < opt.train_depth:
            parameter_list += [
                {"params": netG.encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
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
    return parameter_list


def main():
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
    parser.add_argument('--generator', type=str, default='GeneratorHPVAEGAN', help='generator model')
    parser.add_argument('--discriminator', type=str, default='WDiscriminator2D', help='discriminator model')

    # pyramid parameters:
    parser.add_argument('--scale-factor', type=float, default=0.75, help='pyramid scale factor')
    parser.add_argument('--noise_amp', type=float, default=0.1, help='addative noise cont weight')
    parser.add_argument('--min-size', type=int, default=32, help='image minimal size at the coarser scale')
    parser.add_argument('--max-size', type=int, default=256, help='image minimal size at the coarser scale')

    # optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=50000, help='number of iterations to train per scale')
    parser.add_argument('--lr-g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr-d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lambda-grad', type=float, default=0.1, help='gradient penelty weight')
    parser.add_argument('--rec-weight', type=float, default=10., help='reconstruction loss weight')
    parser.add_argument('--kl-weight', type=float, default=1., help='reconstruction loss weight')
    parser.add_argument('--disc-loss-weight', type=float, default=1.0, help='discriminator weight')
    parser.add_argument('--lr-scale', type=float, default=0.2, help='scaling of learning rate for lower stages')
    parser.add_argument('--train-depth', type=int, default=1, help='how many layers are trained if growing')
    parser.add_argument('--grad-clip', type=float, default=5, help='gradient clip')
    parser.add_argument('--const-amp', action='store_true', default=False, help='constant noise amplitude')
    parser.add_argument('--train-all', action='store_true', default=False, help='train all levels w.r.t. train-depth')

    # Dataset
    parser.add_argument('--image-path', required=True, help="image path")
    parser.add_argument('--hflip', action='store_true', default=False, help='horizontal flip')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--stop-scale-time', type=int, default=-1)
    parser.add_argument('--data-rep', type=int, default=1000, help='data repetition')

    # main arguments
    parser.add_argument('--checkname', type=str, default='DEBUG', help='check name')
    parser.add_argument('--mode', default='train', help='task to be done')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--print-interval', type=int, default=99, help='print interva')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize using tensorboard')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda')

    # MAML arguments
    parser.add_argument('--outer-iterations', type=int, default=500)
    parser.add_argument('--inner-iterations', type=int, default=2000)
    parser.add_argument('--inner-learning_rate', type=float, default=1e-3)
    parser.add_argument('--outer-learning_rate', type=float, default=1e-4)
    parser.add_argument('--validation-interval', type=int, default=50)
    parser.add_argument('--logdir', type=str, default='', help='tensorboard alternative directory')
    parser.add_argument('--scalar-interval', type=int, default=1000, help='scalar interval update in tensorboard')

    parser.set_defaults(hflip=False)
    opt = parser.parse_args()

    assert opt.vae_levels > 0
    assert opt.disc_loss_weight > 0

    if opt.data_rep < opt.batch_size:
        opt.data_rep = opt.batch_size

    # Define Saver
    opt.saver = utils.ImageSaver(opt)

    # Define Tensorboard Summary
    tensorboard_dir = opt.saver.experiment_dir if opt.logdir == '' else opt.logdir
    opt.summary = utils.TensorboardSummary(tensorboard_dir)
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

    # Date
    datasets = [SingleImageDataset(opt, os.path.join(opt.image_path, image_path)) for image_path in os.listdir(opt.image_path)]
    number_of_tasks = len(datasets)
    noise_amps_list = [list() for _ in range(number_of_tasks)]

    if opt.stop_scale_time == -1:
        opt.stop_scale_time = opt.stop_scale

    opt.total_iterations = opt.outer_iterations * opt.inner_iterations

    with logger.LoggingBlock("Commandline Arguments", emph=True):
        for argument, value in sorted(vars(opt).items()):
            if type(value) in (str, int, float, tuple, list):
                logging.info('{}: {}'.format(argument, value))

    with logger.LoggingBlock("Experiment Summary", emph=True):
        video_file_name, checkname, experiment = os.path.normpath(opt.saver.experiment_dir).split(os.path.sep)[-3:]
        logging.info("{}Checkname  :{} {}{}".format(magenta, clear, checkname, clear))
        logging.info("{}Experiment :{} {}{}".format(magenta, clear, experiment, clear))

        with logger.LoggingBlock("Commandline Summary", emph=True):
            logging.info("{}Generator      :{} {}{}".format(blue, clear, opt.generator, clear))
            logging.info("{}Iterations     :{} {}{}".format(blue, clear, opt.total_iterations, clear))
            logging.info("{}Rec. Weight    :{} {}{}".format(blue, clear, opt.rec_weight, clear))

    # Current networks
    assert hasattr(networks_2d, opt.generator)
    netG_meta = getattr(networks_2d, opt.generator)(opt).to(opt.device)
    netG = getattr(networks_2d, opt.generator)(opt)  # on CPU since it never do a feed forward operation

    if opt.netG != '':
        if not os.path.isfile(opt.netG):
            raise RuntimeError("=> no <G> checkpoint found at '{}'".format(opt.netG))
        checkpoint = torch.load(opt.netG)
        opt.scale_idx = checkpoint['scale']
        opt.resumed_idx = checkpoint['scale']
        opt.resume_dir = os.sep.join(opt.netG.split(os.sep)[:-1])
        for _ in range(opt.scale_idx):
            netG.init_next_stage()
            netG_meta.init_next_stage()
        netG_meta.to(opt.device)
        netG.load_state_dict(checkpoint['state_dict'])
        # NoiseAmp
        noise_amps_list = torch.load(os.path.join(opt.resume_dir, 'Noise_Amps.pth'))['data']
    else:
        opt.resumed_idx = -1

    while opt.scale_idx < opt.stop_scale + 1:
        if (opt.scale_idx > 0) and (opt.resumed_idx != opt.scale_idx):
            netG.init_next_stage()
            netG_meta.init_next_stage()
            netG_meta.to(opt.device)

        regular_train = opt.vae_levels < opt.scale_idx + 1

        if regular_train:
            netG_meta.to("cpu")
            netG.to(opt.device)
            multiple_image_dataset = MultipleImageDataset(opt)
            data_loader = DataLoader(multiple_image_dataset,
                                     shuffle=False,
                                     batch_size=1,
                                     num_workers=0)
            noise_amps = noise_amps_list[0]  # some arbitrary selection of index...
            optimizerG, D_curr, optimizerD = orig_train(opt, netG, data_loader, noise_amps)
        else:
            if opt.vae_levels < opt.scale_idx + 1:
                D_curr_meta = getattr(networks_2d, opt.discriminator)(opt).to(opt.device)
                D_curr = getattr(networks_2d, opt.discriminator)(opt)  # on CPU

                if (opt.netG != '') and (opt.resumed_idx == opt.scale_idx):
                    D_curr_meta.load_state_dict(
                        torch.load('{}/netD_{}.pth'.format(opt.resume_dir, opt.scale_idx - 1))['state_dict'])
                elif opt.vae_levels < opt.scale_idx:
                    D_curr_meta.load_state_dict(
                        torch.load('{}/netD_{}.pth'.format(opt.saver.experiment_dir, opt.scale_idx - 1))['state_dict'])

                # Current optimizers
                optimizerD_meta = optim.Adam(D_curr_meta.parameters(), lr=opt.inner_learning_rate, betas=(opt.beta1, 0.999))
                optimizerD = optim.Adam(D_curr.parameters(), lr=opt.outer_learning_rate, betas=(opt.beta1, 0.999))

            optimizerG_meta = optim.Adam(get_G_parameters_list(opt, netG_meta), lr=opt.inner_learning_rate, betas=(opt.beta1, 0.999))
            optimizerG = optim.Adam(get_G_parameters_list(opt, netG), lr=opt.outer_learning_rate, betas=(opt.beta1, 0.999))

            G_curr_meta = netG_meta

            progressbar_args = {
                "iterable": range(opt.outer_iterations),
                "desc": "Training scale [{}/{}]. Outer [{}/{}].".format(opt.scale_idx + 1, opt.stop_scale + 1,
                                                                        1, opt.outer_iterations),
                "train": True,
                "offset": 0,
                "logging_on_update": False,
                "logging_on_close": True,
                "postfix": True
            }
            epoch_iterator = tools.create_progressbar(**progressbar_args)

            generator_optimizer_tuple = (G_curr_meta, optimizerG_meta)

            # 1. init meta generator and discriminator V
            # 2. outer for loop
                # 0. copy weights + set to train mode
                # 1. get random data loader
                # 2. get the corresponding noise amps
                # 3. train for x iterations
                # 4. update meta generator and discriminator weights
                # 5. once in a while - run validation and show visualizations

            discriminator_optimizer_tuple = None

            if opt.validation_interval > 0:
                task_indices = (torch.arange(opt.outer_iterations) % (number_of_tasks-1))[torch.randperm(opt.outer_iterations)]
            else:
                task_indices = (torch.arange(opt.outer_iterations) % number_of_tasks)[torch.randperm(opt.outer_iterations)]

            for outer_iteration in epoch_iterator:
                if opt.vae_levels < opt.scale_idx + 1:
                    discriminator_optimizer_tuple = (D_curr_meta, optimizerD_meta)
                    D_curr_meta.train()
                    D_curr_meta.load_state_dict(D_curr.state_dict())

                G_curr_meta.train()
                G_curr_meta.load_state_dict(netG.state_dict())

                task_index = task_indices[outer_iteration]
                dataset = datasets[task_index]
                noise_amps = noise_amps_list[task_index]

                data_loader = DataLoader(dataset,
                                         shuffle=False,
                                         batch_size=1,
                                         num_workers=0)

                train(opt, data_loader, outer_iteration, noise_amps, generator_optimizer_tuple,
                      discriminator_optimizer_tuple=discriminator_optimizer_tuple, task_index=task_index)

                update_main_model(G_curr_meta, netG, optimizerG)
                if opt.vae_levels < opt.scale_idx + 1:
                    update_main_model(D_curr_meta, D_curr, optimizerD)

                # Update progress bar
                epoch_iterator.set_description('Scale [{}/{}], Outer Iteration [{}/{}]'.format(
                    opt.scale_idx + 1, opt.stop_scale + 1,
                    outer_iteration + 1, opt.outer_iterations,
                ))

                if outer_iteration and opt.validation_interval and outer_iteration % opt.validation_interval == 0:
                    if opt.vae_levels < opt.scale_idx + 1:
                        validate(G_curr_meta, datasets, generator_optimizer_tuple, netG, noise_amps_list,
                                 number_of_tasks, opt, outer_iteration, D_curr, D_curr_meta, discriminator_optimizer_tuple)
                    else:
                        validate(G_curr_meta, datasets, generator_optimizer_tuple, netG, noise_amps_list,
                                 number_of_tasks, opt, outer_iteration)

            epoch_iterator.close()

        # Save data
        opt.saver.save_checkpoint({'data': noise_amps_list}, 'Noise_Amps_list.pth')
        opt.saver.save_checkpoint({
            'scale': opt.scale_idx,
            'state_dict': netG.state_dict(),
            'optimizer': optimizerG.state_dict(),
            'noise_amps_list': noise_amps_list,
        }, 'netG.pth')
        if opt.vae_levels < opt.scale_idx + 1:
            opt.saver.save_checkpoint({
                'scale': opt.scale_idx,
                'state_dict': D_curr.state_dict(),
                'optimizer': optimizerD.state_dict(),
            }, 'netD_{}.pth'.format(opt.scale_idx))

        # Increase scale
        opt.scale_idx += 1


def validate(G_curr_meta, datasets, generator_optimizer_tuple, netG, noise_amps_list, number_of_tasks,
             opt, outer_iteration, D_curr=None, D_curr_meta=None, discriminator_optimizer_tuple=None):

    G_curr_meta.load_state_dict(netG.state_dict())

    if opt.vae_levels < opt.scale_idx + 1:
        D_curr_meta.load_state_dict(D_curr.state_dict())

    validation_task_index = number_of_tasks - 1
    dataset = datasets[validation_task_index]
    noise_amps = noise_amps_list[validation_task_index]
    data_loader = DataLoader(dataset,
                             shuffle=False,
                             batch_size=1,
                             num_workers=0)

    train(opt, data_loader, outer_iteration, noise_amps, generator_optimizer_tuple,
          discriminator_optimizer_tuple=discriminator_optimizer_tuple, validate=True)


def update_main_model(meta_model, model, optimizer):
    for p, meta_p in zip(model.parameters(), meta_model.parameters()):
        diff = p - meta_p.cpu()
        p.grad = diff
    optimizer.step()


if __name__ == '__main__':
    main()
