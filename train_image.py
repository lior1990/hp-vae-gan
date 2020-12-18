import matplotlib

matplotlib.use("Agg")

import argparse
import utils
import random
import os

import neptune

from utils import logger, tools
import logging
import colorama

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from modules import networks_2d
from modules.losses import kl_criterion
from modules.utils import calc_gradient_penalty, pad_with_cls
from datasets.image import SingleImageDataset, MultipleImageDataset

clear = colorama.Style.RESET_ALL
blue = colorama.Fore.CYAN + colorama.Style.BRIGHT
green = colorama.Fore.GREEN + colorama.Style.BRIGHT
magenta = colorama.Fore.MAGENTA + colorama.Style.BRIGHT

use_neptune = True
try:
    neptune.init(project_qualified_name='lior.tau/ff-singan')
except Exception as e:
    print(e)
    use_neptune = False


def calc_classifier_loss(output, class_indices_map, opt):
    loss_fn = torch.nn.CrossEntropyLoss()

    # convert height and width to be part of the "batch" for the cross entropy loss input expected shape
    reshaped_output = output.permute(0, 2, 3, 1)  # b, class, h, w -> b, h, w, class
    shape = reshaped_output.shape
    reshaped_output = output.reshape(shape[0]*shape[1]*shape[2], -1)  # b*h*w, class

    target = class_indices_map.reshape(-1).type(torch.LongTensor).to(opt.device)

    # input shape: (batch, channel), target shape: (batch)
    loss = loss_fn(reshaped_output, target)

    return loss


def train(opt, netG, class_maps_per_scale):
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

    number_of_images = len(os.listdir(opt.image_path))
    map_classifier = networks_2d.WDiscriminator2DMulti(opt, number_of_images).to(opt.device)
    if opt.scale_idx > 0:
        map_classifier.load_state_dict(
            torch.load('{}/classifier_{}.pth'.format(opt.saver.experiment_dir, opt.scale_idx - 1))['state_dict'])

    optimizer_map_classifier = optim.Adam(map_classifier.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    log_softmax = torch.nn.LogSoftmax(dim=1)

    parameter_list = []
    # Generator Adversary

    if not opt.train_all:
        if opt.vae_levels < opt.scale_idx + 1:
            train_depth = min(opt.train_depth, len(netG.body) - opt.vae_levels + 1)
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-train_depth:])]
            # todo: add decoder at all scales?
            # parameter_list += [{"params": netG.encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}]
        else:
            # VAE
            parameter_list += [{"params": netG.encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.decoder_head.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.decoder_base.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.decoder_tail.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}]
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
            idx, real, real_zero = data
        else:
            idx, real = data
            real_zero = real

        idx = idx.to(opt.device)
        idx = idx.unsqueeze(dim=1)
        real = real.to(opt.device)
        real_zero = real_zero.to(opt.device)

        map_classifier.zero_grad()
        classifier_output = map_classifier(real)
        class_indices_map = torch.full((idx.shape[0], 1, real.shape[2], real.shape[3]), 1, device=opt.device) * idx.view(-1, 1, 1, 1)
        classifier_loss = calc_classifier_loss(classifier_output, class_indices_map, opt)
        classifier_loss.backward()
        optimizer_map_classifier.step()

        initial_size = utils.get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size)
        initial_size = [7, 11]
        opt.Z_init_size = [opt.batch_size, opt.latent_dim, *initial_size]

        noise_init = utils.generate_noise(size=opt.Z_init_size, device=opt.device)

        ############################
        # calculate noise_amp
        ###########################
        if iteration == 0:
            full_class_indices_real = torch.stack(
                [torch.full((1, real.shape[2], real.shape[3]), i) for i in range(number_of_images)]).to(
                opt.device)
            class_maps_per_scale.append(full_class_indices_real)

            class_maps_per_scale_for_batch = [class_maps[idx.view(-1)] for class_maps in class_maps_per_scale]

            if opt.const_amp:
                opt.Noise_Amps.append(1)
            else:
                with torch.no_grad():
                    if opt.scale_idx == 0:
                        opt.noise_amp = 1
                        opt.Noise_Amps.append(opt.noise_amp)
                    else:
                        opt.Noise_Amps.append(0)
                        z_reconstruction, _, _ = G_curr(real_zero, class_maps_per_scale_for_batch, opt.Noise_Amps, mode="rec")

                        RMSE = torch.sqrt(F.mse_loss(real, z_reconstruction))
                        opt.noise_amp = opt.noise_amp_init * RMSE.item() / opt.batch_size
                        opt.Noise_Amps[-1] = opt.noise_amp

        ############################
        # (1) Update VAE network
        ###########################
        total_loss = 0

        generated, generated_vae, _ = G_curr(real_zero, class_maps_per_scale_for_batch, opt.Noise_Amps, mode="rec")

        generated_classifier_output = map_classifier(generated)
        err_classifier = calc_classifier_loss(generated_classifier_output, class_indices_map, opt)

        total_loss = err_classifier

        if opt.vae_levels >= opt.scale_idx + 1:
            rec_vae_loss = opt.rec_loss(generated, real) + opt.rec_loss(generated_vae, real_zero)
            vae_loss = opt.rec_weight * rec_vae_loss

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
            fake, _, _ = G_curr(real_zero, class_maps_per_scale_for_batch, opt.Noise_Amps, noise_init=noise_init, mode="rand")

            # Train 3D Discriminator
            output = D_curr(fake.detach())
            errD_fake = output.mean()

            gradient_penalty = calc_gradient_penalty(D_curr, real, fake, opt.lambda_grad, opt)
            errD_total = errD_real + errD_fake + gradient_penalty
            errD_total.backward()
            optimizerD.step()

            ############################
            # (3) Update G network: maximize D(G(z))
            ###########################
            errG_total = 0
            rec_loss = opt.rec_loss(generated, real) + opt.rec_loss(generated_vae, real_zero)

            errG_total += opt.rec_weight * rec_loss

            # Train with 3D Discriminator
            output = D_curr(fake)
            errG = -output.mean() * opt.disc_loss_weight
            errG_total += errG

            fake_classifier_output = map_classifier(fake)
            errG_classifier = calc_classifier_loss(fake_classifier_output, class_indices_map, opt)
            errD_total += errG_classifier

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
            if opt.vae_levels < opt.scale_idx + 1:
                opt.summary.add_scalar('Video/Scale {}/rec loss'.format(opt.scale_idx), rec_loss.item(), iteration)
                opt.summary.add_scalar('Video/Scale {}/errG'.format(opt.scale_idx), errG.item(), iteration)
                opt.summary.add_scalar('Video/Scale {}/errD_fake'.format(opt.scale_idx), errD_fake.item(), iteration)
                opt.summary.add_scalar('Video/Scale {}/errD_real'.format(opt.scale_idx), errD_real.item(), iteration)
            else:
                opt.summary.add_scalar('Video/Scale {}/Rec VAE'.format(opt.scale_idx), rec_vae_loss.item(), iteration)
            opt.summary.add_scalar(f'Video/Scale {opt.scale_idx}/Classifier loss', classifier_loss.item(), iteration)
            opt.summary.add_scalar(f'Video/Scale {opt.scale_idx}/G classifier error', err_classifier.item(), iteration)

            if iteration % opt.print_interval == 0:
                with torch.no_grad():
                    # todo: plot images with classifier's map instead of real maps

                    fake_var = []
                    fake_vae_var = []
                    for _ in range(3):
                        noise_init = utils.generate_noise(ref=noise_init)
                        fake, fake_vae, _ = G_curr(real_zero, class_maps_per_scale_for_batch, opt.Noise_Amps, noise_init=noise_init, mode="rand")
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

    # Save data
    opt.saver.save_checkpoint({'data': opt.Noise_Amps}, 'Noise_Amps.pth')
    opt.saver.save_checkpoint({
        'scale': opt.scale_idx,
        'state_dict': netG.state_dict(),
        'optimizer': optimizerG.state_dict(),
        'noise_amps': opt.Noise_Amps,
    }, 'netG.pth')
    opt.saver.save_checkpoint({
        'scale': opt.scale_idx,
        'state_dict': map_classifier.state_dict(),
        'optimizer': optimizer_map_classifier.state_dict(),
    }, 'classifier_{}.pth'.format(opt.scale_idx))
    if opt.vae_levels < opt.scale_idx + 1:
        opt.saver.save_checkpoint({
            'scale': opt.scale_idx,
            'state_dict': D_curr.module.state_dict() if opt.device == 'cuda' else D_curr.state_dict(),
            'optimizer': optimizerD.state_dict(),
        }, 'netD_{}.pth'.format(opt.scale_idx))


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
    parser.add_argument('--diversity-loss-weight', type=float, default=1., help='diversity loss weight')
    parser.add_argument('--diversity-start-scale', type=int, default=4, help='diversity start scale')
    parser.add_argument('--diversity-total-scales', type=int, default=2, help='Number of scales to use diversity loss')
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
    parser.add_argument('--print-interval', type=int, default=100, help='print interva')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize using tensorboard')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda')
    parser.add_argument('--tag', type=str, default='', help='neptune ai tag')

    parser.set_defaults(hflip=False)
    opt = parser.parse_args()

    assert opt.vae_levels > 0
    assert opt.disc_loss_weight > 0

    if opt.data_rep < opt.batch_size:
        opt.data_rep = opt.batch_size

    # Define Saver
    opt.saver = utils.ImageSaver(opt)

    # Define Tensorboard Summary
    if use_neptune and opt.tag:
        neptune_exp = neptune.create_experiment(name=opt.checkname, params=opt.__dict__, tags=[opt.tag]).__enter__()
        opt.summary = utils.TensorboardSummary(opt.saver.experiment_dir, neptune_exp=neptune_exp)
    else:
        use_neptune = False
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

    if opt.stop_scale_time == -1:
        opt.stop_scale_time = opt.stop_scale

    opt.dataset = dataset
    opt.data_loader = data_loader

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
        # NoiseAmp
        opt.Noise_Amps = torch.load(os.path.join(opt.resume_dir, 'Noise_Amps.pth'))['data']
    else:
        opt.resumed_idx = -1

    class_maps_per_scale = []

    while opt.scale_idx < opt.stop_scale + 1:
        if (opt.scale_idx > 0) and (opt.resumed_idx != opt.scale_idx):
            netG.init_next_stage()

        train(opt, netG, class_maps_per_scale)

        # Increase scale
        opt.scale_idx += 1

    if use_neptune:
        neptune_exp.__exit__(None, None, None)
