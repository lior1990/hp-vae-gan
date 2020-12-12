import matplotlib

from modules.networks_2d import Encode2DAE

matplotlib.use("Agg")

import argparse
import utils
import random
import os

import neptune
import random
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

use_neptune = True
try:
    neptune.init(project_qualified_name='lior.tau/ff-singan')
except Exception as e:
    print(e)
    use_neptune = False


def train(opt, netGs, encoder, reals, reals_zero, class_indices_real_zero, class_indices_real):
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

    optimizerGs = []
    for netG in netGs:
        parameter_list = get_parameters_list(opt, netG)
        optimizerGs.append(optim.Adam(parameter_list, lr=opt.lr_g, betas=(opt.beta1, 0.999)))

    optimizer_encoder = optim.Adam(encoder.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # Parallel
    if opt.device == 'cuda':
        G_currs = [torch.nn.DataParallel(netG) for netG in netGs]
        if opt.vae_levels < opt.scale_idx + 1:
            D_curr = torch.nn.DataParallel(D_curr)
    else:
        G_currs = netGs

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

    indices_per_decoder = [[j for j in range(len(G_currs)) if j != i] for i in range(len(G_currs))]

    for iteration in epoch_iterator:
        encoder.zero_grad()
        z_ae = encoder(torch.cat([reals_zero, class_indices_real_zero], dim=1))

        for i, G_curr in enumerate(G_currs):
            G_curr.to(opt.device)
            z_ae_curr = z_ae[indices_per_decoder[i], :, :, :]
            class_indices_real_zero_curr = class_indices_real_zero[indices_per_decoder[i], :, :, :]
            class_indices_real_curr = class_indices_real[indices_per_decoder[i], :, :, :]
            real_zero = reals_zero[indices_per_decoder[i], :, :, :]
            real = reals[indices_per_decoder[i], :, :, :]
            loo_z_ae = z_ae[[i], :, :, :]
            loo_class = class_indices_real_zero[[i], :, :, :]

            initial_size = utils.get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size)
            initial_size = [7, 11]
            opt.Z_init_size = [z_ae_curr.shape[0], opt.latent_dim, *initial_size]

            noise_init = utils.generate_noise(size=opt.Z_init_size, device=opt.device)

            ############################
            # calculate noise_amp
            ###########################
            if iteration == 0:
                if True:  # todo: handle this later
                #if opt.const_amp:
                    opt.Noise_Amps.append(1)
                else:
                    with torch.no_grad():
                        if opt.scale_idx == 0:
                            opt.noise_amp = 1
                            opt.Noise_Amps.append(opt.noise_amp)
                        else:
                            opt.Noise_Amps.append(0)
                            z_reconstruction, _, _ = G_curr(real_zero, opt.Noise_Amps, mode="rec")

                            RMSE = torch.sqrt(F.mse_loss(real, z_reconstruction))
                            opt.noise_amp = opt.noise_amp_init * RMSE.item() / opt.batch_size
                            opt.Noise_Amps[-1] = opt.noise_amp

            ############################
            # (1) Update VAE network
            ###########################
            total_loss = 0

            generated, generated_vae, _ = G_curr(z_ae_curr, class_indices_real_zero_curr, opt.Noise_Amps, mode="rec")

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
                output = D_curr(torch.cat([real, class_indices_real_curr], dim=1))
                errD_real = -output.mean()

                # train with fake
                #################
                fake, _, _ = G_curr(z_ae_curr + noise_init, class_indices_real_zero_curr, opt.Noise_Amps, mode="rand")

                # Train 3D Discriminator
                output = D_curr(torch.cat([fake.detach(), class_indices_real_curr], dim=1))
                errD_fake = output.mean()

                gradient_penalty = calc_gradient_penalty(D_curr, real, fake, opt.lambda_grad, opt.device, class_indices_real_curr)
                errD_total = errD_real + errD_fake + gradient_penalty
                errD_total /= len(netGs)
                errD_total.backward()
                optimizerD.step()

                ############################
                # (3) Update G network: maximize D(G(z))
                ###########################
                errG_total = 0
                rec_loss = opt.rec_loss(generated, real) + opt.rec_loss(generated_vae, real_zero)

                errG_total += opt.rec_weight * rec_loss

                # Train with 3D Discriminator
                output = D_curr(torch.cat([fake, class_indices_real_curr], dim=1))
                errG = -output.mean() * opt.disc_loss_weight
                errG_total += errG

                total_loss += errG_total

            G_curr.zero_grad()
            total_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(G_curr.parameters(), opt.grad_clip)
            optimizerGs[i].step()

            if opt.visualize:
                # Tensorboard
                opt.summary.add_scalar(f'Video/Scale {opt.scale_idx}_{i}/noise_amp', opt.noise_amp, iteration)
                if opt.vae_levels < opt.scale_idx + 1:
                    opt.summary.add_scalar(f'Video/Scale {opt.scale_idx}_{i}/rec loss', rec_loss.item(), iteration)
                    opt.summary.add_scalar(f'Video/Scale {opt.scale_idx}_{i}/errG', errG.item(), iteration)
                    opt.summary.add_scalar(f'Video/Scale {opt.scale_idx}_{i}/errD_fake', errD_fake.item(), iteration)
                    opt.summary.add_scalar(f'Video/Scale {opt.scale_idx}_{i}/errD_real', errD_real.item(), iteration)
                else:
                    opt.summary.add_scalar(f'Video/Scale {opt.scale_idx}_{i}/Rec VAE', rec_vae_loss.item(), iteration)

                with torch.no_grad():
                    # todo: need to change class_indices_curr to exclude its own class and use others
                    loo_rec = G_curr(loo_z_ae, loo_class, opt.Noise_Amps, mode="rec")[0]
                    loo_real = reals[i].unsqueeze(dim=0)
                    loo_loss = opt.rec_loss(loo_rec, loo_real)
                    opt.summary.add_scalar(f'Video/Scale {opt.scale_idx}_{i}/LOO Rec', loo_loss.item(), iteration)

                if iteration % opt.print_interval == 0:
                    with torch.no_grad():
                        rand_indices_to_plot = [random.randint(0, z_ae_curr.shape[0]-1) for _ in range(3)]
                        noise_init = utils.generate_noise(ref=noise_init)
                        rand_batch = z_ae_curr[rand_indices_to_plot] + noise_init[rand_indices_to_plot]
                        rand_cls = class_indices_real_zero_curr[rand_indices_to_plot]
                        fake_var, fake_vae_var, _ = G_curr(rand_batch, rand_cls, opt.Noise_Amps, mode="rand")

                        loo_rec_vs_real = torch.cat([loo_rec, loo_real])

                    opt.summary.visualize_image(opt, iteration, loo_rec_vs_real, f'LOO {i}')
                    opt.summary.visualize_image(opt, iteration, real[rand_indices_to_plot], f'Real {i}')
                    opt.summary.visualize_image(opt, iteration, generated[rand_indices_to_plot], f'Generated {i}')
                    opt.summary.visualize_image(opt, iteration, generated_vae[rand_indices_to_plot], f'Generated VAE {i}')
                    opt.summary.visualize_image(opt, iteration, fake_var, f'Fake var {i}')
                    opt.summary.visualize_image(opt, iteration, fake_vae_var, f'Fake VAE var {i}')


            del total_loss
            if opt.vae_levels < opt.scale_idx + 1:
                del rec_loss
                del errG
                del errD_fake
                del errD_real
            else:
                del rec_vae_loss

            G_curr.to("cpu")

        optimizer_encoder.step()

        # Update progress bar
        epoch_iterator.set_description('Scale [{}/{}], Iteration [{}/{}]'.format(
            opt.scale_idx + 1, opt.stop_scale + 1,
            iteration + 1, opt.niter,
        ))

    epoch_iterator.close()

    # Save data
    opt.saver.save_checkpoint({'data': opt.Noise_Amps}, 'Noise_Amps.pth')
    opt.saver.save_checkpoint({
        'scale': opt.scale_idx,
        'encoder': encoder.state_dict(),
        'state_dict': {i: netG.state_dict() for i, netG in enumerate(netGs)},
        'optimizer': {i: optimizerG.state_dict() for i, optimizerG in enumerate(optimizerGs)},
        'noise_amps': opt.Noise_Amps,
    }, 'netG.pth')
    if opt.vae_levels < opt.scale_idx + 1:
        opt.saver.save_checkpoint({
            'scale': opt.scale_idx,
            'state_dict': D_curr.module.state_dict() if opt.device == 'cuda' else D_curr.state_dict(),
            'optimizer': optimizerD.state_dict(),
        }, 'netD_{}.pth'.format(opt.scale_idx))
        return D_curr


def get_parameters_list(opt, netG):
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
                # {"params": netG.encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                {"params": netG.decoder_head.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                {"params": netG.decoder_base.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                {"params": netG.decoder_tail.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}]
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-opt.train_depth:])]
    else:
        raise NotImplementedError
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
    return parameter_list



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
    assert os.path.isdir(opt.image_path)
    imgs = os.listdir(opt.image_path)

    if opt.stop_scale_time == -1:
        opt.stop_scale_time = opt.stop_scale

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

    full_dataset = MultipleImageDataset(opt)

    # Current networks
    assert hasattr(networks_2d, opt.generator)
    netGs = [getattr(networks_2d, opt.generator)(opt).to(opt.device) for _ in range(len(full_dataset))]

    if opt.netG != '':
        raise NotImplementedError
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

    encoder = Encode2DAE(opt, out_dim=opt.latent_dim, num_blocks=opt.enc_blocks)
    encoder = encoder.to(opt.device)

    while opt.scale_idx < opt.stop_scale + 1:
        if (opt.scale_idx > 0) and (opt.resumed_idx != opt.scale_idx):
            for netG in netGs:
                netG.init_next_stage()

        if opt.scale_idx > 0:
            # dataset return value: idx, real, real_zero
            reals_zero = torch.stack([full_dataset[i][2] for i in range(len(full_dataset))]).to(opt.device)
            reals = torch.stack([full_dataset[i][1] for i in range(len(full_dataset))]).to(opt.device)
        else:
            # dataset return value: idx, real_zero
            reals_zero = torch.stack([full_dataset[i][1] for i in range(len(full_dataset))]).to(opt.device)
            reals = reals_zero

        class_indices_real_zero = torch.stack([torch.full((1, reals_zero.shape[2], reals_zero.shape[3]), full_dataset[i][0]) for i in range(10)]).to(opt.device)
        class_indices_real = torch.stack(
            [torch.full((1, reals.shape[2], reals.shape[3]), full_dataset[i][0]) for i in range(10)]).to(
            opt.device)

        train(opt, netGs, encoder, reals, reals_zero, class_indices_real_zero, class_indices_real)
        # Increase scale
        opt.scale_idx += 1

    if use_neptune:
        neptune_exp.__exit__(None, None, None)

"""
N decoders 
N different batches of size N-1


full_batch = [...]

z = encoder(full_batch)

for i, decoder in enumerate(decoders):
    decoder_batch = z - z[i]
    decoder(decoder_batch)
    loss = ...
    loss.backward(retain_graph?) -- need it partially, only for encoder 
    decoder_optimizer.step()
    del loss  # maybe this will do the trick (https://github.com/pytorch/pytorch/issues/31185)

encoder_optimizer.step()


"""