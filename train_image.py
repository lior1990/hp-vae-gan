import time

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
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from modules import networks_2d
from modules.utils import calc_gradient_penalty
from datasets.image import SingleImageDataset, MultipleImageDataset, PixelCNNSamplesDataset

clear = colorama.Style.RESET_ALL
blue = colorama.Fore.CYAN + colorama.Style.BRIGHT
green = colorama.Fore.GREEN + colorama.Style.BRIGHT
magenta = colorama.Fore.MAGENTA + colorama.Style.BRIGHT


def train_zero_scale_with_ref(opt, netG):
    # VQVAE
    parameter_list = [{"params": netG.vqvae_encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                       {"params": netG.vector_quantization.parameters(),
                        "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                       {"params": netG.decoder.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}]
    ref_parameter_list = [{"params": netG.vqvae_encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                       {"params": netG.vector_quantization.parameters(),
                        "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},]

    optimizerG = optim.Adam(parameter_list, lr=opt.lr_g, betas=(opt.beta1, 0.999))
    ref_optimizerG = optim.Adam(ref_parameter_list, lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # Parallel
    data_parallel = torch.cuda.device_count() > 1
    if opt.device == 'cuda' and data_parallel:
        G_curr = torch.nn.DataParallel(netG)
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
    ref_iterator = iter(ref_data_loader)

    for iteration in epoch_iterator:
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(opt.data_loader)
            data = next(iterator)

        real_zero = data.to(opt.device)

        ############################
        # (1) Update VAE network
        ###########################
        total_loss = 0

        generated, embedding_loss = G_curr(real_zero, opt.Noise_Amps, mode="rec")

        rec_vae_loss = opt.rec_loss(generated, real_zero)
        vqvae_loss = opt.rec_weight * rec_vae_loss + embedding_loss

        total_loss += vqvae_loss

        G_curr.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(G_curr.parameters(), opt.grad_clip)
        optimizerG.step()

        try:
            ref_data = next(ref_iterator)
        except StopIteration:
            ref_iterator = iter(ref_data_loader)
            ref_data = next(ref_iterator)

        ref_real_zero = ref_data.to(opt.device)

        ref_reconstruction, ref_embedding_loss = G_curr(ref_real_zero, opt.Noise_Amps, mode="rec")

        # todo: add cycle loss?
        # cycle_loss = opt.rec_loss_l1(G_curr.vqvae_encode(ref_reconstruction), ref_z_e)

        ref_reconstruction_loss = opt.rec_loss(ref_reconstruction, ref_real_zero)
        ref_total_loss = ref_embedding_loss + ref_reconstruction_loss

        G_curr.zero_grad()
        ref_total_loss.backward()
        ref_optimizerG.step()

        # Update progress bar
        epoch_iterator.set_description('Scale [{}/{}], Iteration [{}/{}]'.format(
            opt.scale_idx + 1, opt.stop_scale + 1,
            iteration + 1, opt.niter,
        ))

        if opt.visualize:
            # Tensorboard
            opt.summary.add_scalar('Video/Scale {}/noise_amp'.format(opt.scale_idx), opt.noise_amp, iteration)
            opt.summary.add_scalar('Video/Scale {}/embedding loss'.format(opt.scale_idx), embedding_loss.item(), iteration)
            opt.summary.add_scalar('Video/Scale {}/Rec VAE'.format(opt.scale_idx), rec_vae_loss.item(), iteration)
            opt.summary.add_scalar('Video/Scale {}/Ref embedding loss'.format(opt.scale_idx), ref_embedding_loss.item(), iteration)
            opt.summary.add_scalar('Video/Scale {}/Ref Rec VAE'.format(opt.scale_idx), ref_reconstruction_loss.item(), iteration)

            if iteration % opt.print_interval == 0:
                with torch.no_grad():
                    fake_var = []
                    for _ in range(3):
                        fake, _ = G_curr(real_zero, opt.Noise_Amps, mode="vq_rand")
                        fake_var.append(fake)
                    fake_var = torch.cat(fake_var, dim=0)

                opt.summary.visualize_image(opt, iteration, real_zero, 'Real')
                opt.summary.visualize_image(opt, iteration, generated, 'Generated')
                opt.summary.visualize_image(opt, iteration, fake_var, 'Fake var')

    epoch_iterator.close()

    # Save data
    opt.saver.save_checkpoint({'data': opt.Noise_Amps}, 'Noise_Amps.pth')
    opt.saver.save_checkpoint({
        'scale': opt.scale_idx,
        'state_dict': netG.state_dict(),
        'optimizer': optimizerG.state_dict(),
        'noise_amps': opt.Noise_Amps,
    }, 'netG.pth')

    return ref_real_zero.shape[-2], ref_real_zero.shape[-1]


def train_with_pixel_cnn(opt, netG, pixel_cnn_samples_data_loader):
    if opt.vae_levels < opt.scale_idx + 1:
        D_curr = getattr(networks_2d, opt.discriminator)(opt).to(opt.device)

        if opt.vae_levels < opt.scale_idx:
            if (opt.netG != '') and opt.resumed_idx != -1:
                D_curr.load_state_dict(
                    torch.load('{}/netD_{}.pth'.format(opt.resume_dir, opt.scale_idx - 1))['state_dict'])
            else:
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
            # VQVAE
            parameter_list += [{"params": netG.vqvae_encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.vector_quantization.parameters(),
                                "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
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
    data_parallel = torch.cuda.device_count() > 1
    if opt.device == 'cuda' and data_parallel:
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
    samples_iterator = iter(pixel_cnn_samples_data_loader)

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
                        opt.Noise_Amps.append(opt.noise_amp)
                    else:
                        opt.Noise_Amps.append(0)
                        z_reconstruction, _ = G_curr(real_zero, opt.Noise_Amps, mode="rec")

                        RMSE = torch.sqrt(F.mse_loss(real, z_reconstruction))
                        opt.noise_amp = opt.noise_amp_init * RMSE.item() / opt.batch_size
                        opt.Noise_Amps[-1] = opt.noise_amp

        ############################
        # (1) Update VAE network
        ###########################
        total_loss = 0

        generated, embedding_loss = G_curr(real_zero, opt.Noise_Amps, mode="rec")

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
            try:
                samples = next(samples_iterator)
            except StopIteration:
                samples_iterator = iter(pixel_cnn_samples_data_loader)
                samples = next(samples_iterator)

            samples = samples.squeeze(dim=1)
            samples.to(opt.device)

            # Train 3D Discriminator
            D_curr.zero_grad()
            output = D_curr(real)
            errD_real = -output.mean()

            # train with fake
            #################
            fake, _ = G_curr(real_zero, opt.Noise_Amps, samples=samples, mode="vq_rand")

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
                opt.summary.add_scalar('Video/Scale {}/embedding loss'.format(opt.scale_idx), embedding_loss.item(), iteration)
            else:
                opt.summary.add_scalar('Video/Scale {}/rec loss'.format(opt.scale_idx), rec_loss.item(), iteration)
            if opt.vae_levels < opt.scale_idx + 1:
                opt.summary.add_scalar('Video/Scale {}/errG'.format(opt.scale_idx), errG.item(), iteration)
                opt.summary.add_scalar('Video/Scale {}/errD_fake'.format(opt.scale_idx), errD_fake.item(), iteration)
                opt.summary.add_scalar('Video/Scale {}/errD_real'.format(opt.scale_idx), errD_real.item(), iteration)
            else:
                opt.summary.add_scalar('Video/Scale {}/Rec VAE'.format(opt.scale_idx), rec_vae_loss.item(), iteration)

            if iteration % opt.print_interval == 0:
                with torch.no_grad():
                    fake_var = []
                    for _ in range(3):
                        if opt.vae_levels < opt.scale_idx + 1:
                            fake = G_curr(real_zero, opt.Noise_Amps, samples=samples, mode="vq_rand")[0]
                        else:
                            fake = G_curr(real_zero, opt.Noise_Amps, mode="vq_rand")[0]
                        fake_var.append(fake)
                    fake_var = torch.cat(fake_var, dim=0)

                opt.summary.visualize_image(opt, iteration, real, 'Real')
                opt.summary.visualize_image(opt, iteration, generated, 'Generated')
                opt.summary.visualize_image(opt, iteration, fake_var, 'Fake var')

    epoch_iterator.close()

    # Save data
    opt.saver.save_checkpoint({'data': opt.Noise_Amps}, 'Noise_Amps.pth')
    opt.saver.save_checkpoint({
        'scale': opt.scale_idx,
        'state_dict': netG.state_dict(),
        'optimizer': optimizerG.state_dict(),
        'noise_amps': opt.Noise_Amps,
    }, 'netG.pth')
    if opt.vae_levels < opt.scale_idx + 1:
        opt.saver.save_checkpoint({
            'scale': opt.scale_idx,
            'state_dict': D_curr.module.state_dict() if data_parallel else D_curr.state_dict(),
            'optimizer': optimizerD.state_dict(),
        }, 'netD_{}.pth'.format(opt.scale_idx))


def train_pixel_cnn_model(opt, netG):
    from pixelcnn.generate_dataset_vqvae import generate
    from pixelcnn.train_pixel_cnn import train_epoch
    from pixelcnn.pixel_cnn_dataset import PixelCNNDataset
    from pixelcnn.pixel_cnn_modules import GatedPixelCNN
    ref_dataset = MultipleImageDataset(opt, load_ref_image=True, data_rep=1)
    ref_data_loader = DataLoader(ref_dataset,
                                 shuffle=True,
                                 drop_last=False,
                                 batch_size=opt.batch_size * 5,
                                 num_workers=0)
    encodings = generate(opt, netG, ref_data_loader)
    repetitions = 1000
    print(f"Going to train PixelCNN model over {len(encodings)} with {repetitions} repetitions")
    dataset = PixelCNNDataset(encodings=encodings, data_repetition=repetitions)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size*5, shuffle=False, num_workers=0)

    pixel_cnn_model = GatedPixelCNN(opt.n_embeddings, opt.nfc, opt.num_layer).to(opt.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pixel_cnn_model.parameters(), lr=opt.lr_g)

    train_epoch(opt, pixel_cnn_model, data_loader, criterion, optimizer, opt.summary, 0)

    return pixel_cnn_model


def create_pixel_cnn_samples_data_loader(pixel_cnn_model, opt, h, w, samples_to_generate=100):
    pixel_cnn_model.eval()

    with torch.no_grad():
        print(f"Generating {samples_to_generate} samples...")
        start_time = time.time()
        samples_batch = pixel_cnn_model.generate(shape=(h, w), batch_size=samples_to_generate)
        if h != w:
            # the model supports only NxN input, need to fix that
            samples_batch = samples_batch[:, :h, :w]
        print(f"Done generating samples in {time.time() - start_time} seconds")

    return samples_batch


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
    parser.add_argument('--pooling', action='store_true', default=False, help='use interpolation in encoder-decoder')

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
    parser.add_argument('--ref-image-path', required=True, help="image path")
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

    parser.set_defaults(hflip=False)
    opt = parser.parse_args()

    assert opt.vae_levels > 0
    assert opt.disc_loss_weight > 0

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

    pixel_cnn_samples_data_loader = None

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

        samples = torch.load(os.path.join(opt.resume_dir, 'pixelcnn-samples.pth'))["samples"]
        pixel_cnn_samples_data_loader = DataLoader(PixelCNNSamplesDataset(samples), num_workers=0,
                                                   batch_size=opt.batch_size)
    else:
        opt.resumed_idx = -1


    while opt.scale_idx < opt.stop_scale + 1:
        if opt.scale_idx > 0:
            netG.init_next_stage()
            netG.to(opt.device)
            train_with_pixel_cnn(opt, netG, pixel_cnn_samples_data_loader)
        else:
            h, w = train_zero_scale_with_ref(opt, netG)
            pixel_cnn = train_pixel_cnn_model(opt, netG)
            samples = create_pixel_cnn_samples_data_loader(pixel_cnn, opt, h, w)
            pixel_cnn_samples_data_loader = DataLoader(PixelCNNSamplesDataset(samples), num_workers=0, batch_size=opt.batch_size)
            opt.saver.save_checkpoint({
                'samples': samples,
            }, 'pixelcnn-samples.pth')
            opt.saver.save_checkpoint({
                'state_dict': pixel_cnn.state_dict(),
            }, 'pixelcnn.pth')
            del pixel_cnn
            print("first scale completed")

        # Increase scale
        opt.scale_idx += 1
        if opt.resumed_idx != -1:
            opt.resumed_idx = -1
