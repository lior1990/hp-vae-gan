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


log_softmax = torch.nn.LogSoftmax(dim=1)


def calc_classifier_loss(output, class_indices_map, opt):
    loss_fn = torch.nn.CrossEntropyLoss()

    # convert height and width to be part of the "batch" for the cross entropy loss input expected shape
    target = class_indices_map.squeeze(dim=1).type(torch.LongTensor).to(opt.device)

    # input shape: (batch, channel), target shape: (batch)
    loss = loss_fn(output, target)

    class_map = log_softmax(output).max(dim=1).indices
    true_preds = (target == class_map).sum()
    total = float(class_map.nelement())

    acc = true_preds / total
    return loss, acc


def train(opt):
    number_of_images = len(os.listdir(opt.image_path))

    if opt.classifier == "regular":
        map_classifier = networks_2d.WDiscriminator2DMulti(opt, number_of_images).to(opt.device)
    else:
        map_classifier = networks_2d.MultiClassClassifier(number_of_images).to(opt.device)

    if opt.scale_idx > 0:
        map_classifier.load_state_dict(
            torch.load('{}/classifier_{}.pth'.format(opt.saver.experiment_dir, opt.scale_idx - 1))['state_dict'])

    optimizer_map_classifier = optim.Adam(map_classifier.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

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

        map_classifier.zero_grad()
        classifier_output = map_classifier(real)
        class_indices_map = torch.full((idx.shape[0], 1, real.shape[2], real.shape[3]), 1, device=opt.device) * idx.view(-1, 1, 1, 1)
        classifier_loss, acc = calc_classifier_loss(classifier_output, class_indices_map, opt)
        classifier_loss.backward()
        optimizer_map_classifier.step()

        classifier_loss_value = classifier_loss.item()
        classifier_acc_value = acc.item()
        # Update progress bar
        epoch_iterator.set_description('Scale [{}/{}], Iteration [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(
            opt.scale_idx + 1, opt.stop_scale + 1,
            iteration + 1, opt.niter,
            classifier_loss_value,
            classifier_acc_value,
        ))

        if opt.visualize:
            opt.summary.add_scalar(f'Video/Scale {opt.scale_idx}/Classifier loss', classifier_loss_value, iteration)
            opt.summary.add_scalar(f'Video/Scale {opt.scale_idx}/Classifier Accuracy', classifier_acc_value, iteration)

    epoch_iterator.close()

    # Save data
    opt.saver.save_checkpoint({
        'scale': opt.scale_idx,
        'state_dict': map_classifier.state_dict(),
        'optimizer': optimizer_map_classifier.state_dict(),
    }, 'classifier_{}.pth'.format(opt.scale_idx))


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
    parser.add_argument('--classifier', type=str, default='regular')

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

    # Initial parameters
    opt.scale_idx = 0
    opt.nfc_prev = 0

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

    opt.resumed_idx = -1

    while opt.scale_idx < opt.stop_scale + 1:

        train(opt)

        # Increase scale
        opt.scale_idx += 1
