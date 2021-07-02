import matplotlib.pyplot as plt
import argparse
import logging
import os
import random
import torch
from torch.utils.data.dataloader import DataLoader

import utils
from datasets.image import SingleImageDataset, MultipleImageDataset
from modules import networks_2d


def parse_opt():
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
    parser.add_argument('--decoder-normalization-method', type=str, default="bn", help='decoder normalization method')
    parser.add_argument('--g-normalization-method', type=str, default="bn", help='generator normalization method')
    parser.add_argument('--generator', type=str, default='GeneratorHPVAEGAN', help='generator model')
    parser.add_argument('--discriminator', type=str, default='WDiscriminator2D', help='discriminator model')

    # pyramid parameters:
    parser.add_argument('--scale-factor', type=float, default=0.75, help='pyramid scale factor')
    parser.add_argument('--noise_amp', type=float, default=0.1, help='addative noise cont weight')
    parser.add_argument('--min-size', type=int, default=32, help='image minimal size at the coarser scale')
    parser.add_argument('--max-size', type=int, default=256, help='image minimal size at the coarser scale')
    parser.add_argument('--interpolation-method', type=str, default="bilinear", help="upscale interpolation method")

    # optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=2, help='number of iterations to train per scale')
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
    parser.add_argument('--fixed-scales', action='store_true', default=False, help='use hard-coded scales')

    # Dataset
    parser.add_argument('--image-path', required=True, help="image path")
    parser.add_argument('--hflip', action='store_true', default=False, help='horizontal flip')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--stop-scale-time', type=int, default=-1)
    parser.add_argument('--data-rep', type=int, default=1, help='data repetition')

    # main arguments
    parser.add_argument('--checkname', type=str, default='DEBUG', help='check name')
    parser.add_argument('--mode', default='train', help='task to be done')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--print-interval', type=int, default=100, help='print interva')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize using tensorboard')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda')

    # vq vae arguments
    parser.add_argument('--n_embeddings', type=int, default=100, help='number of embeddings (keys of the vqvae dict)')
    parser.add_argument('--embedding_dim', type=int, default=64, help='embedding dimension (values of the vqvae dict)')
    parser.add_argument('--vqvae_beta', type=float, default=.25)
    parser.add_argument('--positional_encoding_weight', type=int, default=1)
    parser.add_argument('--pooling', action='store_true', default=False, help='pooling in encoder&decoder')
    parser.add_argument('--vqvae-version', type=str, default="vqvae2")
    parser.add_argument('--fake-mode', type=str, default="rec", help='fake mode (rec/rec_noise)')
    parser.add_argument('--spade-scales', type=str, default="6")

    parser.set_defaults(hflip=False)
    opt = parser.parse_args()
    return opt


keys = ["nfc", "embedding_dim", "n_embeddings", "vae_levels", "enc_blocks", "positional_encoding_weight", "min_size",
        "num_layer", "encoder_normalization_method", "decoder_normalization_method", "g_normalization_method",
        "padding_mode", "interpolation_method", "fixed_scales", "vqvae_version", "fake_mode", "spade_scales"]
results = {}


def load_params(net_g_path):
    folder = os.path.dirname(net_g_path)
    f = open(os.path.join(folder, "logbook.txt"))
    for line in f.readlines():
        for key in keys:
            search_key = f"{key}: "
            if search_key in line:
                idx = line.index(search_key)
                val = line.strip()[idx+len(search_key):]
                try:
                    if val == "False":
                        results[key] = False
                    elif "," in val or val == '[]':
                        results[key] = list(eval(val))
                    else:
                        results[key] = int(val)
                except ValueError:
                    results[key] = val
                break
    return results


def init(opt):
    if opt.data_rep < opt.batch_size:
        opt.data_rep = opt.batch_size

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

    # this is required because inside the __init__ some opt values are being initialized (e.g. opt.ar)
    if os.path.isdir(opt.image_path):
        dataset = MultipleImageDataset(opt)
    elif os.path.isfile(opt.image_path):
        dataset = SingleImageDataset(opt)
    else:
        raise NotImplementedError

    opt.n_images = dataset.num_of_images
    initial_size = utils.get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size, opt.fixed_scales)
    initial_size = [int(initial_size * opt.ar), initial_size]
    opt.Z_init_size = [opt.batch_size, opt.latent_dim, *initial_size]
    noise_init = utils.generate_noise(size=opt.Z_init_size, device=opt.device)

    dataset_size = len(dataset)
    # Current networks
    assert hasattr(networks_2d, opt.generator)
    netG = getattr(networks_2d, opt.generator)(opt).to(opt.device)

    assert opt.netG != ''
    if not os.path.isfile(opt.netG):
        raise RuntimeError("=> no <G> checkpoint found at '{}'".format(opt.netG))

    checkpoint = torch.load(opt.netG, map_location=torch.device('cpu'))

    opt.scale_idx = checkpoint['scale']
    opt.resumed_idx = checkpoint['scale']
    opt.resume_dir = os.sep.join(opt.netG.split(os.sep)[:-1])
    for _ in range(opt.scale_idx):
        netG.init_next_stage()

    netG.load_state_dict(checkpoint["state_dict"])
    # NoiseAmp
    opt.Noise_Amps = torch.load(os.path.join(opt.resume_dir, 'Noise_Amps.pth'))['data']
    netG.to(opt.device)
    return netG


def eval_netG(image_path, save_dir, opt, netG):
    opt.image_path = image_path
    dataset = MultipleImageDataset(opt)
    test_data_loader = DataLoader(dataset, batch_size=1, num_workers=0)

    netG.eval()
    with torch.no_grad():
        def norm(t):
            def norm_ip(img, min, max):
                img.clamp_(min=min, max=max)
                img.add_(-min).div_(max - min + 1e-5)

            norm_ip(t, float(t.min()), float(t.max()))

        def plot_tensor(t, ax):
            norm(t)
            ax.imshow(t.squeeze().cpu().permute((1, 2, 0)))

        for idx, (img, img_zero) in enumerate(test_data_loader):
            fig, axes = plt.subplots(1, 2, figsize=(20, 5))

            for plot_idx in range(2):
                axes[plot_idx].set_xticks([])
                axes[plot_idx].set_yticks([])

            img_zero[0] = img_zero[0].to(opt.device)
            rec_output = netG(img_zero[0], opt.Noise_Amps, mode="rec")[0]

            plot_tensor(img[0], axes[0])
            plot_tensor(rec_output, axes[1])
            fig.savefig(os.path.join(save_dir, f"{idx}.png"))  # save the figure to file
            plt.close(fig)


def main():
    base_folder = r"run/5imgs-2-pos-enc-aug-hflip-t-all-scales-final"
    dataset_for_eval = "data/imgs/misc/"
    experiments = os.listdir(base_folder)
    for exp in experiments:
        print(f"Working on {exp}")
        opt = parse_opt()
        exp_folder = os.path.join(base_folder, exp)
        samples_folder = os.path.join(exp_folder, "generated_images")
        os.makedirs(samples_folder, exist_ok=True)
        opt.netG = os.path.join(exp_folder, "netG.pth")
        params = load_params(opt.netG)
        for k, v in params.items():
            setattr(opt, k, v)
        netG = init(opt)
        print(f"Starting eval on {exp}. Dataset: {dataset_for_eval}. Results will be save on {samples_folder}")
        eval_netG(dataset_for_eval, samples_folder, opt, netG)
        print("Done")


if __name__ == '__main__':
    main()
