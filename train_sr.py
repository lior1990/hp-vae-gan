
import os
import numpy as np
from torch.utils.data.dataloader import DataLoader

from datasets.image import MultipleImageDataset
from utils import tools

import torch
import torch.optim as optim
import torch.nn.functional as F

from utils.cutmix import rand_bbox


def train_sr(opt, sr_generator):
    sr_optimizer = optim.Adam(sr_generator.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # Parallel
    is_parallel = torch.cuda.device_count() > 1
    if is_parallel:
        print("Working with data parallel")
        sr_generator = torch.nn.DataParallel(sr_generator)

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

    for iteration in epoch_iterator:
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(opt.data_loader)
            data = next(iterator)

        real, _ = data
        real = real.to(opt.device)

        for _ in range(opt.n_times_cutmix):
            lam = np.random.uniform()
            bbx1, bby1, bbx2, bby2 = rand_bbox(real.size(), lam)
            batch_index_permutation = torch.randperm(real.size()[0])

            real[:, :, bbx1:bbx2, bby1:bby2] = real[batch_index_permutation, :, bbx1:bbx2, bby1:bby2]

        cutmix_blurry_real = F.interpolate(real, size=(42, 62))

        sr_generator.zero_grad()
        sr_real = sr_generator(cutmix_blurry_real)
        sr_loss = opt.rec_loss(real, sr_real)
        sr_loss.backward()
        sr_optimizer.step()

        # Update progress bar
        epoch_iterator.set_description('Scale [{}/{}], Iteration [{}/{}]'.format(
            opt.scale_idx + 1, opt.stop_scale + 1,
            iteration + 1, opt.niter,
        ))

        if opt.visualize:
            # Tensorboard
            opt.summary.add_scalar('Video/Scale {}/rec loss'.format(opt.scale_idx), sr_loss.item(), iteration)

            if iteration % opt.print_interval == 0:
                # todo: add eval using ref_data_loader?
                opt.summary.visualize_image(opt, iteration, real, 'SR Target')
                opt.summary.visualize_image(opt, iteration, sr_real, 'SR Output')
                opt.summary.visualize_image(opt, iteration, cutmix_blurry_real, 'SR Input', dim=3)

    epoch_iterator.close()

    # Save data
    opt.saver.save_checkpoint({
        'scale': opt.scale_idx,
        'state_dict': sr_generator.state_dict(),
        'optimizer': sr_optimizer.state_dict(),
    }, 'SR.pth')


def eval_sr(opt, sr_generator, netG):
    import matplotlib.pyplot as plt

    save_dir = os.path.join(opt.saver.experiment_dir, f"generated_images_{opt.scale_idx}")
    os.makedirs(save_dir, exist_ok=True)

    original_image_path = opt.image_path
    original_rep = opt.data_rep
    original_hflip = opt.hflip
    opt.hflip = False  # disable hflip in eval
    opt.image_path = opt.eval_dataset
    opt.data_rep = 1
    dataset = MultipleImageDataset(opt)
    test_data_loader = DataLoader(dataset, batch_size=1, num_workers=0)

    fakes_folder = os.path.join(save_dir, "fakes")
    reals_folder = os.path.join(save_dir, "reals")
    os.makedirs(fakes_folder, exist_ok=True)
    os.makedirs(reals_folder, exist_ok=True)

    netG.eval()
    sr_generator.eval()
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
            fig, axes = plt.subplots(1, 2, figsize=(20, 5))  # todo: define figsize to be real images' size

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
            sr_output = sr_generator(rec_output)

            real_tensor_to_plot = tensor_to_plot(real)
            rec_tensor_to_plot = tensor_to_plot(sr_output)
            axes[0].imshow(real_tensor_to_plot)
            axes[1].imshow(rec_tensor_to_plot)
            fig.savefig(os.path.join(save_dir, f"{idx}.png"))  # save the figure to file
            plt.close(fig)
            plt.imsave(os.path.join(reals_folder, f"real_{idx}.png"), real_tensor_to_plot)
            plt.imsave(os.path.join(fakes_folder, f"reconstruction_{idx}.png"), rec_tensor_to_plot)

    sr_generator.train()
    netG.train()

    opt.image_path = original_image_path
    opt.data_rep = original_rep
    opt.hflip = original_hflip
