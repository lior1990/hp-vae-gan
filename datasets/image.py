import random
from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset
import imageio
import kornia as K
from torchvision.transforms import transforms

import utils
import cv2
import logging
import os


class ImageDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, opt):
        self.zero_scale_frames = None
        self.frames = None
        self.opt = opt
        self.random_grayscale = transforms.RandomGrayscale(0.2)
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        self.to_pil = transforms.ToPILImage()
        self.normalize = transforms.Normalize((0.5,), (0.5,))
        self.to_tensor = transforms.ToTensor()
        self.hflip = transforms.RandomHorizontalFlip()

    def _transform_image(self, image, scale_index: int, hflip: bool):
        images_zero_scale = self._generate_image(image, scale_index)
        images_zero_scale = K.image_to_tensor(images_zero_scale)
        images_zero_scale = images_zero_scale / 255
        images_zero_scale_transformed = self._get_transformed_images(images_zero_scale, hflip)
        return images_zero_scale_transformed

    def _get_transformed_images(self, images, hflip):
        images_transformed = self.to_pil(images)

        if hflip:
            images_transformed = self.hflip(images_transformed)

        if self.opt.hflip and (self.opt.vqvae_levels >= self.opt.scale_idx + 1):
            if random.random() > 0.5:
                images_transformed = self.color_jitter(images_transformed)
            images_transformed = self.random_grayscale(images_transformed)

        # Normalize
        images_transformed = self.to_tensor(images_transformed)
        images_transformed = self.normalize(images_transformed)

        return images_transformed

    def _generate_image(self, image_full_scale, scale_idx):
        base_size = utils.get_scales_by_index(scale_idx, self.opt.scale_factor, self.opt.stop_scale, self.opt.img_size)
        scaled_size = [int(base_size * self.opt.ar), base_size]
        if scale_idx == 0:
            scaled_size[0] -= scaled_size[0] % 4
        self.opt.scaled_size = scaled_size
        img = cv2.resize(image_full_scale, tuple(scaled_size[::-1]))
        return img

    def __getitem__(self, idx):
        image_full_scale = self._get_image(idx)

        # Horizontal flip (Until Kornia will handle videos
        hflip = random.random() < 0.5 if self.opt.hflip else False

        image_transformed = self._transform_image(image_full_scale, self.opt.scale_idx, hflip)

        input_imgs = []
        for scale_idx in range(min(self.opt.scale_idx+1, self.opt.vqvae_levels)):
            input_imgs.append(self._transform_image(image_full_scale, scale_idx, hflip))

        # todo: there is an overlap here in the first scales... need to handle it
        return image_transformed, input_imgs

    @abstractmethod
    def _get_image(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass


class SingleImageDataset(ImageDataset):
    def __init__(self, opt):
        super(SingleImageDataset, self).__init__(opt)

        self.image_path = opt.image_path
        if not os.path.exists(opt.image_path):
            logging.error("invalid path")
            exit(0)

        # Get original frame size and aspect-ratio
        self.image_full_scale = imageio.imread(self.image_path)[:, :, :3]
        self.org_size = [self.image_full_scale.shape[0], self.image_full_scale.shape[1]]
        h, w = self.image_full_scale.shape[:2]
        opt.ar = h / w  # H2W

    def __len__(self):
        return self.opt.data_rep

    def _get_image(self, idx):
        return self.image_full_scale


class MultipleImageDataset(ImageDataset):
    def __init__(self, opt):
        super(MultipleImageDataset, self).__init__(opt)

        if not (os.path.exists(opt.image_path) and os.path.isdir(opt.image_path)):
            logging.error("invalid path")
            exit(0)

        self.images = []
        for img_path in os.listdir(opt.image_path):
            image_full_scale = imageio.imread(os.path.join(opt.image_path, img_path))[:, :, :3]
            self.images.append(image_full_scale)

        self.num_of_images = len(self.images)

        assert self.num_of_images > 0

        # Get original frame size and aspect-ratio
        # assumption: all images are of the same size
        h, w = self.images[0].shape[:2]
        opt.ar = h / w  # H2W

    def __len__(self):
        return self.opt.data_rep * self.num_of_images

    def _get_image(self, idx):
        return self.images[idx % self.num_of_images]
