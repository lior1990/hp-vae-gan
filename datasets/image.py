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

from utils.images import SR_FIXED_SIZES


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

    def _transform_image(self, image, scale_index: int, hflip: bool):
        images_zero_scale = self._generate_image(image, scale_index)
        images_zero_scale = self._get_transformed_images(images_zero_scale, hflip)
        return images_zero_scale

    @staticmethod
    def _get_transformed_images(images, hflip):
        images = K.image_to_tensor(images)
        images = images / 255

        images_transformed = images

        if hflip:
            images_transformed = K.hflip(images_transformed)

        # Normalize
        images_transformed = K.normalize(images_transformed, 0.5, 0.5)

        return images_transformed

    def _generate_image(self, image_full_scale, scale_idx):
        if self.opt.fixed_scales:
            scaled_size = SR_FIXED_SIZES[scale_idx]
        else:
            base_size = utils.get_scales_by_index(scale_idx, self.opt.scale_factor, self.opt.stop_scale, self.opt.img_size, self.opt.fixed_scales)
            scaled_size = [int(base_size * self.opt.ar), base_size]
        # for pooling support
        # if scale_idx == 0:
        #     scaled_size[0] -= scaled_size[0] % 4
        self.opt.scaled_size = scaled_size
        img = cv2.resize(image_full_scale, tuple(scaled_size[::-1]))
        return img

    def __getitem__(self, idx):
        image_full_scale = self._get_image(idx)

        # Horizontal flip (Until Kornia will handle videos
        hflip = random.random() < 0.5 if self.opt.hflip else False

        image = self._transform_image(image_full_scale, self.opt.scale_idx, hflip)

        # Extract o-level index
        if self.opt.scale_idx > 0:
            images_zero_scale = self._transform_image(image_full_scale, 0, hflip)

            return image, images_zero_scale

        return image

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
    def __init__(self, opt, load_ref_image=False):
        super(MultipleImageDataset, self).__init__(opt)

        image_path = opt.ref_image_path if load_ref_image else opt.image_path

        if not (os.path.exists(image_path) and os.path.isdir(image_path)):
            logging.error("invalid path")
            exit(0)

        self.images = []
        for img_path in os.listdir(image_path):
            image_full_scale = imageio.imread(os.path.join(image_path, img_path))[:, :, :3]
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


class CachedMultipleImageDataset(MultipleImageDataset):
    def __init__(self, opt, load_ref_image=False):
        super(CachedMultipleImageDataset, self).__init__(opt, load_ref_image=load_ref_image)

        self._zero_scale_cache = {}
        self._scale_cache = {}

    def init_cache(self):
        cache = self._zero_scale_cache if self.opt.scale_idx == 0 else self._scale_cache

        for img_idx, img in enumerate(self.images):
            cache[img_idx] = self._generate_image(img, self.opt.scale_idx)

    def __getitem__(self, idx):
        img_zero_scale = self._zero_scale_cache[idx % self.num_of_images]

        hflip = random.random() < 0.5 if self.opt.hflip else False

        img_zero_scale = self._get_transformed_images(img_zero_scale, hflip)

        if self.opt.scale_idx > 0:
            img = self._scale_cache[idx % self.num_of_images]
            img = self._get_transformed_images(img, hflip)
            return img, img_zero_scale

        return img_zero_scale


class AllScalesMultipleImageDataset(MultipleImageDataset):
    def __getitem__(self, idx):
        image_full_scale = self._get_image(idx)

        return [self._transform_image(image_full_scale, i, False) for i in range(self.opt.scale_idx)]
