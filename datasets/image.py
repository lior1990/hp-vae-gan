import random
from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset
import imageio
import kornia as K
import utils
import cv2
import logging
import os


class ImageDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, opt, transforms=None):
        self.zero_scale_frames = None
        self.frames = None
        self.transforms = transforms
        self.opt = opt

    def _transform_image(self, image, scale_index: int, hflip: bool):
        images_zero_scale = self._generate_image(image, scale_index)
        images_zero_scale = K.image_to_tensor(images_zero_scale)
        images_zero_scale = images_zero_scale / 255
        images_zero_scale_transformed = self._get_transformed_images(images_zero_scale, hflip)
        return images_zero_scale_transformed

    @staticmethod
    def _get_transformed_images(images, hflip):

        images_transformed = images

        if hflip:
            images_transformed = K.hflip(images_transformed)

        # Normalize
        images_transformed = K.normalize(images_transformed, 0.5, 0.5)

        return images_transformed

    def _generate_image(self, image_full_scale, scale_idx):
        base_size = utils.get_scales_by_index(scale_idx, self.opt.scale_factor, self.opt.stop_scale, self.opt.img_size)
        scaled_size = [int(base_size * self.opt.ar), base_size]
        self.opt.scaled_size = scaled_size
        img = cv2.resize(image_full_scale, tuple(scaled_size[::-1]))
        return img

    def __getitem__(self, idx):
        idx, image_full_scale = self._get_image(idx)

        # Horizontal flip (Until Kornia will handle videos
        hflip = random.random() < 0.5 if self.opt.hflip else False

        images_transformed = self._transform_image(image_full_scale, self.opt.scale_idx, hflip)

        # Extract o-level index
        if self.opt.scale_idx > 0:
            images_zero_scale_transformed = self._transform_image(image_full_scale, 0, hflip)

            return [idx, images_transformed, images_zero_scale_transformed]

        return idx, images_transformed

    @abstractmethod
    def _get_image(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass


class SingleImageDataset(ImageDataset):
    def __init__(self, opt, transforms=None):
        super(SingleImageDataset, self).__init__(opt, transforms=transforms)

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
    def __init__(self, opt, transforms=None):
        super(MultipleImageDataset, self).__init__(opt, transforms=transforms)

        if not (os.path.exists(opt.image_path) and os.path.isdir(opt.image_path)):
            logging.error("invalid path")
            exit(0)

        self.images = []
        for idx, img_path in enumerate(os.listdir(opt.image_path), start=1):
            image_full_scale = imageio.imread(os.path.join(opt.image_path, img_path))[:, :, :3]
            self.images.append((idx, image_full_scale))

        self.num_of_images = len(self.images)

        assert self.num_of_images > 0

        # Get original frame size and aspect-ratio
        # assumption: all images are of the same size
        h, w = self.images[0][1].shape[:2]
        opt.ar = h / w  # H2W

    def __len__(self):
        return self.opt.data_rep * self.num_of_images

    def _get_image(self, idx):
        return self.images[idx % self.num_of_images]
