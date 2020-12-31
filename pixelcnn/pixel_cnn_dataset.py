from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import os

from pixelcnn.generate_dataset import DATASET_FORMAT_BY_SCALE


class ConditionedPixelCNNDataset(Dataset):
    def __init__(self, path: str, scale_idx: int, data_repetition: int):
        if not (os.path.exists(path) and os.path.isdir(path)):
            raise Exception(f"Invalid path {path}")

        self.image_and_map_tuples_list: "List[Tuple[torch.Tensor, torch.Tensor]]" = \
            torch.load(os.path.join(path, DATASET_FORMAT_BY_SCALE.format(scale_idx=scale_idx)))

        self.num_of_images = len(self.image_and_map_tuples_list)
        self.num_of_images_with_repetition = self.num_of_images * data_repetition

        assert self.num_of_images > 0

    def __getitem__(self, idx):
        return self.image_and_map_tuples_list[idx % self.num_of_images]

    def __len__(self):
        return self.num_of_images_with_repetition
