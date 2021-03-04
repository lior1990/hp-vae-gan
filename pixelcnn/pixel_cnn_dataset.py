from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import os


class PixelCNNDataset(Dataset):
    def __init__(self, path: str, data_repetition: int):
        if not (os.path.exists(path)):
            raise Exception(f"Invalid path {path}")

        self.encodings: "List[torch.Tensor]" = torch.load(path)

        self.num_of_images = len(self.encodings)
        self.num_of_images_with_repetition = self.num_of_images * data_repetition

        assert self.num_of_images > 0

    def __getitem__(self, idx):
        return self.encodings[idx % self.num_of_images]

    def __len__(self):
        return self.num_of_images_with_repetition
