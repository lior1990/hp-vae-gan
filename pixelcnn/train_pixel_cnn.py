import math
import os
from pathlib import Path
from typing import NamedTuple, Tuple

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import time

from tqdm import tqdm

from pixelcnn.pixel_cnn_dataset import ConditionedPixelCNNDataset
from pixelcnn.pixel_cnn_modules import GatedPixelCNN

import argparse

from utils import TensorboardSummary, tools


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name", type=str, help='generated dataset path (by generate_dataset.py)', required=True)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gen-samples", action="store_true")

    parser.add_argument("--generated-dataset-path",  type=str, help='generated dataset path (by generate_dataset.py)', required=True)
    parser.add_argument("--data-repetition", type=int, default=1, help="number of times to repeat every item in the DS in each epoch")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-layers", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--number-of-classes", type=int, default=10,
                        help="the number of classes that the classifier of HP-VAE-GAN was trained with."
                             "This is equivalent to the number of embeddings in VQ-VAE")
    parser.add_argument("--hidden-dim", type=int, default=64)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def train_epoch(args, pixel_cnn_model, data_loader, criterion, optimizer, summary, epoch, scale_idx):
    train_loss = []

    total_iterations = math.ceil(len(data_loader.dataset) / args.batch_size)
    progress_iterator = tqdm(iterable=enumerate(data_loader), total=total_iterations)

    for batch_idx, (img, img_map) in progress_iterator:
        img_map = img_map.squeeze(dim=1).type(torch.LongTensor).to(args.device)
        img = img.to(args.device)
        logits = pixel_cnn_model(img_map, img)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        # todo: set cross-entropy loss for images?
        loss = criterion(
            logits.view(-1, args.number_of_classes),
            img_map.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        summary.add_scalar(f"scale_{scale_idx}__epoch__{epoch}__loss", loss.item(), batch_idx)

        progress_iterator.set_description(f'Loss: {loss.item()}')


def generate_samples(args, pixel_cnn_model, scale_idx, summary, epoch):
    dataset = ConditionedPixelCNNDataset(args.generated_dataset_path, scale_idx, 1)
    test_batch_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False, num_workers=args.num_workers)

    opt = NamedTuple("dummy_opt", [("scale_idx", int)])(scale_idx)

    img, _ = next(iter(data_loader))
    img = img.to(args.device)

    if epoch == 0:
        # save real images only once
        summary.visualize_image(opt, epoch, img, "real image")

    img_map = pixel_cnn_model.generate(img, shape=(img.shape[-2], img.shape[-1]), batch_size=test_batch_size)
    summary.visualize_image(opt, epoch, img_map.unsqueeze(dim=1)/255, f"generated samples img", normalize=False)
    return img_map


def train_single_scale(args, scale_idx: int, summary) -> "Tuple[GatedPixelCNN, torch.Tensor]":
    dataset = ConditionedPixelCNNDataset(args.generated_dataset_path, scale_idx, args.data_repetition)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    pixel_cnn_model = GatedPixelCNN(args.number_of_classes, args.hidden_dim, args.n_layers).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pixel_cnn_model.parameters(), lr=args.learning_rate)

    samples = None

    for epoch in range(0, args.epochs):
        print("\nEpoch {}:".format(epoch))
        train_epoch(args, pixel_cnn_model, data_loader, criterion, optimizer, summary, epoch, scale_idx)

        if args.gen_samples:
            with torch.no_grad():
                start_time = time.time()
                print("Generating samples...")
                samples = generate_samples(args, pixel_cnn_model, scale_idx, summary, epoch)
                print(f"Done generating samples in {time.time() - start_time} seconds")

    return pixel_cnn_model, samples


def main():
    args = parse_arguments()
    results_path = Path(os.path.join("results", args.exp_name))
    results_path.mkdir(exist_ok=True, parents=True)
    results_path = str(results_path.absolute())
    summary = TensorboardSummary(results_path)
    number_of_scales = len(os.listdir(args.generated_dataset_path))

    for scale_idx in range(number_of_scales):
        print(f"Starting to train scale {scale_idx}")
        pixel_cnn_trained_model, latest_samples = train_single_scale(args, scale_idx, summary)
        torch.save(pixel_cnn_trained_model, os.path.join(results_path, f"pixel_cnn_scale_{scale_idx}.pt"))
        torch.save(latest_samples, os.path.join(results_path, f"samples_scale_{scale_idx}.pt"))
        del pixel_cnn_trained_model
        print(f"Done training scale {scale_idx}")


if __name__ == '__main__':
    main()
