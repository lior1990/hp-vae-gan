import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import math
import os
from pathlib import Path
from typing import NamedTuple, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import time

from tqdm import tqdm

from pixelcnn.pixel_cnn_dataset import PixelCNNDataset
from pixelcnn.pixel_cnn_modules import GatedPixelCNN
import torch.nn.functional as F

import argparse

from utils import TensorboardSummary, tools


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name", type=str, help='generated dataset path (by generate_dataset_vqvae.py)', required=True)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gen-samples", action="store_true")

    parser.add_argument("--generated-dataset-file",  type=str, help='generated dataset file (by generate_dataset_vqvae.py)', required=True)
    parser.add_argument("--data-repetition", type=int, default=1, help="number of times to repeat every item in the DS in each epoch")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-layers", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n_embeddings", type=int, default=10, help="The number of embeddings in VQ-VAE")
    parser.add_argument("--hidden-dim", type=int, default=64)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def train_epoch(args, pixel_cnn_model, data_loader, criterion, optimizer, summary, epoch):
    train_loss = []

    total_iterations = math.ceil(len(data_loader.dataset) / args.batch_size)
    progress_iterator = tqdm(iterable=enumerate(data_loader), total=total_iterations)

    for batch_idx, encoding in progress_iterator:
        img_map = encoding.type(torch.LongTensor).to(args.device)
        if img_map.shape[-1] > img_map.shape[-2]:
            # the model supports only NxN input, need to fix that
            img_map = F.pad(img_map, pad=(0, 0, 0, img_map.shape[-1] - img_map.shape[-2]))

        logits = pixel_cnn_model(img_map)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        # todo: set cross-entropy loss for images?
        loss = criterion(
            logits.view(-1, args.n_embeddings),
            img_map.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        summary.add_scalar(f"epoch__{epoch}__loss", loss.item(), batch_idx)

        progress_iterator.set_description(f'Loss: {loss.item()}')


def generate_samples(args, pixel_cnn_model, summary, epoch):
    dataset = PixelCNNDataset(args.generated_dataset_file, 1)
    test_batch_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False, num_workers=args.num_workers)

    encodings = next(iter(data_loader))

    if epoch == 0:
        # save real images only once
        visualize_maps(encodings, summary, "real encodings")

    generated_img_map = pixel_cnn_model.generate(shape=(encodings.shape[-1], encodings.shape[-1]), batch_size=test_batch_size)
    if encodings.shape[-1] > encodings.shape[-2]:
        # the model supports only NxN input, need to fix that
        generated_img_map = generated_img_map[:, :encodings.shape[-2], :encodings.shape[-1]]

    visualize_maps(generated_img_map, summary, "generated samples img", epoch)
    return generated_img_map


def visualize_maps(img_map: "torch.LongTensor", summary, name: str, idx: "Optional[int]" = None):
    number_of_plots = img_map.shape[0]
    figure, axes = plt.subplots(1, number_of_plots)

    for plot_idx in range(number_of_plots):
        axes[plot_idx].set_xticks([])
        axes[plot_idx].set_yticks([])
        axes[plot_idx].imshow(img_map[plot_idx].cpu())
    plt.close("all")
    summary.writer.add_figure(name, figure, global_step=idx)


def train_pixel_cnn(args, summary) -> "Tuple[GatedPixelCNN, torch.Tensor]":
    dataset = PixelCNNDataset(args.generated_dataset_file, args.data_repetition)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    pixel_cnn_model = GatedPixelCNN(args.n_embeddings, args.hidden_dim, args.n_layers).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pixel_cnn_model.parameters(), lr=args.learning_rate)

    samples = None

    for epoch in range(0, args.epochs):
        print("\nEpoch {}:".format(epoch))
        train_epoch(args, pixel_cnn_model, data_loader, criterion, optimizer, summary, epoch)

        if args.gen_samples:
            with torch.no_grad():
                start_time = time.time()
                print("Generating samples...")
                samples = generate_samples(args, pixel_cnn_model, summary, epoch)
                print(f"Done generating samples in {time.time() - start_time} seconds")

    return pixel_cnn_model, samples


def main():
    args = parse_arguments()
    results_path = Path(os.path.join("results", args.exp_name))
    results_path.mkdir(exist_ok=True, parents=True)
    results_path = str(results_path.absolute())
    summary = TensorboardSummary(results_path)

    print(f"Starting to train scale")
    pixel_cnn_trained_model, latest_samples = train_pixel_cnn(args, summary)
    torch.save(pixel_cnn_trained_model, os.path.join(results_path, f"pixel_cnn.pt"))
    if args.gen_samples:
        torch.save(latest_samples, os.path.join(results_path, f"samples.pt"))
    print(f"Done training scale")


if __name__ == '__main__':
    main()
