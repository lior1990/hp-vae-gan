import os
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
import time

from pixelcnn.pixel_cnn_dataset import ConditionedPixelCNNDataset
from pixelcnn.pixel_cnn_modules import GatedPixelCNN

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-directory", type=str, help='trained model directory', required=True)
    parser.add_argument("--generated-dataset-path",  type=str, help='generated dataset path (by generate_dataset.py)', required=True)
    parser.add_argument("--scale-limit", type=int, default=0, help="inclusive limit for scale generation")
    parser.add_argument("--samples-to-generate", type=int, default=5, help="number of samples to generate for each image")

    # model initialization parameters
    parser.add_argument("--n-layers", type=int, default=15)
    parser.add_argument("--number-of-classes", type=int, default=10,
                        help="the number of classes that the classifier of HP-VAE-GAN was trained with."
                             "This is equivalent to the number of embeddings in VQ-VAE")
    parser.add_argument("--hidden-dim", type=int, default=64)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def generate_samples(args, pixel_cnn_model, scale_idx):
    dataset = ConditionedPixelCNNDataset(args.generated_dataset_path, scale_idx, 1)
    test_batch_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

    img, _ = next(iter(data_loader))
    img = img.to(args.device)

    img_map = pixel_cnn_model.generate(img, shape=(img.shape[-2], img.shape[-1]), batch_size=test_batch_size)
    return img_map, img


def generate_single_scale(args, scale_idx: int):
    pixel_cnn_model = GatedPixelCNN(args.number_of_classes, args.hidden_dim, args.n_layers)
    pixel_cnn_model.load_state_dict(torch.load(os.path.join(args.model_directory, f"pixel_cnn_scale_{scale_idx}.pt"),
                                               map_location=args.device).state_dict())
    pixel_cnn_model.to(args.device)
    pixel_cnn_model.eval()

    samples_and_reals = []

    with torch.no_grad():
        for _ in range(args.samples_to_generate):
            start_time = time.time()
            print("Generating samples...")
            samples_and_reals.append(generate_samples(args, pixel_cnn_model, scale_idx))
            print(f"Done generating samples in {time.time() - start_time} seconds")

    return samples_and_reals


def main():
    args = parse_arguments()

    samples_path = Path(os.path.join(args.model_directory, "samples"))
    samples_path.mkdir(exist_ok=True)

    for scale_idx in range(args.scale_limit+1):
        print(f"Starting to generate samples for scale {scale_idx}")
        samples_and_reals = generate_single_scale(args, scale_idx)
        torch.save(samples_and_reals, os.path.join(samples_path, f"samples_and_reals_{scale_idx}.pt"))
        print(f"Done scale {scale_idx}")


if __name__ == '__main__':
    main()
