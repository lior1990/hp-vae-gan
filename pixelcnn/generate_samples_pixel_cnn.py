import os
from pathlib import Path

import torch
import time

from pixelcnn.pixel_cnn_dataset import PixelCNNDataset
from pixelcnn.pixel_cnn_modules import GatedPixelCNN

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-directory", type=str, help='trained model directory', required=True)
    parser.add_argument("--generated-dataset-file",  type=str, help='generated dataset file (by generate_dataset_vqvae.py)', required=True)
    parser.add_argument("--samples-to-generate", type=int, default=5, help="number of samples to generate for each image")

    # model initialization parameters
    parser.add_argument("--n-layers", type=int, default=15)
    parser.add_argument("--n_embeddings", type=int, default=10, help="The number of embeddings in VQ-VAE")
    parser.add_argument("--hidden-dim", type=int, default=64)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def generate(args):
    pixel_cnn_model = GatedPixelCNN(args.n_embeddings, args.hidden_dim, args.n_layers)
    pixel_cnn_model.load_state_dict(torch.load(os.path.join(args.model_directory, f"pixel_cnn.pt"),
                                               map_location=args.device).state_dict())
    pixel_cnn_model.to(args.device)
    pixel_cnn_model.eval()

    samples = []

    dataset = PixelCNNDataset(args.generated_dataset_file, 1)

    with torch.no_grad():
        for img_idx in range(len(dataset)):
            encodings = dataset[img_idx]

            print(f"Generating samples for img {img_idx}...")
            start_time = time.time()
            samples_batch = pixel_cnn_model.generate(shape=(encodings.shape[-1], encodings.shape[-1]),
                                                     batch_size=args.samples_to_generate)
            if encodings.shape[-1] > encodings.shape[-2]:
                # the model supports only NxN input, need to fix that
                samples_batch = samples_batch[:, :encodings.shape[-2], :encodings.shape[-1]]
            samples_list = torch.split(samples_batch, 1)
            print(f"Done generating samples in {time.time() - start_time} seconds for img {img_idx}")
            samples.extend(samples_list)

    return samples


def main():
    args = parse_arguments()

    samples_path = Path(os.path.join(args.model_directory, "samples"))
    samples_path.mkdir(exist_ok=True)

    print(f"Starting to generate samples")
    samples = generate(args)
    torch.save(samples, os.path.join(samples_path, f"samples.pt"))
    print(f"Done")


if __name__ == '__main__':
    main()
