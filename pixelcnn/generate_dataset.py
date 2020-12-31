import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch

import utils
from datasets.image import MultipleImageDataset
from modules import networks_2d

DATASET_FORMAT_BY_SCALE = "data_for_scale_{scale_idx}.pt"


def parse_arguments():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--classifiers-folder', help='path to trained classifiers', required=True)
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    # networks hyper parameters:
    parser.add_argument('--nc-im', type=int, default=3, help='# channels')
    parser.add_argument('--nfc', type=int, default=64, help='model basic # channels')
    parser.add_argument('--ker-size', type=int, default=3, help='kernel size')
    parser.add_argument('--num-layer', type=int, default=5, help='number of layers')
    parser.add_argument('--stride', default=1, help='stride')
    parser.add_argument('--padd-size', type=int, default=1, help='net pad size')

    # Dataset
    parser.add_argument('--image-path', required=True, help="image path")
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--scale-factor', type=float, default=0.75, help='pyramid scale factor')
    parser.add_argument('--min-size', type=int, default=32, help='image minimal size at the coarser scale')
    parser.add_argument('--max-size', type=int, default=256, help='image minimal size at the coarser scale')
    parser.add_argument('--stop-scale-time', type=int, default=-1)

    opt = parser.parse_args()
    opt.hflip = False
    opt.scale_factor_init = opt.scale_factor

    utils.adjust_scales2image(opt.img_size, opt)

    return opt


def load_classifiers(opt, classifier_path_format: str, default_number_of_classes: int = 10) -> "List[networks_2d.WDiscriminator2DMulti]":
    """
    Load classifiers trained by HP-GAN-VAE
    """
    classifiers = []
    idx = 0
    any_file_found = False
    while True:
        try:
            clf_checkpoint = torch.load(classifier_path_format.format(idx), map_location=torch.device('cpu'))
            any_file_found = True
        except FileNotFoundError:
            if any_file_found:
                break
            else:
                raise
        number_of_classes = clf_checkpoint.get("number_of_classes", default_number_of_classes)
        classifier = networks_2d.WDiscriminator2DMulti(opt, number_of_classes)
        classifier.load_state_dict(clf_checkpoint["state_dict"])
        classifiers.append(classifier)

        idx += 1

    return classifiers


def get_image_and_class_map_per_scale_by_classifier(opt, dataset, img_idx, classifiers) -> "List[Tuple[torch.Tensor, torch.Tensor]]":
    """
    Given image (by idx) in a dataset, create its maps in all scales
    :param opt:
    :param dataset:
    :param img_idx:
    :param classifiers:
    :return: list of tuples s.t. each tuple is the image in the current scale and the corresponding map
    """
    res = []
    log_softmax = torch.nn.LogSoftmax(dim=1)
    for scale_idx in range(len(classifiers)):
        opt.scale_idx = scale_idx
        img_current_scale = dataset[img_idx][1]
        class_maps = classifiers[scale_idx](img_current_scale.unsqueeze(dim=0))
        class_map = log_softmax(class_maps).max(dim=1).indices.type(torch.FloatTensor)
        res.append((img_current_scale, class_map))

    return res


def generate(opt, classifiers) -> "List[List[Tuple[torch.Tensor, torch.Tensor]]]":
    """
    Generate for all images their maps in all scales
    :param opt:
    :param classifiers:
    :return: list of lists, s.t. every list is the output of get_image_and_class_map_per_scale_by_classifier
    """
    dataset = MultipleImageDataset(opt)
    number_of_images = len(os.listdir(opt.image_path))

    images_and_class_maps_per_scale = []

    with torch.no_grad():
        for idx in range(number_of_images):
            image_and_class_map_per_scale = get_image_and_class_map_per_scale_by_classifier(opt, dataset, idx, classifiers)
            images_and_class_maps_per_scale.append(image_and_class_map_per_scale)
    return images_and_class_maps_per_scale


def save_dataset(opt, images_and_class_maps_per_scale):
    """
    Save the generate images_and_class_maps_per_scale in *.pt files, for each scale
    """
    folder_name = os.path.basename(opt.image_path)
    p = Path(os.path.join("generated_dataset", folder_name))
    p.mkdir(exist_ok=True)

    data_per_scale = defaultdict(list)
    for img_idx, image_and_class_map_per_scale in enumerate(images_and_class_maps_per_scale):
        for scale_idx, (image, class_map) in enumerate(image_and_class_map_per_scale):
            data_per_scale[scale_idx].append((image, class_map))

    for scale_idx in data_per_scale:
        torch.save(data_per_scale[scale_idx], os.path.join(p, DATASET_FORMAT_BY_SCALE.format(scale_idx=scale_idx)))
        print(f"Saved data for scale {scale_idx}")


def main():
    opt = parse_arguments()

    classifier_path_format = os.path.join(opt.classifiers_folder, "classifier_{}.pth")
    classifiers = load_classifiers(opt, classifier_path_format)
    images_and_class_maps_per_scale = generate(opt, classifiers)
    save_dataset(opt, images_and_class_maps_per_scale)


if __name__ == '__main__':
    main()
