"""Source-data sampling for EATA's Fisher regularizer."""
import random
import math
from pathlib import Path
import torch
import torchvision
from torchvision import transforms
from core.model.imagenet_subsets import IMAGENET_D109_MAPPING


def get_source_loader(dataset_name, root_dir, adaptation, batch_size, train_split=True,
                      ckpt_path=None, num_samples=None, percentage=1.0, workers=4, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.ImageNet(
        root=str(Path(root_dir) / "ImageNet"), split="train" if train_split else "val", transform=transform,
    )
    if dataset_name == "imagenet_d109":
        dataset.samples = [(path, IMAGENET_D109_MAPPING[label]) for path, label in dataset.samples
                           if IMAGENET_D109_MAPPING[label] != -1]
    if num_samples or percentage < 1:
        count = min(num_samples, len(dataset.samples)) if num_samples else int(math.ceil(len(dataset.samples) * percentage))
        dataset.samples = random.sample(dataset.samples, count)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                        num_workers=workers, drop_last=False)
    return dataset, loader
