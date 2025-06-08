"""
Functions related to fashionMNIST dataset.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST


def load_fashion_mnist(
    batch_size: int,
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Loads the fashionMNIST dataset and splits trainval into train and validation splits.
    The dataset is resized to 3x224x224 to match the input shape of CV models.

    Args:
        batch_size (int): Number of samples per dataloader batch.
        train_ratio (float): Percentage of samples to use in train split.
        random_seed (int): Random seed used in splitting the dataset.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, val and test dataloaders.

    Raises:
        ValueError: When train_ratio is not in range 0.0 - 1.0.
    """

    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("Train ratio should be a value in range 0.0 - 1.0!")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1)),
        ]
    )

    dataset_root = "./data"

    trainval_dataset = FashionMNIST(
        root=dataset_root,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = FashionMNIST(
        root=dataset_root,
        train=False,
        download=True,
        transform=transform,
    )

    train_size = int(train_ratio * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size

    generator = torch.Generator().manual_seed(random_seed)

    train_dataset, val_dataset = random_split(
        trainval_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
