"""
Model training experiment script.
"""

# pylint: disable=too-many-locals, too-many-statements

import argparse
import json
import logging
from logging import Logger
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.datasets import FakeData, FashionMNIST
from torchvision.datasets.imagenet import ImageNet
from torchvision.models import resnet50, vit_b_16
from tqdm import tqdm

from ..config.data_model import (
    ExperimentConfig,
    ModelName,
    OptimizerName,
)
from ..optimizers.lion import Lion
from ..utils.early_stopping import EarlyStopping


def train_model(
    config: ExperimentConfig,
    logger: Logger,
    *,
    random_seed: int = 42,
    device: torch.device = "cpu",
) -> nn.Module:
    """
    Train the model according to the configuration.

    Args:
        config (ExperimentConfig): Configuration of the experiment.
        logger (Logger): Logger to log information to.
        random_seed (int): The random seed to use for all stochastic processes.
        device ()

    Returns:
        Module: Trained pyTorch model.
    """
    # Set random seed for torch devices.
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    if torch.backends.mps.is_available():
        torch.backends.mps.mps_set_random_seed(random_seed)

    # Initialize the model.
    match config.model_name:
        case ModelName.RES_NET_50:
            model = resnet50()
        case ModelName.VIT_B_16:
            model = vit_b_16()
        case _:
            raise ValueError(f"Invalid model name {config.model_name.value}!")

    model.to(device)

    # Initialize the optimizer.
    match config.optimizer_name:
        case OptimizerName.SGD:
            optimizer_class = SGD
        case OptimizerName.ADAM:
            optimizer_class = AdamW
        case OptimizerName.LION:
            optimizer_class = Lion
        case _:
            raise ValueError(f"Invalid optimizer name {config.optimizer_name.value}!")

    optimizer = optimizer_class(model.parameters(), lr=config.learning_rate)

    # Get the dataset
    # dataset = ImageNet()  # TODO get Imagenet dataset
    dataset = FakeData(
        size=10000,
        image_size=(3, 224, 224),
        num_classes=1000,
        transform=transforms.ToTensor(),
    )

    # Define the transformation
    transform = transforms.ToTensor()

    # Load the dataset
    train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)


    # Split dataset into train and test splits.
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    print(train_dataset)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Loss function and early
    loss_function = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(**config.early_stopping.model_dump())

    for epoch in range(config.epochs):
        # Training step
        model.train()

        train_loss = 0
        train_correct_samples = 0
        train_num_samples = 0

        for x_batch, y_batch in tqdm(train_loader, desc="Training...", unit="batch"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_predicted = model(x_batch)
            loss = loss_function(y_predicted, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)

            predicted_labels = torch.max(y_predicted, 1)[1]
            train_correct_samples += (predicted_labels == y_batch).sum().item()
            train_num_samples += y_batch.size(0)

        # Compute train loss and accuracy
        train_avg_loss = train_loss / train_num_samples
        train_accuracy = train_correct_samples / train_num_samples

        # Validation step
        with torch.no_grad():
            model.eval()

            loss_value = 0
            num_correct_samples = 0
            num_all_samples = 0

            for x_batch, y_batch in tqdm(
                val_loader, desc="Validating...", unit="batch"
            ):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                y_predicted = model(x_batch)

                loss_value += loss_function(y_predicted, y_batch).item() * x_batch.size(
                    0
                )

                predicted_labels = torch.max(y_predicted, 1)[1]
                num_correct_samples += (predicted_labels == y_batch).sum().item()
                num_all_samples += y_batch.size(0)

            val_avg_loss = loss_value / num_all_samples
            val_accuracy = num_correct_samples / num_all_samples

        # Log epoch metrics.
        logger.info(
            f"Epoch {epoch+1}/{config.epochs}: "
            f"train loss: {train_avg_loss:.4f}, "
            f"train accuracy: {train_accuracy:.4f}, "
            f"val loss: {val_avg_loss:.4f}, "
            f"val accuracy: {val_accuracy:.4f}"
        )

        # Check for early stopping.
        early_stopping(val_avg_loss, model)

        if early_stopping.stop():
            logger.info(
                f"Early stopping in epoch {epoch+1} due to lack of improvement."
            )
            model.load_state_dict(early_stopping.best_model_state)
            break

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=Path,
        help="Path to config JSON file.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file_handle:
        config = ExperimentConfig(**json.load(file_handle))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    if torch.backends.mps.is_available():
        device = torch.device("mps")  # apple silicon
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info("Using device %s", device)

    trained_model = train_model(config, logger, device=device)
