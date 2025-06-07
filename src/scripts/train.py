"""
Model training experiment script.
"""

# pylint: disable=too-many-locals, too-many-statements

import argparse
import json
import logging
from logging import Logger
from pathlib import Path
from typing import Tuple, Type

import torch
from torch import nn
from torch.optim import SGD, AdamW, Optimizer
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.models import resnet50, vit_b_16
from tqdm import tqdm

from ..config.data_model import (
    ExperimentConfig,
    ModelName,
    OptimizerName,
)
from ..datasets.fashion_mnist import load_fashion_mnist_trainval
from ..optimizers.lion import Lion
from ..utils.early_stopping import EarlyStopping


def select_optimizer_class(optimizer_name: OptimizerName) -> Type[Optimizer]:
    """
    Given the name of the gradient optimizer return the class of the optimizer.

    Args:
        optimizer_name (OptimizerName): Name optimizer from the OptimizerName enumeration.

    Returns:
        Type[Optimizer]: Class of the optimizer.

    Raises:
        ValueError: When the name of the optimizer is invalid.
    """
    match optimizer_name:
        case OptimizerName.SGD:
            return SGD
        case OptimizerName.ADAM:
            return AdamW
        case OptimizerName.LION:
            return Lion

    raise ValueError(f"Invalid optimizer name {optimizer_name.value}!")


def create_model(model_name: ModelName) -> nn.Module:
    """
    Create a model of given architecture.

    Args:
        model_name (ModelName): Name of the desired model from ModelName enumeration.

    Returns:
        nn.Module: Initialized model of a given architecture.

    Raises:
        ValueError: When the name of the model is invalid.
    """
    match config.model_name:
        case ModelName.RES_NET_50:
            return resnet50()
        case ModelName.VIT_B_16:
            return vit_b_16()

    raise ValueError(f"Invalid model name {model_name.value}!")


def get_available_device() -> torch.device:
    """
    Detect the best performing available device.

    Returns:
        torch.device: The best available torch device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")  # apple silicon
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_model(
    config: ExperimentConfig,
    logger: Logger,
    *,
    random_seed: int = 42,
    device: torch.device = "cpu",
) -> nn.Module:
    """
    TODO docstring
    TODO get loaders as arguments
    TODO get model as argument
    """
    # Set random seed for torch devices.
    # TODO remove? or make random seed optional
    # torch.manual_seed(random_seed)

    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(random_seed)

    # if torch.backends.mps.is_available():
    #     torch.backends.mps.mps_set_random_seed(random_seed)

    # Initialize the model.
    model = create_model(config.model_name)
    model.to(device)

    # Initialize the optimizer.
    optimizer_class = select_optimizer_class(config.optimizer_name)
    optimizer = optimizer_class(model.parameters(), lr=config.learning_rate)

    # load dataset
    train_loader, val_loader = load_fashion_mnist_trainval(config.batch_size)

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

    device = get_available_device()

    logger.info("Using device %s", device)

    trained_model = train_model(config, logger, device=device)
