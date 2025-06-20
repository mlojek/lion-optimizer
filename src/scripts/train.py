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
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config.data_model import ExperimentConfig
from ..datasets.fashion_mnist import load_fashion_mnist
from ..models.create_model import create_model
from ..optimizers.create_optimizer import create_optimizer
from ..utils.device_utils import get_available_device
from ..utils.early_stopping import EarlyStopping


def train_model(
    config: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: Logger,
    *,
    random_seed: int = None,
    device: torch.device = "cpu",
) -> nn.Module:
    # TODO add docstring
    # Set random seed if specified
    if random_seed:
        torch.manual_seed(random_seed)

    # Initialize the model.
    model = create_model(config.model_name)
    model.to(device)

    # Initialize the optimizer.
    optimizer = create_optimizer(model, config)

    # Loss function and early stopping
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

        # validation loss
        val_avg_loss, val_accuracy = evaluate_model(
            model, loss_function, val_loader, device
        )

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


def evaluate_model(
    model: nn.Module,
    loss_function,
    dataloader: DataLoader,
    device: torch.device,
):
    with torch.no_grad():
        model.eval()

        loss_value = 0
        num_correct_samples = 0
        num_all_samples = 0

        for x_batch, y_batch in tqdm(dataloader, desc="Validating...", unit="batch"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_predicted = model(x_batch)

            loss_value += loss_function(y_predicted, y_batch).item() * x_batch.size(0)

            predicted_labels = torch.max(y_predicted, 1)[1]
            num_correct_samples += (predicted_labels == y_batch).sum().item()
            num_all_samples += y_batch.size(0)

        average_loss = loss_value / num_all_samples
        accuracy = num_correct_samples / num_all_samples

    return average_loss, accuracy


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

    # load dataset
    train_loader, val_loader, test_loader = load_fashion_mnist(config.batch_size)

    trained_model = train_model(
        config,
        train_loader,
        val_loader,
        logger,
        device=device,
        random_seed=0
    )

    print(evaluate_model(trained_model, nn.CrossEntropyLoss(), test_loader, device))

    torch.save(
        trained_model.state_dict(),
        f"{config.model_name.value}_{config.optimizer_name.value}.pth",
    )
