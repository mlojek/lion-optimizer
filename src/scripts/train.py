"""
Model training experiment script.
"""

from logging import Logger

import torch
from torch.nn import Module
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torchvision.datasets.imagenet import ImageNet
from torchvision.models import resnet50, vit_b_16

from ..config.data_model import (
    ExperimentConfig,
    ModelName,
    OptimizerName,
)
from ..optimizers.lion import Lion
from ..utils.early_stopping import EarlyStopping


# TODO random seed
# TODO logger
def train_model(config: ExperimentConfig) -> Module:
    """
    Train the model according to the configuration.

    Args:
        config (ExperimentConfig): Configuration of the experiment.

    Returns:
        Module: Trained pyTorch model.
    """
    match config.model_name:
        case ModelName.RES_NET_50:
            model = resnet50()
        case ModelName.VIT_B_16:
            model = vit_b_16()
        case _:
            raise ValueError(f"Invalid model name {config.model_name.value}!")

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

    dataset = ImageNet()

    # TODO get imagenet dataset
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    loss_function = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(**config.early_stopping.model_dump())

    for epoch in range(config.epochs):
        # Training step
        model.train()

        train_loss = 0
        train_correct_samples = 0
        train_num_samples = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_predicted = model(x_batch)
            loss = loss_function(y_predicted, y_batch)
            model.grad_counter += 1
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
        # TODO
        val_avg_loss, val_accuracy = model.evaluate(val_loader, loss_function)

        logger.info(
            f"Epoch {epoch+1}/{config.epochs}: "
            f"train loss: {train_avg_loss:.4f}, train accuracy: {train_accuracy:.4f}, "
            f"val loss: {val_avg_loss:.4f}, val accuracy: {val_accuracy:.4f}"
        )

        # Early stopping
        early_stopping(val_avg_loss, model)

        if early_stopping.stop():
            logger.info(
                f"Early stopping in epoch {epoch+1} due to lack of improvement."
            )
            model.load_state_dict(early_stopping.best_model_state)
            break

    return model


if __name__ == "__main__":
    pass
