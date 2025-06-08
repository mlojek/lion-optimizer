"""
Function to create the optimizer based on the experiment config.
"""

from torch.optim import SGD, AdamW, Optimizer

from ..config.data_model import ExperimentConfig, OptimizerName
from .lion import Lion
from torch import nn


def create_optimizer(model: nn.Module, config: ExperimentConfig) -> Optimizer:
    """
    Given the configuration of the experiment return the gradient optimizer.

    Args:
        model (nn.Module): Model to optimize using the optimizer.
        config (ExperimentConfig): Configuration of the experiment with optimizer configuration.

    Returns:
        Optimizer: Gradient optimizer specified in the config.

    Raises:
        ValueError: When the name of the optimizer is invalid.
    """
    match config.optimizer_name:
        case OptimizerName.SGD:
            return SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        case OptimizerName.ADAMW:
            return AdamW(
                model.parameters(),
                lr=config.learning_rate,
                betas=config.betas,
                weight_decay=config.weight_decay,
            )
        case OptimizerName.LION:
            return Lion(
                model.parameters(),
                lr=config.learning_rate,
                betas=config.betas,
                weight_decay=config.weight_decay,
            )

    raise ValueError(f"Invalid optimizer name {optimizer_name.value}!")
