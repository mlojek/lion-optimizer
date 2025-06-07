"""
Function to select the optimizer class based on the name of the optimizer in the config.
"""

from typing import Type

from torch.optim import SGD, AdamW, Optimizer

from ..config.data_model import OptimizerName
from .lion import Lion


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
        case OptimizerName.ADAMW:
            return AdamW
        case OptimizerName.LION:
            return Lion

    raise ValueError(f"Invalid optimizer name {optimizer_name.value}!")
