"""
Class to handle early stopping of model training.
"""

from torch import nn


class EarlyStopping:
    """
    Class to handle early stopping of model training.
    """

    def __init__(self, patience: int, delta: float) -> None:
        """
        Class constructor.

        Args:
            patience (int): Number of epochs to wait for improvement.
            delta (float): Minimum change in validation loss to count as an improvement.
        """
        self.patience = patience
        self.delta = delta

        # Best val loss and model params so far
        self.best_score = None
        self.best_model_state = None

        # Counter of calls with no improvement
        self.counter = 0

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """
        Call after a training epoch with validation loss value.

        Args:
            val_loss (float): Validation loss value.
            model (nn.Module): Torch model being trained.
        """
        if self.best_score is None or val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1

    def stop(self) -> bool:
        """
        Returns information if training should be stopped.

        Returns:
            bool: True if training should be stopped early.
        """
        return self.counter >= self.patience
