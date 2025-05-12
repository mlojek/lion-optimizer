"""
Implementation of the Lion optimizer by Google Research.
Publication: https://arxiv.org/pdf/2302.06675

This implementation is a refactored implementation from:
https://github.com/google/automl/blob/master/lion/lion_pytorch.py
published under Apache 2.0 license.
"""

from typing import List, Tuple

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """
    Lion gradient optimizer created by Google Research.
    """

    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float = 1e-4,
        *,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        """
        Initialize the Optimizer.

        Args:
            params (List[torch.Tensor]): Parameters of model to optimize.
            lr (float): Learning rate of the optimizer, default is 1e-4.
            betas (Tuple[float, float]): Coefficients used for computing running averages
                of gradient and its square, default values are 0.9, 0.99.
            weight_decay (float): Weight decay coefficient, default is 0.0.
        """

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate {lr}, value must be positive!")

        for index, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                raise ValueError(
                    f"Invalid beta value at index {index}: {beta}, "
                    "value must be between 0.0 and 1.0!"
                )

        if not 0.0 <= weight_decay < 1.0:
            raise ValueError(
                f"Invalid weight_decay {weight_decay}, ",
                "value must be between 0.0 and 1.0!",
            )

        super().__init__(
            params,
            defaults={
                "lr": lr,
                "betas": betas,
                "weight_decay": weight_decay,
            },
        )

    @torch.no_grad()
    def step(self, closure: callable = None) -> float | None:
        """
        Performs a single optimization step.

        Args:
            closure (callable): Optional, a closure that reevaluates the model
                and returns the loss value.

        Returns:
            Union[float, None]: Loss value if closure was provided, else None.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for params in group["params"]:
                if params.grad is None:
                    continue

                # Perform weights decay.
                params.data.mul_(1 - group["lr"] * group["weight_decay"])

                gradient = params.grad
                state = self.state[params]

                # Initialize state.
                if len(state) == 0:
                    state["exponential_average"] = torch.zeros_like(params)

                exponential_average = state["exponential_average"]
                beta1, beta2 = group["betas"]

                # Update the weights.
                update = exponential_average * beta1 + gradient * (1 - beta1)
                params.add_(update.sign_(), alpha=-group["lr"])

                # Update the exponential average.
                exponential_average.mul_(beta2).add_(gradient, alpha=1 - beta2)

        return loss
