"""
Implementation of the Lion optimizer.
Publication: https://arxiv.org/pdf/2302.06675
"""

from typing import List

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """
    Lion optimizer discovered by Google. TODO describe in detail
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-4,
        *,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
    ):
        """
        Class constructor.

        Args:
            params (List[Tensor]): Model parameters to optimize.
            lr (float): Learning rate of the optimizer.
            beta1 (float): Weight used in model weights update, default 0.9 (from publication).
            beta2 (float): Weight used in momentum update, default 0.99 (from publication).
            weight_decay (float): Weight decay coefficient, default 0 (from publication).
        """

        if lr <= 0.0:
            raise ValueError(f"Learning rate has to be positive, got invalid value {lr}!")
    
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 value: {beta1}, expected value between 0.0 and 1.0")
    
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}, expected value between 0.0 and 1.0")

        super().__init__(
            params,
            defaults={
                "lr": lr,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": weight_decay,
            },
        )

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Based on program 1 on page 2 of the publication.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)

                # update = update + lr
                p.add_(update.sign_(), alpha=-group["lr"])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
