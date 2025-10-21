import torch
import torch.optim as optim
import math
from typing import Optional


class TransformerOptimizer:
    def __init__(self, model, model_size, factor=1.0, warmup=4000, optimizer=None):
        """
        Initialize Transformer optimizer.
        Parameters:
        - model: Model parameters
        - model_size: Model dimension
        - factor: Learning rate scaling factor
        - warmup: Number of warmup steps
        - optimizer: Base optimizer
        """
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self._step = 0

        if optimizer is None:
            optimizer = optim.Adam(
                model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
            )

        self.inner_optimizer = optimizer

    def step(self, closure=None):
        """
        Execute a single optimization step.
        Parameters:
        - closure: Closure to reevaluate the model and return loss
        Returns:
        - Loss value
        """
        lr = self.rate()
        for param_group in self.inner_optimizer.param_groups:
            param_group["lr"] = lr

        self._step += 1
        return self.inner_optimizer.step(closure)

    def rate(self):
        """
        Calculate current learning rate.
        Returns:
        - Learning rate
        """
        # Ensure step_num is not zero before calculation
        step_num = max(1, self._step)
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step_num ** (-0.5), step_num * self.warmup ** (-1.5))
        )

    def zero_grad(self, set_to_none=False):
        """
        Clear gradients of all parameters.
        Parameters:
        - set_to_none: If True, set gradients to None instead of 0
        """
        self.inner_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """
        Return optimizer state dictionary.
        Returns:
        - State dictionary
        """
        return {
            "step_num": self._step,
            "model_size": self.model_size,
            "factor": self.factor,
            "warmup": self.warmup,
            "optimizer": self.inner_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """
        Load optimizer state.
        Parameters:
        - state_dict: State dictionary
        """
        self._step = state_dict["step_num"]
        self.model_size = state_dict["model_size"]
        self.factor = state_dict["factor"]
        self.warmup = state_dict["warmup"]
        self.inner_optimizer.load_state_dict(state_dict["optimizer"])


def get_optimizer(model, model_size, factor=2.0, warmup=4000):
    """
    Create Transformer optimizer.
    Parameters:
    - model: Model
    - model_size: Model dimension
    - factor: Learning rate scaling factor
    - warmup: Number of warmup steps
    Returns:
    - TransformerOptimizer instance
    """
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    return TransformerOptimizer(
        model=model,
        model_size=model_size,
        factor=factor,
        warmup=warmup,
        optimizer=optimizer,
    )
