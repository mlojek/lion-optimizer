"""
Utilities related to torch devices.
"""

import torch


def get_available_device() -> torch.device:
    """
    Detect the best performing available device.

    Returns:
        torch.device: The best available torch device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")  # apple silicon

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
