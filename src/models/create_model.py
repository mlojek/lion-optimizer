"""
Create a model of architecture specified in the config.
"""

from torch import nn
from torchvision.models import resnet50, vit_b_16

from ..config.data_model import ModelName


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
    match model_name:
        case ModelName.RES_NET_50:
            return resnet50()
        case ModelName.VIT_B_16:
            return vit_b_16()

    raise ValueError(f"Invalid model name {model_name.value}!")
