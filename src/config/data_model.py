"""
Data model of the experiment configuration.
"""

from enum import Enum

from pydantic import BaseModel


class OptimizerName(Enum):
    """
    Enumeration of available graident optimizers.
    """

    SGD = "sgd"
    "Stochastic gradient descent."

    ADAM = "adam"
    "State-of-the-art Adam optimizer."

    LION = "lion"
    "Our Lion optimizer implementation."


class ModelName(Enum):
    """
    Enumeration of neural network model architectures.
    """

    RES_NET_50 = "res_net_50"
    "ResNet-50 architecture - residual network."

    VIT_B_16 = "vit_b_16"
    "Vit-B 16 architecture - visual transformer."


class EarlyStoppingConfig(BaseModel):
    """
    Configuration of the early stopping.
    """

    patience: int
    "Number of epochs with no improvement in loss value before training is stopped early."

    delta: float
    "Minimum difference in loss value before training is stopped early."


class ExperimentConfig(BaseModel):
    """
    Configuration of an experiment.
    """

    optimizer_name: OptimizerName
    "Gradient optimizer to use."

    learning_rate: float
    "Learning rate of the optimizer."

    model_name: ModelName
    "Architecture of the deep learning model."

    early_stopping: EarlyStoppingConfig
    "Configuration of the EarlyStopping component."

    epochs: int
    "Number of epochs to train the model for."

    batch_size: int
    "Number of samples per one batch."
