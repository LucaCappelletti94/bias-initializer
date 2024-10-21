"""Bias inizialer for unbalanced prediction tasks."""

from abc import ABC, abstractmethod
from typing import Type, Tuple
from keras.api.initializers import Initializer
from keras.api.utils import register_keras_serializable
import numpy as np
from bias_initializer.initializers import (
    BiasInitializer as OriginalBiasInitializer,
    SigmoidBiasInitializer as OriginalSigmoidBiasInitializer,
    SoftmaxBiasInitializer as OriginalSoftmaxBiasInitializer,
    LinearBiasInitializer as OriginalLinearBiasInitializer,
)


@register_keras_serializable(package="bias_initializer")
class BiasInitializer(Initializer, ABC):
    """Initializer that sets the bias based on the prior distribution of the training labels."""

    def __init__(self, initializer: Type[OriginalBiasInitializer]):
        """
        Create a new initializer with the prior probabilities of the labels.

        :param prior_probs: numpy array of prior probabilities for each label
        """
        self._initializer: Type[OriginalBiasInitializer] = initializer

    @classmethod
    @abstractmethod
    def initializer_class(cls) -> Type[OriginalBiasInitializer]:
        """Return the original bias initializer class."""

    @classmethod
    def from_labels(cls, labels: np.ndarray) -> "LogitBiasInitializer":
        """
        Create an initializer from the one-hot encoded labels.

        :param labels: numpy array of one-hot encoded labels
        :return: the initializer object
        """
        return cls(cls.initializer_class().from_labels(labels))

    def __call__(self, shape: Tuple[int], dtype=None):
        """
        Compute the bias values based on the log-odds of each class.

        :param shape: shape of the bias tensor (should match the number of output classes)
        :param dtype: the dtype of the tensor (optional)
        :return: initialized bias tensor
        """
        bias: np.ndarray = self._initializer.bias
        if bias.shape != shape:
            raise ValueError(
                f"Shape of the bias tensor {bias.shape} does not match the expected shape {shape}."
            )
        if dtype is not None:
            bias = bias.astype(dtype)
        return bias

    @classmethod
    def from_config(cls, config: dict) -> "LogitBiasInitializer":
        """
        Create an initializer from a configuration dictionary.

        :param config: configuration dictionary
        :return: the initializer object
        """
        return cls(cls.initializer_class().from_config(config))

    def get_config(self):
        """
        Return the configuration of the initializer for serialization.
        """
        return self._initializer.get_config()


@register_keras_serializable(package="bias_initializer")
class SigmoidBiasInitializer(BiasInitializer):
    """Bias initializer for an output layer with Sigmoind activation."""

    @classmethod
    def initializer_class(cls) -> Type[OriginalSigmoidBiasInitializer]:
        """Return the original bias initializer class."""
        return OriginalSigmoidBiasInitializer


@register_keras_serializable(package="bias_initializer")
class SoftmaxBiasInitializer(BiasInitializer):
    """Bias inizializer for layers with softmax activation."""

    @classmethod
    def initializer_class(cls) -> Type[OriginalSoftmaxBiasInitializer]:
        """Return the original bias initializer class."""
        return OriginalSoftmaxBiasInitializer


@register_keras_serializable(package="bias_initializer")
class LinearBiasInitializer(BiasInitializer):
    """Bias initializer for an output layer with Linear activation."""

    @classmethod
    def initializer_class(cls) -> Type[OriginalLinearBiasInitializer]:
        """Return the original bias initializer class."""
        return OriginalLinearBiasInitializer
