"""Abstract class defining a bias initializer for a neural network output layer."""

from typing import Any, Dict, Type
from abc import ABC, abstractmethod
from typeguard import typechecked
import numpy as np
from bias_initializer.exceptions import (
    IllegalEpsilonError,
    UnexpectedScoresShapeError,
    EmptyScoresError,
    IllegalScoresDataTypeError,
    NonFiniteScoresError,
)


class BiasInitializer(ABC):
    """Initializer that sets the bias based on the prior distribution of the training labels."""

    @typechecked
    def __init__(self, scores: np.ndarray, epsilon: float = 1e-7):
        """
        Create a new initializer with the prior probabilities of the labels.

        :param scores: numpy array of prior probabilities for each label
        """
        # Some healthy defensive programming
        if epsilon <= 0.0 or epsilon >= 1.0:
            raise IllegalEpsilonError(epsilon)

        if scores.ndim != 1:
            raise UnexpectedScoresShapeError(scores.shape)

        if scores.size == 0:
            raise EmptyScoresError()

        if scores.dtype not in [np.float32, np.float64, np.float128, np.float16]:
            raise IllegalScoresDataTypeError(scores.dtype)

        if not np.all(np.isfinite(scores)):
            raise NonFiniteScoresError(scores.size - np.sum(np.isfinite(scores)))

        self._scores = scores.astype(np.float32)
        self._epsilon = epsilon

    @property
    def scores(self) -> np.ndarray:
        """Return the scores used for the bias initialization."""
        return self._scores

    @property
    @abstractmethod
    def bias(self) -> np.ndarray:
        """Return the bias tensor initialized by the initializer."""

    @property
    def epsilon(self) -> float:
        """Return the epsilon used for the bias initialization."""
        return self._epsilon

    @abstractmethod
    @classmethod
    def from_labels(cls, labels: np.ndarray, epsilon: float) -> Type["BiasInitializer"]:
        """Create a bias initializer from the labels."""

    @classmethod
    @typechecked
    def from_config(cls, config: Dict[str, Any]) -> Type["BiasInitializer"]:
        """
        Create an initializer from a configuration dictionary.

        :param config: configuration dictionary
        :return: the initializer object
        """
        return cls(scores=np.array(config["scores"]), epsilon=config["epsilon"])

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the initializer for serialization.
        """
        return {"scores": self._scores.tolist(), "epsilon": self._epsilon}
