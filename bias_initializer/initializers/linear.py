"""Bias initializer for an output layer with Linear activation."""

import numpy as np
from typeguard import typechecked
from bias_initializer.initializers.bias_initializer import BiasInitializer


class LinearBiasInitializer(BiasInitializer):
    """Initializer that sets the bias based on the prior distribution of the training labels."""

    @property
    def bias(self) -> np.ndarray:
        """Return the bias tensor initialized by the initializer."""
        return self.scores

    @classmethod
    @typechecked
    def from_labels(
        cls, labels: np.ndarray, epsilon: float = 1e-7
    ) -> "LinearBiasInitializer":
        """Create a regression bias initializer from the labels."""
        # We check that labels is a 2D array
        if len(labels.shape) != 2:
            raise ValueError("Labels must be a 2D array.")

        return cls(scores=labels.mean(axis=0), epsilon=epsilon)
