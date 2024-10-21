"""Bias initializer for an output layer with Sigmoind activation."""

import warnings
import numpy as np
from typeguard import typechecked
from bias_initializer.initializers.bias_initializer import BiasInitializer


class SigmoidBiasInitializer(BiasInitializer):
    """Initializer that sets the bias based on the prior distribution of the training labels."""

    @property
    def bias(self) -> np.ndarray:
        """Return the bias tensor initialized by the initializer."""
        log_odds = np.log(self.epsilon + self.scores / (1 - self.scores + self.epsilon))

        assert not np.any(np.isnan(log_odds)), "Log-odds contain NaN values."
        assert not np.any(np.isinf(log_odds)), "Log-odds contain infinite values."

        return log_odds

    @classmethod
    @typechecked
    def from_labels(
        cls, labels: np.ndarray, epsilon: float = 1e-7
    ) -> "SigmoidBiasInitializer":
        """
        Create an initializer from the one-hot encoded labels.

        :param labels: numpy array of one-hot encoded labels
        :param epsilon: epsilon value to prevent divide by zero
        :return: the initializer object
        """
        # We check that labels is a 2D array
        if len(labels.shape) != 2:
            raise ValueError("Labels must be a 2D array.")

        # We check that labels is a binary array
        if not np.array_equal(np.unique(labels), np.array([0, 1])):
            raise ValueError("Labels must be a binary array.")

        # Calculate the frequency of each label
        label_means = np.mean(labels, axis=0)

        # If all labels are missing, we raise an error
        if np.all(label_means == 0):
            raise ValueError("All labels are missing.")

        # If one or more labels never appear in the training set, we issue a warning
        if np.any(label_means == 0):
            num_missing_labels = np.sum(label_means == 0)
            warnings.warn(
                f"{num_missing_labels} label(s) never appear in the training set.",
                UserWarning,
            )

        return cls(scores=label_means, epsilon=epsilon)
