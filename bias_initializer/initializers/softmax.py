"""Bias inizializer for layers with softmax activation."""

import warnings
import numpy as np
from typeguard import typechecked
from bias_initializer.initializers.bias_initializer import BiasInitializer


class SoftmaxBiasInitializer(BiasInitializer):
    """Initializer that sets the bias based on the prior distribution of the training labels."""

    @property
    def bias(self) -> np.ndarray:
        """Return the bias tensor initialized by the initializer."""
        log_odds = np.log(self.epsilon + self.scores)

        assert not np.any(np.isnan(log_odds)), "Log-odds contain NaN values."
        assert not np.any(np.isinf(log_odds)), "Log-odds contain infinite values."

        return log_odds

    @classmethod
    @typechecked
    def from_labels(
        cls, labels: np.ndarray, epsilon: float = 1e-7
    ) -> "SoftmaxBiasInitializer":
        """Return new bias initializer based on the labels.

        Parameters
        ----------
        labels : np.ndarray
            The labels can either be one-hot encoded or integer encoded.
            In the former case, the shape of the array should be (n_samples, n_classes),
            in the latter case, the shape should be (n_samples,) or (n_samples, 1).
            Note that, if the labels are integer encoded and some labels are missing,
            the one-hot encoding will be created with the number of classes equal to the
            maximum available label value, as we cannot know the total number of classes.
        epsilon : float, optional
            The epsilon value to prevent divide by zero, by default 1e-7.
        """
        # If the labels are provided in a shape that suggests they are integer encoded,
        # we convert them to one-hot encoded labels.
        if not len(labels.shape) <= 2:
            raise ValueError(
                "Labels must be a 1D or 2D array, "
                f"but got array with shape {labels.shape}."
            )

        if len(labels.shape) == 1 or labels.shape[1] == 1:
            n_classes = np.max(labels) + 1
            one_hot_labels = np.zeros((labels.size, n_classes))
            one_hot_labels[np.arange(labels.size), labels.flatten()] = 1
            labels = one_hot_labels

        # Calculate the frequency of each label
        labels_sum = np.sum(labels, axis=0)

        # If all labels are missing, we raise an error
        if np.all(labels_sum == 0):
            raise ValueError("All labels are missing.")

        # If one or more labels never appear in the training set, we issue a warning
        if np.any(labels_sum == 0):
            num_missing_labels = np.sum(labels_sum == 0)
            warnings.warn(
                f"{num_missing_labels} label(s) never appear in the training set.",
                UserWarning,
            )

        return cls(scores=labels_sum / np.sum(labels_sum), epsilon=epsilon)
