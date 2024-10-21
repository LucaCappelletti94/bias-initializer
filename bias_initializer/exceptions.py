"""Exceptions use in the bias_initializer module."""

from typing import Type


class BiasInitializerError(Exception):
    """Base exception for the bias_initializer module."""


class IllegalEpsilonError(BiasInitializerError):
    """Exception raised when the epsilon value is not in the correct range."""

    def __init__(self, epsilon: float):
        """
        Create a new IllegalEpsilonError.

        :param epsilon: the illegal epsilon value
        """
        super().__init__(
            f"Epsilon value must be a strictly positive small float, but found {epsilon}."
        )


class UnexpectedScoresShapeError(BiasInitializerError):
    """Exception raised when the scores array has an unexpected shape."""

    def __init__(self, scores_shape: tuple):
        """
        Create a new UnexpectedScoresShapeError.

        :param scores_shape: the shape of the scores array
        """
        super().__init__(
            f"Scores array must be a 1D array, but found shape {scores_shape}."
        )


class EmptyScoresError(BiasInitializerError):
    """Exception raised when the scores array is empty."""

    def __init__(self):
        """Create a new EmptyScoresError."""
        super().__init__("Scores array must not be empty.")


class IllegalScoresDataTypeError(BiasInitializerError):
    """Exception raised when the scores array has an unexpected data type."""

    def __init__(self, scores_dtype: Type):
        """
        Create a new IllegalScoresDataTypeError.

        :param scores_dtype: the data type of the scores array
        """
        super().__init__(
            f"Scores array must have a floating point data type, but found {scores_dtype}."
        )


class NonFiniteScoresError(BiasInitializerError):
    """Exception raised when the scores array contains non-finite values."""

    def __init__(self, number_of_non_finite: int):
        """Create a new NonFiniteScoresError."""
        super().__init__(
            f"Scores array must not contain non-finite values, but found {number_of_non_finite}."
        )
