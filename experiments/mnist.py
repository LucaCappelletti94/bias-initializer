"""Experiment on the impact of Bias initialization using MNIST dataset."""

from typing import Optional
from experiment import BiasExperiment
from extra_keras_metrics import get_minimal_multiclass_metrics
from keras.api.models import Sequential
from keras.api.initializers import GlorotNormal
from keras.api.layers import (
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
    Input,
    BatchNormalization,
)
from keras.api.datasets.mnist import load_data
import numpy as np
from bias_initializer.keras_impl import SigmoidBiasInitializer


class MNISTExperiment(BiasExperiment):
    """Experiment on the impact of Bias initialization using MNIST dataset."""

    @classmethod
    def load_dataset(cls) -> tuple:
        """Load the MNIST dataset."""
        (x_train, y_train), (x_test, y_test) = load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32") / 255
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32") / 255
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]
        return (x_train, y_train), (x_test, y_test)

    @classmethod
    def experiment_name(cls) -> str:
        """Return the name of the experiment."""
        return "MNIST"

    @classmethod
    def create_initializer(cls, labels: np.ndarray) -> SigmoidBiasInitializer:
        """Create a bias initializer from the training labels."""
        return SigmoidBiasInitializer.from_labels(labels)

    @classmethod
    def model_instance(cls, bias_initializer: Optional[SigmoidBiasInitializer] = None):
        """Create a simple CNN model for the MNIST dataset."""
        model = Sequential(
            [
                Input(shape=(28, 28, 1)),
                Conv2D(64, kernel_size=(3, 3), activation="relu"),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation="relu"),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dense(
                    10,
                    activation="sigmoid",
                    kernel_initializer=GlorotNormal(),
                    bias_initializer=bias_initializer,
                ),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=get_minimal_multiclass_metrics(),
        )
        return model
