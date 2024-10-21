"""Experiment on the impact of Bias initialization using California Housing dataset."""

from typing import Optional
from experiment import BiasExperiment
from keras.api.models import Sequential
from keras.api.initializers import GlorotNormal
from keras.api.datasets.california_housing import load_data
from keras.api.layers import Dense, BatchNormalization, Input
import numpy as np
from bias_initializer.keras_impl import LinearBiasInitializer
from sklearn.preprocessing import RobustScaler


class CaliforniaHousingExperiment(BiasExperiment):

    @classmethod
    def load_dataset(cls) -> tuple:
        """Load the California Housing dataset."""
        (x_train, y_train), (x_test, y_test) = load_data()
        scaler = RobustScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        return (x_train, y_train), (x_test, y_test)

    @classmethod
    def experiment_name(cls) -> str:
        """Return the name of the experiment."""
        return "CaliforniaHousing"

    @classmethod
    def create_initializer(cls, labels: np.ndarray) -> LinearBiasInitializer:
        """Create a bias initializer from the training labels."""
        return LinearBiasInitializer.from_labels(labels)

    @classmethod
    def model_instance(cls, bias_initializer: Optional[LinearBiasInitializer] = None):
        """Create a simple CNN model for the California Housing dataset."""
        model = Sequential(
            [
                Input(shape=(8,)),
                Dense(32, activation="relu"),
                # BatchNormalization(),
                Dense(64, activation="relu"),
                # BatchNormalization(),
                Dense(64, activation="relu"),
                # BatchNormalization(),
                Dense(
                    1,
                    activation="linear",
                    kernel_initializer=GlorotNormal(),
                    bias_initializer=bias_initializer,
                ),
            ]
        )
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model
