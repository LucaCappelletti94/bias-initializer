"""Keras Callback to store at a given position the bias vector of a layer of a neural network."""

from typing import Any, Dict, List
import os
from keras.api.layers import Layer
from keras.api.callbacks import Callback
from typeguard import typechecked
import numpy as np


class BiasTracker(Callback):
    """Keras Callback to store at a given position the bias vector of a layer of a neural network."""

    @typechecked
    def __init__(self, layer_name: str, directory: str):
        """
        Create a new BiasTracker.

        :param layer_name: name of the layer to track
        :param directory: directory where to store the bias vector
        """
        super().__init__()
        if not os.path.exists(directory):
            os.makedirs(directory)
        self._layer_name = layer_name
        self._directory = directory
        self._epoch_biases: List[np.ndarray] = []

    def on_train_begin(self, logs: Dict[str, Any] = None):
        """Initialize the list of bias vectors."""
        layer: Layer = self.model.get_layer(self._layer_name)
        bias: np.ndarray = layer.get_weights()[1]
        self._epoch_biases.append(bias)

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Convert the list of bias vectors to a numpy array and save it to disk."""
        layer: Layer = self.model.get_layer(self._layer_name)
        bias: np.ndarray = layer.get_weights()[1]
        self._epoch_biases.append(bias)

    def on_train_end(self, logs: Dict[str, Any] = None):
        """Remove the list of bias vectors."""
        np.save(f"{self._directory}/biases.npy", np.array(self._epoch_biases))
