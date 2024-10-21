"""Abstract interface for a bias initialization experiment."""

from abc import ABC, abstractmethod
from typing import Type, Tuple, Optional
import os
import numpy as np
import pandas as pd
from keras.api.models import Model
from keras.api.initializers import Initializer
from keras.api.callbacks import EarlyStopping
from keras.api.utils import set_random_seed
from tqdm.keras import TqdmCallback
from tqdm.auto import tqdm, trange
from bias_tracker import BiasTracker
from bias_initializer.keras_impl import BiasInitializer


class BiasExperiment(ABC):
    """Abstract interface for a bias initialization experiment."""

    def __init__(
        self,
        verbose: bool = False,
        maximal_epochs: int = 1000,
        batch_size: int = 1024,
        number_of_repetitions: int = 10,
        random_seed: int = 812,
        histories_dir: str = "histories",
        biases_dir: str = "biases",
    ):
        """
        Create a new bias experiment.

        :param verbose: whether to print debug information
        """
        self.verbose = verbose
        self.maximal_epochs = maximal_epochs
        self.batch_size = batch_size
        self.number_of_repetitions = number_of_repetitions
        self.random_seed = random_seed
        self.histories_dir = histories_dir
        self.biases_dir = biases_dir

    @classmethod
    @abstractmethod
    def load_dataset(
        cls,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Load the dataset for the experiment."""

    @classmethod
    @abstractmethod
    def create_initializer(cls, labels: np.ndarray) -> BiasInitializer:
        """Create a bias initializer from the labels."""

    @classmethod
    @abstractmethod
    def experiment_name(cls) -> str:
        """Return the name of the experiment."""

    @classmethod
    @abstractmethod
    def model_instance(cls, bias_initializer: Optional[Type[Initializer]]) -> Model:
        """Return a compiled model instance for the experiment."""

    def run(self) -> pd.DataFrame:
        """Run the bias initialization experiment."""
        (x_train, y_train), (x_test, y_test) = self.load_dataset()

        initializer = self.create_initializer(y_train)

        performance = []

        for initializer in tqdm(
            [None, initializer],
            desc="Bias Initializer",
            disable=not self.verbose,
            leave=False,
            dynamic_ncols=True,
        ):
            for repetition in trange(
                self.number_of_repetitions,
                desc="Repetition",
                disable=not self.verbose,
                leave=False,
                dynamic_ncols=True,
            ):
                set_random_seed(812 * (repetition + 1))

                model = self.model_instance(bias_initializer=initializer)

                # We extract the output layer name from the model
                output_layer_name = model.layers[-1].name

                history = model.fit(
                    x_train,
                    y_train,
                    validation_split=0.2,
                    shuffle=True,
                    epochs=self.maximal_epochs,
                    batch_size=self.batch_size,
                    verbose=0,
                    callbacks=[
                        TqdmCallback(verbose=int(self.verbose), leave=False),
                        BiasTracker(
                            layer_name=output_layer_name,
                            directory=os.path.join(
                                self.biases_dir,
                                self.experiment_name(),
                                (
                                    "with_bias_initializer"
                                    if initializer is not None
                                    else "without_bias_initializer"
                                ),
                                f"repetition_{repetition}",
                            ),
                        ),
                        EarlyStopping(
                            monitor="val_loss",
                            patience=200,
                            mode="min",
                            restore_best_weights=True,
                        ),
                    ],
                )

                history: pd.DataFrame = pd.DataFrame(history.history)

                path = os.path.join(
                    self.histories_dir,
                    self.experiment_name(),
                    (
                        "with_bias_initializer"
                        if initializer is not None
                        else "without_bias_initializer"
                    ),
                )

                os.makedirs(
                    path,
                    exist_ok=True,
                )

                history.to_csv(
                    os.path.join(
                        path,
                        f"repetition_{repetition}.csv",
                    ),
                    index=False,
                )

                performance.append(
                    {
                        "set": "train",
                        "name": self.experiment_name(),
                        "with_bias_initializer": initializer is not None,
                        **model.evaluate(x_train, y_train, return_dict=True, verbose=0),
                    }
                )
                performance.append(
                    {
                        "set": "test",
                        "name": self.experiment_name(),
                        "with_bias_initializer": initializer is not None,
                        **model.evaluate(x_test, y_test, return_dict=True, verbose=0),
                    }
                )

        return pd.DataFrame(performance)
