"""Experiments executor."""

import argparse
from typing import List, Type, Optional, Dict
import os
import shutil
from multiprocessing import Pool, cpu_count
import silence_tensorflow.auto  # pylint: disable=unused-import
from experiment import BiasExperiment
from mnist import MNISTExperiment
from cifar10 import CIFAR10Experiment
from cifar100 import CIFAR100Experiment
from fashion_mnist import FashionMNISTExperiment
from california_housing import CaliforniaHousingExperiment
from sklearn.decomposition import PCA
from plot_keras_history import plot_history
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange


def run():
    """Run the experiments."""
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
        "--verbose", action="store_true", help="Print the results of the experiments."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a smoke test on the experiments.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the previous results of the experiments.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of repetitions for each experiment.",
    )
    args = parser.parse_args()

    if args.clear:
        shutil.rmtree("histories", ignore_errors=True)
        shutil.rmtree("biases", ignore_errors=True)
        if os.path.exists("performance.csv"):
            os.remove("performance.csv")

    maximal_epochs = 1 if args.smoke_test else 10_000
    repetitions = 1 if args.smoke_test else args.repetitions

    experiments: List[Type[BiasExperiment]] = [
        # MNISTExperiment(
        #     verbose=args.verbose,
        #     maximal_epochs=maximal_epochs,
        #     number_of_repetitions=repetitions,
        # ),
        # FashionMNISTExperiment(
        #     verbose=args.verbose,
        #     maximal_epochs=maximal_epochs,
        #     number_of_repetitions=repetitions,
        # ),
        # CIFAR10Experiment(
        #     verbose=args.verbose,
        #     maximal_epochs=maximal_epochs,
        #     number_of_repetitions=repetitions,
        # ),
        # CIFAR100Experiment(
        #     verbose=args.verbose,
        #     maximal_epochs=maximal_epochs,
        #     number_of_repetitions=repetitions,
        # ),
        CaliforniaHousingExperiment(
            verbose=args.verbose,
            maximal_epochs=maximal_epochs,
            number_of_repetitions=repetitions,
        ),
    ]

    performance: List[pd.DataFrame] = []
    previous_performance: Optional[pd.DataFrame] = None
    if os.path.exists("performance.csv"):
        previous_performance = pd.read_csv("performance.csv")
        performance.append(previous_performance)

    # We filter away experiments that are already present in the histories directory
    filtered_experiments = [
        experiment
        for experiment in experiments
        if previous_performance is None
        or experiment.experiment_name() not in previous_performance["name"].values
    ]

    for experiment in tqdm(
        filtered_experiments,
        desc="Experiments",
        disable=not args.verbose,
        dynamic_ncols=True,
        leave=False,
    ):
        performance.append(experiment.run())

    pd.concat(performance).to_csv("performance.csv", index=False)


def analyse_histories():
    """Analyse the histories of the experiments."""

    results = []
    for experiment in tqdm(
        os.listdir("histories"),
        desc="Experiments",
        dynamic_ncols=True,
        leave=False,
    ):
        experiment_dir = os.path.join("histories", experiment)
        if not os.path.isdir(experiment_dir):
            continue

        for bias_approach in tqdm(
            os.listdir(experiment_dir),
            desc="Bias approach",
            dynamic_ncols=True,
            leave=False,
        ):
            history_path = os.path.join(experiment_dir, bias_approach)
            if not os.path.isdir(history_path):
                continue

            all_histories = []
            early_stopping_epochs = []
            for repetition in tqdm(
                os.listdir(history_path),
                desc="Repetitions",
                dynamic_ncols=True,
                leave=False,
            ):
                if not repetition.endswith(".csv"):
                    continue

                repetition_path = os.path.join(history_path, repetition)
                history = pd.read_csv(repetition_path)

                # We identify the epoch with minimal validation loss
                early_stopping_epoch = history["val_loss"].idxmin()
                early_stopping_epochs.append(early_stopping_epoch)

                all_histories.append(history)

            results.append(
                {
                    "experiment": experiment,
                    "bias_approach": bias_approach,
                    "mean_early_stopping_epoch": np.mean(early_stopping_epochs),
                }
            )

            assert len(all_histories) > 0

            plot_history(
                histories=all_histories,
                monitor="val_loss",
                monitor_mode="min",
                path=os.path.join(history_path, "metrics.png"),
            )

    results = pd.DataFrame(results)
    results.to_csv("history_analysis.csv", index=False)


def analyse_biases():
    """For each repetition, we load the biases, chain all epochs, and compute the L2 distance from the expected optimal bias."""
    for experiment in tqdm(
        os.listdir("biases"),
        desc="Experiments",
        dynamic_ncols=True,
        leave=False,
    ):
        repetitions_per_approach: Dict[str, List[np.ndarray]] = {}

        for bias_approach in tqdm(
            os.listdir(os.path.join("biases", experiment)),
            desc="Bias approach",
            dynamic_ncols=True,
            leave=False,
        ):
            bias_path = os.path.join("biases", experiment, bias_approach)
            if not os.path.isdir(bias_path):
                continue

            for repetition in tqdm(
                os.listdir(bias_path),
                desc="Repetitions",
                dynamic_ncols=True,
                leave=False,
            ):
                repetition_path = os.path.join(bias_path, repetition)
                if not os.path.isdir(repetition_path):
                    continue

                biases_in_epoch: np.ndarray = np.vstack(
                    [
                        np.load(os.path.join(repetition_path, path))
                        for path in tqdm(os.listdir(repetition_path), desc="Epochs", leave=False)
                        if path.endswith(".npy")
                    ]
                )

                repetitions_per_approach.setdefault(bias_approach, []).append(
                    biases_in_epoch
                )

        # The optimal bias is the first bias of the 'with_bias_initializer' approach,
        # of each repetition. Since this initial bias is equal for all repetitions,
        # we can just take the first repetition.
        optimal_bias = repetitions_per_approach["with_bias_initializer"][0][0]

        # We compute the L2 distance from the optimal bias for each repetition,
        # at each batch (row).
        l2_distances_per_approach: dict[str, List[np.ndarray]] = {}
        for bias_approach, repetitions in repetitions_per_approach.items():
            l2_distances = []
            for repetition in repetitions:
                l2_distances.append(repetition.flatten())#np.linalg.norm(repetition - optimal_bias, axis=1))
            l2_distances_per_approach[bias_approach] = l2_distances

        # We plot the L2 distance for each repetition, for each bias approach.
        fig, ax = plt.subplots(figsize=(16, 8), dpi=200)
        colors = {
            "with_bias_initializer": "tab:blue",
            "without_bias_initializer": "tab:orange",
        }
        for bias_approach, l2_distances in l2_distances_per_approach.items():
            for l2_distance in l2_distances:
                ax.plot(l2_distance, alpha=0.5, color=colors[bias_approach])
        ax.set_xlabel("Batch")
        ax.set_ylabel("Bias")
        ax.set_yscale("symlog")
        ax.set_title(experiment)
        ax.legend(l2_distances_per_approach.keys())
        fig.savefig(os.path.join("biases", experiment, "l2_distances.png"))
        plt.close()


def draw_frame(
    experiment: str,
    range_of_frames: range,
    pca_per_approach: dict[str, List[np.ndarray]],
):
    for frame in range_of_frames:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        for bias_approach, pca_repetitions in pca_per_approach.items():
            for repetition in pca_repetitions:
                this_frame = min(frame, repetition.shape[0] - 1)
                ax.plot(
                    repetition[: this_frame + 1, 0],
                    repetition[: this_frame + 1, 1],
                    alpha=0.5,
                    color=(
                        "tab:blue"
                        if bias_approach == "with_bias_initializer"
                        else "tab:orange"
                    ),
                )
                ax.plot(
                    repetition[0, 0],
                    repetition[0, 1],
                    marker="s",
                    markersize=10,
                    color=(
                        "tab:blue"
                        if bias_approach == "with_bias_initializer"
                        else "tab:orange"
                    ),
                )
                ax.plot(
                    repetition[this_frame, 0],
                    repetition[this_frame, 1],
                    marker="o",
                    markersize=10,
                    color=(
                        "tab:blue"
                        if bias_approach == "with_bias_initializer"
                        else "tab:orange"
                    ),
                )
        ax.set_title(experiment)
        ax.axis("equal")
        ax.axis("off")
        fig.savefig(
            os.path.join("biases", experiment, "frames", f"frame_{frame:04d}.png")
        )
        plt.close()


def _draw_frame(args):
    draw_frame(*args)


def animate_biases():
    """Creates animations for the biases.

    Implementation details
    ----------------------
    We load the biases for each repetition, and then we initially concatenate
    all of the repetitions within a single experiment so as to compute the PCA
    of the biases. We then decompose each repetition into its principal components,
    and start to create the frames of the animation.

    Each frame always shows the starting point for the biases, as an orange or blue
    square depending on the bias approach. The subsequent frames show the divergence of
    the biases from the starting point, always including all repetitions. This shows
    the trace of the biases, with increasing transparency, where the points currently
    being plotted are the most opaque. Not all repetitions have the same length, so
    when they end, we just keep the last point for the remaining frames.
    """
    for experiment in tqdm(
        os.listdir("biases"),
        desc="Experiments",
        dynamic_ncols=True,
        leave=False,
    ):
        repetitions_per_approach: Dict[str, List[np.ndarray]] = {}

        for bias_approach in tqdm(
            os.listdir(os.path.join("biases", experiment)),
            desc="Bias approach",
            dynamic_ncols=True,
            leave=False,
        ):
            bias_path = os.path.join("biases", experiment, bias_approach)
            if not os.path.isdir(bias_path):
                continue

            for repetition in tqdm(
                os.listdir(bias_path),
                desc="Repetitions",
                dynamic_ncols=True,
                leave=False,
            ):
                repetition_path = os.path.join(bias_path, repetition)
                if not os.path.isdir(repetition_path):
                    continue

                biases_in_epoch: np.ndarray = np.vstack(
                    [
                        np.load(os.path.join(repetition_path, path))
                        for path in tqdm(
                            os.listdir(repetition_path),
                            desc="Epochs",
                            leave=False,
                        )
                        if path.endswith(".npy")
                    ]
                )

                repetitions_per_approach.setdefault(bias_approach, []).append(
                    biases_in_epoch
                )

        # We concatenate all biases
        concatenated_biases = np.vstack(
            [
                np.vstack(repetitions)
                for repetitions in repetitions_per_approach.values()
            ]
        )

        # We compute the PCA of the biases
        pca = PCA(n_components=2)
        pca.fit(concatenated_biases)

        # We decompose each repetition into its principal components
        pca_per_approach: dict[str, List[np.ndarray]] = {
            bias_approach: [pca.transform(repetition) for repetition in repetitions]
            for bias_approach, repetitions in repetitions_per_approach.items()
        }

        longest_repetitions = max(
            repetition.shape[0]
            for repetitions in repetitions_per_approach.values()
            for repetition in repetitions
        )

        # We create a directory to store the frames of the animation
        os.makedirs(os.path.join("biases", experiment, "frames"), exist_ok=True)

        number_of_cpus = cpu_count()
        tasks = [
            (
                experiment,
                range(
                    int(longest_repetitions / number_of_cpus * num),
                    int(longest_repetitions / number_of_cpus * (num + 1)),
                ),
                pca_per_approach,
            )
            for num in range(number_of_cpus)
        ]

        with Pool(number_of_cpus) as pool:
            list(
                tqdm(
                    pool.imap(_draw_frame, tasks),
                    total=number_of_cpus,
                    leave=False,
                    desc="Frames",
                )
            )

        # We create the animation
        os.system(
            f"ffmpeg -y -r 10 -i 'biases/{experiment}/frames/frame_%04d.png' -c:v libx264 -vf 'fps=60,format=yuv420p' 'biases/{experiment}/animation.mp4'"
        )


if __name__ == "__main__":
    run()
    analyse_histories()
    analyse_biases()
    # animate_biases()
