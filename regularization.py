import inspect
from pyexpat import model
from typing import Callable, Dict, List, Tuple

import fire
import numpy as np
import tensorflow as tf


def get_model(model_str: str) -> Callable[..., tf.keras.models.Model]:
    """
    Load Keras application classifier architecture.

    Args:
        model_str: String name of model.
    Returns:
        Model constructor.
    """

    # get Keras applications classes
    models = {
        name: obj
        for name, obj in inspect.getmembers(tf.keras.applications)
        if inspect.isfunction(obj)
    }
    assert (
        model_str in models.keys()
    ), f"Model name {model_str} not in Keras applications. Options are {models.keys()}"

    return models[model_str]


def load_dataset(
    train_data_ratio: float,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load a subset of the train data while maintaining the same class label ratios.

    Args:
        train_data_ratio: Subsample of the data to load in the the range (0, 1].
    Returns:
        Train data and labels, test data and labels.
    """

    assert 0 < train_data_ratio <= 1, "train_data_ratio must be in range (0, 1]"

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # counts of unique values
    unique_labels_tuple: Tuple[np.ndarray, np.ndarray] = np.unique(
        y_train, return_counts=True
    )
    unique_label_dict: Dict[int, int] = dict(zip(*unique_labels_tuple))
    assert len(unique_label_dict) == 10, "Wrong MNIST class count"

    # sample trainx2_data_ratio fraction of each class
    resampled_classes: List[Tuple[np.ndarray, np.ndarray]] = []

    for label, count in unique_label_dict.items():

        # new count of samples for class
        resampled_count = int(train_data_ratio * count)

        x_train_subset = x_train[y_train == label][:resampled_count]
        y_train_subset = y_train[y_train == label][:resampled_count]

        assert np.all(y_train_subset == label), "Label sampling incorrect"

        resampled_classes.append((x_train_subset, y_train_subset))

    # recombine sampled class arrays
    x_train_array_list, y_train_array_list = zip(*resampled_classes)
    x_train_resampled = np.concatenate(x_train_array_list, axis=0)
    y_train_resampled = np.concatenate(y_train_array_list, axis=0)

    return (x_train_resampled, y_train_resampled), (x_test, y_test)


def regularization_experiment(
    model_str: str = "EfficientNetB0", train_data_ratio: float = 0.1
) -> None:
    """

    Args:
        train_data_ratio: Subsample of the data to load in the the range (0, 1].
    """

    (x_train, y_train), (x_test, y_test) = load_dataset(
        train_data_ratio=train_data_ratio
    )

    model_constructor = get_model(model_str=model_str)

    model = model_constructor()

    help(model_constructor)


def train_model(
    data: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    model_str: str,
):
    ...


if __name__ == "__main__":
    fire.Fire(regularization_experiment)
