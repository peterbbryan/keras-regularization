import inspect
from typing import Any, Callable, Dict, List, Tuple

import fire
import numpy as np
import tensorflow as tf
from PIL import Image


def get_interpolant(interpolant_str: str) -> Image.Resampling:
    """`
    Get interpolant strategy for reshaping input imagery.

    Args:
        interpolant_str: Pillow interpolant name.
            options: NEAREST, BOX, BILINEAR, HAMMING, BICUBIC, LANCZOS, or NEAREST.
    Returns:
        Pillow interpolant.
    """

    interpolants = {
        name: obj
        for name, obj in inspect.getmembers(Image.Resampling)
        if isinstance(obj, Image.Resampling)
    }
    assert (
        interpolant_str in interpolants.keys()
    ), f"Interpolant name {interpolant_str}. Options are {list(interpolants.keys())}"

    return interpolants[interpolant_str]


def get_model(model_str: str) -> Callable[..., tf.keras.models.Model]:
    """
    Get Keras application classifier architecture.

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
    ), f"Model name {model_str} not in Keras applications. Options are {list(models.keys())}"

    return models[model_str]


def get_optimizer(optimizer_str: str) -> Callable[..., tf.keras.optimizers.Optimizer]:
    """
    Get Keras optimizer.

    Args:
        optimizer_str: String name of optimizer.
    Returns:
        Optimizer constructor.
    """

    # get Keras optimizer
    optimizers = {
        optimizer.__name__: optimizer
        for optimizer in tf.keras.optimizers.Optimizer.__subclasses__()
    }
    assert (
        optimizer_str in optimizers.keys()
    ), f"Optimizer name {optimizer_str} not in Keras optimizers. Options are {list(optimizers.keys())}"

    return optimizers[optimizer_str]


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


def rectify_inputs(
    data: np.ndarray, model: tf.keras.models.Model, interpolant: Image.Resampling
) -> np.ndarray:
    """
    Interpolate to correct pixel dimensionality and increase channel count.

    Args:
        data: Shape assumed samples x rows x columns x channels.
        model: Keras applications model.
        interpolant: Pillow interpolant.
    Returns:
        Reshaped array to match model input shape.
    """

    # shape projection
    original_data_shape: Tuple[int, ...] = data.shape
    target_data_shape: Tuple[int, ...] = model.input_shape

    if len(original_data_shape) == 3:
        original_data_shape = (*original_data_shape, 1)

    assert (
        original_data_shape[1] == original_data_shape[2]
    ), "Non-square original dimensionality"
    assert (
        target_data_shape[1] == target_data_shape[2]
    ), "Non-square target dimensionality"
    assert (
        target_data_shape[0] == None
    ), "Assumed None dimension for target dimensionality"
    assert (
        len(original_data_shape) == 4
    ), "Expected original shape samples x rows x columns x channels"
    assert (
        len(target_data_shape) == 4
    ), "Expected target shape samples x rows x columns x channels"

    # drop the samples dimension from target
    _, *target_data_shape_list = target_data_shape

    # dimensions for reshape
    original_samples, *_ = original_data_shape
    _, target_rows, target_columns, target_channels = target_data_shape

    target_array = np.empty(
        (original_samples, target_rows, target_columns, target_channels)
    )

    for index, im in enumerate(data):

        # resized array up to new row/col dimensionality
        im_pillow = Image.fromarray(im)
        resized_im_pillow = im_pillow.resize(
            size=(target_rows, target_columns), resample=interpolant
        )
        resized_im_arr = np.array(resized_im_pillow)

        # repeated array up to necessary channel count
        resize_im_arr_repeated = np.repeat(
            resized_im_arr[:, :, np.newaxis], repeats=target_channels, axis=-1
        )

        target_array[index, :, :, :] = resize_im_arr_repeated

    return target_array


def regularization_experiment(
    model_str: str = "EfficientNetB0",
    constructor_kwargs: Dict[str, Any] = {},
    interpolant_str: str = "BICUBIC",
    optimizer_str: str = "Adam",
    optimizer_kwargs: Dict[str, Any] = {},
    compile_kwargs: Dict[str, Any] = {},
    train_data_ratio: float = 0.1,
) -> None:
    """
    Run regularization experiment demonstrating the impact of the approach.

    Args:
        model_str: String name of model.
        constructor_kwargs: Arguments for model constructor.
        interpolant_str: Pillow interpolant name.
            options: NEAREST, BOX, BILINEAR, HAMMING, BICUBIC, LANCZOS, or NEAREST.
        optimizer_str: String name of optimizer.
        optimizer_kwargs: Arguments for optimizer constructor.
        train_data_ratio: Subsample of the data to load in the the range (0, 1].
    """

    (x_train, y_train), (x_test, y_test) = load_dataset(
        train_data_ratio=train_data_ratio
    )

    # build model
    model_constructor: Callable[..., tf.keras.models.Model] = get_model(
        model_str=model_str
    )
    model: tf.keras.models.Model = model_constructor(**constructor_kwargs)
    interpolant: Image.Resampling = get_interpolant(interpolant_str=interpolant_str)

    # reshape data to input dimensionality
    x_train: np.ndarray = rectify_inputs(
        data=x_train, model=model, interpolant=interpolant
    )
    x_test: np.ndarray = rectify_inputs(
        data=x_test, model=model, interpolant=interpolant
    )

    optimizer_constructor: Callable[..., tf.keras.optimizers.Optimizer] = get_optimizer(
        optimizer_str=optimizer_str
    )
    optimizer: tf.keras.optimizers.Optimizer = optimizer_constructor(**optimizer_kwargs)

    # model.compile(optimizer=optimizer, **)


if __name__ == "__main__":
    fire.Fire(regularization_experiment)
