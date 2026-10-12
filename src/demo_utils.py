"""Image preprocessing and inference checks for the interactive demo."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.network import Network

INPUT_SHAPE = (784, 1)
OUTPUT_SHAPE = (10, 1)
EXPECTED_SIZES = [784, 30, 10]


def validate_candidate_network(network: Network) -> None:
    """Reject checkpoints that do not match the selected candidate architecture."""
    if network.sizes != EXPECTED_SIZES:
        raise ValueError(
            f"Expected architecture {EXPECTED_SIZES}, got {network.sizes}"
        )
    if network.initialization != "glorot_normal":
        raise ValueError("Expected the saved Glorot-normal candidate checkpoint")

    expected_weights = [(30, 784), (10, 30)]
    expected_biases = [(30, 1), (10, 1)]
    if len(network.weights) != len(expected_weights) or len(network.biases) != len(expected_biases):
        raise ValueError("Checkpoint has the wrong number of weight matrices or bias vectors")
    for index, (parameter, shape) in enumerate(zip(network.weights, expected_weights)):
        if parameter.shape != shape or not np.isfinite(parameter).all():
            raise ValueError(
                f"Weight matrix {index} must have finite values and shape {shape}"
            )
    for index, (parameter, shape) in enumerate(zip(network.biases, expected_biases)):
        if parameter.shape != shape or not np.isfinite(parameter).all():
            raise ValueError(
                f"Bias vector {index} must have finite values and shape {shape}"
            )


def preprocess_canvas_image(canvas_image: np.ndarray) -> np.ndarray:
    """Convert a canvas RGBA/RGB image to an MNIST-like ``(784, 1)`` input.

    The expected drawing uses white ink on a black background. Ink is cropped,
    scaled without distortion to fit within 20x20 pixels, placed on a 28x28
    image, and shifted so its intensity-weighted center is near the image center.
    Blank canvases return an all-zero input column.
    """
    image = np.asarray(canvas_image, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] not in (3, 4) or min(image.shape[:2]) <= 0:
        raise ValueError("Canvas image must have shape (height, width, 3 or 4)")
    if not np.isfinite(image).all():
        raise ValueError("Canvas image must contain only finite pixel values")
    if image.min() < 0 or image.max() > 255:
        raise ValueError("Canvas pixel values must be in the range [0, 255]")

    if image.max() <= 1.0:
        image = image * 255.0

    rgb = image[:, :, :3]
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4] / 255.0
        rgb = rgb * alpha  # Composite transparency over the black MNIST background.
    grayscale = np.clip(
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722,
        0,
        255,
    ).astype(np.uint8)

    foreground = grayscale > 8  # Ignore faint canvas antialiasing/noise when cropping.
    if not foreground.any():
        return np.zeros(INPUT_SHAPE, dtype=np.float64)

    rows, columns = np.where(foreground)
    cropped = grayscale[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
    crop_height, crop_width = cropped.shape
    scale = min(20 / crop_height, 20 / crop_width)
    resized_size = (
        max(1, int(round(crop_width * scale))),
        max(1, int(round(crop_height * scale))),
    )
    resized = np.asarray(
        Image.fromarray(cropped, mode="L").resize(
            resized_size, resample=Image.Resampling.LANCZOS
        ),
        dtype=np.float64,
    ) / 255.0

    prepared = np.zeros((28, 28), dtype=np.float64)
    top = (28 - resized.shape[0]) // 2
    left = (28 - resized.shape[1]) // 2
    prepared[top : top + resized.shape[0], left : left + resized.shape[1]] = resized

    mass = prepared.sum()
    if mass > 0:
        row_grid, column_grid = np.indices(prepared.shape)
        center_row = float((row_grid * prepared).sum() / mass)
        center_column = float((column_grid * prepared).sum() / mass)
        shift_row = int(round(13.5 - center_row))
        shift_column = int(round(13.5 - center_column))
        centered = np.zeros_like(prepared)
        source_top = max(0, -shift_row)
        source_bottom = min(28, 28 - shift_row)
        source_left = max(0, -shift_column)
        source_right = min(28, 28 - shift_column)
        destination_top = max(0, shift_row)
        destination_left = max(0, shift_column)
        centered[
            destination_top : destination_top + source_bottom - source_top,
            destination_left : destination_left + source_right - source_left,
        ] = prepared[source_top:source_bottom, source_left:source_right]
        prepared = centered

    return prepared.reshape(784, 1)


def predict_digit(network: Network, input_column: np.ndarray) -> tuple[int, np.ndarray]:
    """Return the winning digit and its ten sigmoid output scores."""
    values = np.asarray(input_column, dtype=np.float64)
    if values.shape != INPUT_SHAPE:
        raise ValueError(f"Input must have shape {INPUT_SHAPE}; got {values.shape}")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Input pixels must be finite and in the range [0, 1]")

    scores = np.asarray(network.feedforward(values), dtype=np.float64)
    if scores.shape != OUTPUT_SHAPE:
        raise ValueError(f"Network output must have shape {OUTPUT_SHAPE}; got {scores.shape}")
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Network output scores must be finite values in [0, 1]")
    return int(np.argmax(scores)), scores
