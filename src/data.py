"""MNIST data loading helpers."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

MNIST_URL = "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mnist.pkl.gz"


def load_data(path: Path = DATA_PATH) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Load Nielsen's gzip-compressed MNIST pickle and return three (images, labels) splits.

    Images are float64 arrays normalized to [0, 1], with shape (N, 784).
    Labels are integer arrays with shape (N,).
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MNIST from {MNIST_URL}")
        urlretrieve(MNIST_URL, path)

    with gzip.open(path, "rb") as file:
        training, validation, test = pickle.load(file, encoding="latin1")

    splits = []
    for name, (images, labels) in zip(
        ("training", "validation", "test"), (training, validation, test)
    ):
        images = np.asarray(images, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if images.ndim != 2 or images.shape[1] != 784:
            raise ValueError(f"{name} images should have shape (N, 784); got {images.shape}")
        if labels.shape != (images.shape[0],):
            raise ValueError(f"{name} labels should have shape ({images.shape[0]},); got {labels.shape}")
        if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError(f"{name} pixels should be finite and normalized to [0, 1]")
        splits.append((images, labels))
    return tuple(splits)
