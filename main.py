"""Milestone 1: inspect MNIST and visualize a few samples."""

from pathlib import Path

import matplotlib.pyplot as plt

from src.data import load_data


def main() -> None:
    (training_images, training_labels), (validation_images, validation_labels), (test_images, test_labels) = load_data()

    for name, images, labels in (
        ("training", training_images, training_labels),
        ("validation", validation_images, validation_labels),
        ("test", test_images, test_labels),
    ):
        print(
            f"{name}: images={images.shape}, labels={labels.shape}, "
            f"pixel range=[{images.min():.1f}, {images.max():.3f}], "
            f"label range=[{labels.min()}, {labels.max()}]"
        )

    figure, axes = plt.subplots(1, 5, figsize=(10, 2.5))
    for index, axis in enumerate(axes):
        axis.imshow(training_images[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"Label: {training_labels[index]}")
        axis.axis("off")
    figure.suptitle("MNIST training examples (28 × 28 pixels, flattened to 784 inputs)")
    figure.tight_layout()
    preview_path = Path(__file__).resolve().parent / "data" / "mnist_examples.png"
    figure.savefig(preview_path, dpi=150, bbox_inches="tight")
    print(f"Sample visualization saved to {preview_path}")
    plt.show()


if __name__ == "__main__":
    main()
