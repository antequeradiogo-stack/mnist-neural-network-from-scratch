"""Run the reproducible full MNIST training and held-out evaluation experiment."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import load_data
from src.network import Network

ARCHITECTURE = [784, 30, 10]
DEFAULT_SEED = 2026
DEFAULT_EPOCHS = 30
DEFAULT_MINI_BATCH_SIZE = 10
DEFAULT_LEARNING_RATE = 3.0
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "training_run"


def parse_args() -> argparse.Namespace:
    """Parse reproducible experiment settings, with the requested full run as defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--mini-batch-size", type=int, default=DEFAULT_MINI_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--initialization",
        choices=("glorot_normal", "standard_normal"),
        default="glorot_normal",
        help="Parameter initialization (standard_normal reproduces the original baseline).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--training-size", type=int, default=50_000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def save_model(network: Network, seed: int, path: Path) -> None:
    """Save sizes, parameters, initialization, and seed in a NumPy NPZ archive."""
    arrays: dict[str, np.ndarray] = {
        "sizes": np.asarray(network.sizes, dtype=np.int64),
        "seed": np.asarray(seed, dtype=np.int64),
        "initialization": np.asarray(network.initialization),
    }
    for layer, (weight, bias) in enumerate(zip(network.weights, network.biases)):
        arrays[f"weight_{layer}"] = weight
        arrays[f"bias_{layer}"] = bias
    np.savez_compressed(path, **arrays)


def save_confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, output_dir: Path
) -> np.ndarray:
    """Build, plot, and save the integer confusion matrix and its CSV."""
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)

    csv_path = output_dir / "test_confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *range(10)])
        for digit, row in enumerate(matrix):
            writer.writerow([digit, *row.tolist()])

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, label="Example count")
    axis.set(
        title="MNIST test confusion matrix",
        xlabel="Predicted digit",
        ylabel="True digit",
        xticks=np.arange(10),
        yticks=np.arange(10),
    )
    threshold = matrix.max() / 2
    for row in range(10):
        for column in range(10):
            axis.text(
                column,
                row,
                f"{matrix[row, column]}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(output_dir / "test_confusion_matrix.png", dpi=160)
    plt.close(figure)
    return matrix


def save_test_examples(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    output_dir: Path,
    per_kind: int = 4,
) -> list[int]:
    """Save a grid with correct and incorrect test predictions; return shown indices."""
    correct_indices = np.flatnonzero(predictions == labels)[:per_kind]
    incorrect_indices = np.flatnonzero(predictions != labels)[:per_kind]
    chosen = list(correct_indices) + list(incorrect_indices)

    figure, axes = plt.subplots(2, per_kind, figsize=(10, 6))
    for axis, kind, indices in (
        (axes[0], "Correct", correct_indices),
        (axes[1], "Incorrect", incorrect_indices),
    ):
        for axis_item, index in zip(axis, indices):
            axis_item.imshow(images[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axis_item.set_title(
                f"{kind}\nTrue: {labels[index]} | Pred: {predictions[index]}"
            )
            axis_item.axis("off")
        for axis_item in axis[len(indices) :]:
            axis_item.axis("off")
    figure.suptitle("Held-out test examples after training", y=0.97)
    figure.subplots_adjust(top=0.84, bottom=0.05, hspace=0.55, wspace=0.12)
    figure.savefig(output_dir / "test_prediction_examples.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return [int(index) for index in chosen]


def main() -> None:
    """Train on the selected training data and evaluate validation and test splits."""
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.mini_batch_size <= 0:
        raise SystemExit("--mini-batch-size must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be finite and positive")
    if args.training_size <= 0:
        raise SystemExit("--training-size must be positive")

    (all_training_images, all_training_labels), validation_data, test_data = load_data()
    if args.training_size > len(all_training_images):
        raise SystemExit(
            f"--training-size cannot exceed available examples ({len(all_training_images)})"
        )
    if args.training_size == len(all_training_images):
        selected = np.arange(len(all_training_images))
    else:
        subset_rng = np.random.default_rng(args.seed)
        selected = np.sort(
            subset_rng.choice(len(all_training_images), args.training_size, replace=False)
        )
    training_data = (all_training_images[selected], all_training_labels[selected])
    validation_images, validation_labels = validation_data

    validation_class_counts = np.bincount(validation_labels, minlength=10)
    validation_majority_digit = int(np.argmax(validation_class_counts))
    validation_baseline_correct = int(validation_class_counts[validation_majority_digit])
    validation_baseline = validation_baseline_correct / len(validation_labels)

    network = Network(
        ARCHITECTURE, seed=args.seed, initialization=args.initialization
    )
    print(
        f"Experiment: architecture={ARCHITECTURE}, training examples={len(selected)}, "
        f"validation examples={len(validation_labels)}, epochs={args.epochs}, "
        f"mini_batch_size={args.mini_batch_size}, learning_rate={args.learning_rate}, "
        f"seed={args.seed}, initialization={args.initialization}"
    )
    print(
        f"Validation majority-class baseline: digit {validation_majority_digit}, "
        f"{validation_baseline_correct}/{len(validation_labels)} "
        f"({validation_baseline:.1%})"
    )

    started = time.perf_counter()
    history = network.stochastic_gradient_descent(
        training_data,
        epochs=args.epochs,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.learning_rate,
        evaluation_data=validation_data,
        seed=args.seed,
    )
    training_seconds = time.perf_counter() - started
    print(f"Training and per-epoch metrics time: {training_seconds:.2f} seconds")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    epochs = [int(row["epoch"]) for row in history]
    costs = [float(row["training_cost"]) for row in history]
    accuracies = [float(row["validation_accuracy"]) for row in history]

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, costs, marker="o")
    axis.set(title="Mean training quadratic cost", xlabel="Epoch", ylabel="Cost")
    axis.set_xticks(epochs)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output_dir / "training_cost.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, np.array(accuracies) * 100, marker="o", label="Validation")
    axis.axhline(
        validation_baseline * 100,
        color="tab:red",
        linestyle="--",
        label="Validation majority baseline",
    )
    axis.set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axis.set_xticks(epochs)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "validation_accuracy.png", dpi=160)
    plt.close(figure)

    model_path = args.output_dir / "mnist_network.npz"
    save_model(network, args.seed, model_path)
    print(f"Model parameters saved to {model_path}")
    print(f"Training metrics saved to {metrics_path}")

    # The held-out test data is first evaluated after training and all validation reporting.
    test_images, test_labels = test_data
    test_predictions = np.empty(len(test_labels), dtype=np.int64)
    for index, image in enumerate(test_images):
        test_predictions[index] = int(
            np.argmax(network.feedforward(image.reshape(784, 1)))
        )
    test_correct = int(np.count_nonzero(test_predictions == test_labels))
    test_accuracy = test_correct / len(test_labels)

    test_matrix = save_confusion_matrix(test_labels, test_predictions, args.output_dir)
    shown_indices = save_test_examples(
        test_images, test_labels, test_predictions, args.output_dir
    )
    print("Shown test examples (index: true label -> predicted label):")
    for index in shown_indices:
        print(f"  {index}: {test_labels[index]} -> {test_predictions[index]}")

    summary = {
        "architecture": ARCHITECTURE,
        "training_examples": len(selected),
        "validation_examples": len(validation_labels),
        "test_examples": len(test_labels),
        "epochs": args.epochs,
        "mini_batch_size": args.mini_batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "initialization": args.initialization,
        "training_seconds_including_epoch_metrics": training_seconds,
        "final_training_cost": costs[-1],
        "final_validation_accuracy": accuracies[-1],
        "validation_majority_digit": validation_majority_digit,
        "validation_majority_baseline": validation_baseline,
        "final_test_correct": test_correct,
        "final_test_accuracy": test_accuracy,
        "test_confusion_matrix_diagonal_sum": int(np.trace(test_matrix)),
    }
    summary_path = args.output_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"Final validation accuracy: {int(history[-1]['validation_correct'])}/"
        f"{len(validation_labels)} ({accuracies[-1]:.2%})"
    )
    print(
        f"Final test accuracy: {test_correct}/{len(test_labels)} ({test_accuracy:.2%})"
    )
    print(
        f"Validation majority-class baseline: "
        f"{validation_baseline_correct}/{len(validation_labels)} ({validation_baseline:.2%})"
    )
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Confusion matrix: {args.output_dir / 'test_confusion_matrix.png'}")
    print(f"Correct/incorrect examples: {args.output_dir / 'test_prediction_examples.png'}")
    print(f"Experiment summary: {summary_path}")


if __name__ == "__main__":
    main()
