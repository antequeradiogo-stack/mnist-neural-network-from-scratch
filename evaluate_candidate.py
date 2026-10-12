"""Verify the selected checkpoint on validation, then report held-out test metrics."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import load_data
from src.network import Network

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "mnist_glorot_candidate.npz"
CHECKPOINT_METADATA = ROOT / "models" / "mnist_glorot_candidate.json"
OUTPUT_DIR = ROOT / "outputs" / "candidate_test_evaluation"


def load_checkpoint(path: Path) -> Network:
    """Restore weights and biases from the selected Glorot model archive."""
    with np.load(path, allow_pickle=False) as saved:
        sizes = saved["sizes"].astype(int).tolist()
        seed = int(saved["seed"])
        initialization = str(saved["initialization"].item())
        network = Network(sizes, seed=seed, initialization=initialization)
        network.weights = [saved[f"weight_{i}"].copy() for i in range(len(sizes) - 1)]
        network.biases = [saved[f"bias_{i}"].copy() for i in range(len(sizes) - 1)]
    return network


def predict(network: Network, images: np.ndarray) -> np.ndarray:
    """Classify images using the saved network parameters."""
    predictions = np.empty(len(images), dtype=np.int64)
    for index, image in enumerate(images):
        output = network.feedforward(image.reshape(network.sizes[0], 1))
        predictions[index] = int(np.argmax(output))
    return predictions


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Compute confusion matrix and per-class precision, recall, and F1."""
    class_count = 10
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    true_positive = np.diag(matrix)
    predicted_count = matrix.sum(axis=0)
    support = matrix.sum(axis=1)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(class_count, dtype=np.float64),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(class_count, dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(class_count, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    per_digit = [
        {
            "digit": digit,
            "precision": float(precision[digit]),
            "recall": float(recall[digit]),
            "f1": float(f1[digit]),
            "correct": int(true_positive[digit]),
            "support": int(support[digit]),
            "predicted_count": int(predicted_count[digit]),
        }
        for digit in range(class_count)
    ]
    return matrix, per_digit


def save_confusion_matrix(matrix: np.ndarray) -> None:
    with (OUTPUT_DIR / "test_confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *range(10)])
        for digit, row in enumerate(matrix):
            writer.writerow([digit, *row.tolist()])

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, label="Test examples")
    axis.set(
        title="Candidate checkpoint test confusion matrix",
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
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "test_confusion_matrix.png", dpi=160)
    plt.close(figure)


def save_prediction_examples(
    images: np.ndarray, labels: np.ndarray, predictions: np.ndarray
) -> None:
    """Plot a few actual correct and incorrect candidate predictions."""
    correct_indices = np.flatnonzero(predictions == labels)[:4]
    incorrect_indices = np.flatnonzero(predictions != labels)[:4]
    figure, axes = plt.subplots(2, 4, figsize=(10, 6))
    for row_axes, title, indices in (
        (axes[0], "Correct", correct_indices),
        (axes[1], "Incorrect", incorrect_indices),
    ):
        for axis, index in zip(row_axes, indices):
            axis.imshow(images[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axis.set_title(
                f"{title}\nTrue {labels[index]} | Pred {predictions[index]}",
                fontsize=11,
                pad=7,
            )
            axis.axis("off")
        for axis in row_axes[len(indices) :]:
            axis.axis("off")
    figure.suptitle("Saved Glorot candidate: held-out test examples", y=0.98)
    figure.subplots_adjust(top=0.86, bottom=0.05, hspace=0.38, wspace=0.15)
    figure.savefig(OUTPUT_DIR / "test_prediction_examples.png", dpi=160)
    plt.close(figure)


def main() -> None:
    if not CHECKPOINT.exists() or not CHECKPOINT_METADATA.exists():
        raise SystemExit("The tracked candidate checkpoint and metadata are required")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_metadata = json.loads(CHECKPOINT_METADATA.read_text(encoding="utf-8"))
    checkpoint_hash = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if checkpoint_hash != checkpoint_metadata["checkpoint_sha256"]:
        raise RuntimeError("Candidate checkpoint SHA-256 does not match its metadata")

    network = load_checkpoint(CHECKPOINT)
    if network.sizes != [784, 30, 10] or network.initialization != "glorot_normal":
        raise SystemExit("Checkpoint does not match the selected Glorot [784, 30, 10] candidate")

    # The loader returns three splits. Verify validation first; only then evaluate test.
    _training_data, validation_data, test_data = load_data()
    validation_images, validation_labels = validation_data
    expected_accuracy = checkpoint_metadata["validation_accuracy"]
    validation_predictions = predict(network, validation_images)
    validation_accuracy = float(np.mean(validation_predictions == validation_labels))
    if not np.isclose(validation_accuracy, expected_accuracy, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Checkpoint validation accuracy {validation_accuracy:.12f} does not match "
            f"recorded {expected_accuracy:.12f}; stopping before test evaluation"
        )
    print(
        f"Checkpoint validation verification: {np.count_nonzero(validation_predictions == validation_labels)}"
        f"/{len(validation_labels)} ({validation_accuracy:.2%}), matches saved comparison."
    )

    # Test predictions are computed only after the saved checkpoint passes validation verification.
    test_images, test_labels = test_data
    test_predictions = predict(network, test_images)
    correct = int(np.count_nonzero(test_predictions == test_labels))
    accuracy = correct / len(test_labels)
    confusion, per_digit = classification_metrics(test_labels, test_predictions)
    with (OUTPUT_DIR / "test_per_digit_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(per_digit[0]))
        writer.writeheader()
        writer.writerows(per_digit)
    save_confusion_matrix(confusion)
    save_prediction_examples(test_images, test_labels, test_predictions)

    report = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "architecture": network.sizes,
        "initialization": network.initialization,
        "seed": int(checkpoint_metadata["seed"]),
        "test_examples": int(len(test_labels)),
        "correctly_classified": correct,
        "test_accuracy": accuracy,
        "validation_accuracy_verified_before_test": validation_accuracy,
        "per_digit_metrics": per_digit,
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        "test_results_used_for_tuning": False,
    }
    (OUTPUT_DIR / "test_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Test accuracy: {correct}/{len(test_labels)} ({accuracy:.2%})")
    print(f"Saved test-only metrics and confusion matrix to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
