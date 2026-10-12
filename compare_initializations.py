"""Validation-only comparison of standard-normal and Glorot-normal initialization.

This script deliberately uses only the training and validation splits. It does not
evaluate or report any held-out test-set results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import load_data
from src.network import Network, sigmoid, sigmoid_prime

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "glorot_comparison"
BASELINE_MODEL = ROOT / "models" / "mnist_standard_normal_baseline.npz"
BASELINE_METADATA = ROOT / "models" / "mnist_standard_normal_baseline.json"
BASELINE_METRICS = ROOT / "results" / "standard_normal_training_metrics.csv"
ARCHITECTURE = [784, 30, 10]
SEED = 2026
EPOCHS = 30
MINI_BATCH_SIZE = 10
LEARNING_RATE = 3.0


def restore_baseline(path: Path) -> Network:
    """Load the original standard-normal full-run checkpoint without modifying it."""
    metadata = json.loads(BASELINE_METADATA.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata["checkpoint_sha256"]:
        raise RuntimeError("Standard-normal baseline checkpoint SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as saved:
        sizes = saved["sizes"].tolist()
        network = Network(
            sizes, seed=int(saved["seed"]), initialization="standard_normal"
        )
        network.weights = [saved[f"weight_{i}"].copy() for i in range(len(sizes) - 1)]
        network.biases = [saved[f"bias_{i}"].copy() for i in range(len(sizes) - 1)]
    return network


def run_trace(network: Network, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return output pre-activations and activations for validation images in batches."""
    zs = np.empty((len(images), network.sizes[-1]), dtype=np.float64)
    activations = np.empty_like(zs)
    batch_size = 512
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        hidden_z = batch @ network.weights[0].T + network.biases[0].ravel()
        hidden_a = sigmoid(hidden_z)
        output_z = hidden_a @ network.weights[1].T + network.biases[1].ravel()
        zs[start : start + len(batch)] = output_z
        activations[start : start + len(batch)] = sigmoid(output_z)
    return zs, activations


def summarize(values: np.ndarray) -> dict[str, float]:
    """Compact finite distribution summary."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def evaluate_validation(
    network: Network, images: np.ndarray, labels: np.ndarray, name: str
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Compute validation-only classification and neuron saturation diagnostics."""
    preactivations, outputs = run_trace(network, images)
    predictions = np.argmax(outputs, axis=1)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    support = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive, predicted_counts, out=np.zeros(10, dtype=float), where=predicted_counts != 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros(10, dtype=float), where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(10, dtype=float),
        where=(precision + recall) != 0,
    )

    class_metrics = [
        {
            "digit": digit,
            "precision": float(precision[digit]),
            "recall": float(recall[digit]),
            "f1": float(f1[digit]),
            "support": int(support[digit]),
            "predicted_count": int(predicted_counts[digit]),
        }
        for digit in range(10)
    ]
    with (OUTPUT_DIR / f"{name}_per_digit_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(class_metrics[0]))
        writer.writeheader()
        writer.writerows(class_metrics)

    with (OUTPUT_DIR / f"{name}_confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *range(10)])
        for digit, row in enumerate(confusion):
            writer.writerow([digit, *row.tolist()])
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(confusion, cmap="Blues", interpolation="nearest")
    figure.colorbar(image, ax=axis, label="Validation examples")
    axis.set(
        title=f"{name.replace('_', ' ').title()} validation confusion matrix",
        xlabel="Predicted digit",
        ylabel="True digit",
        xticks=np.arange(10),
        yticks=np.arange(10),
    )
    threshold = confusion.max() / 2
    for row in range(10):
        for column in range(10):
            axis.text(
                column,
                row,
                str(confusion[row, column]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if confusion[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{name}_confusion_matrix.png", dpi=160)
    plt.close(figure)

    with (OUTPUT_DIR / f"{name}_output_activation_by_predicted_digit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = ["predicted_digit", "output_neuron", "count", "mean", "std", "p05", "median", "p95"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for predicted_digit in range(10):
            subset = outputs[predictions == predicted_digit]
            for neuron in range(10):
                row = {"predicted_digit": predicted_digit, "output_neuron": neuron, "count": len(subset)}
                row.update(
                    {
                        key: value
                        for key, value in summarize(subset[:, neuron]).items()
                        if key in {"mean", "std", "p05", "median", "p95"}
                    }
                    if len(subset)
                    else {key: float("nan") for key in ("mean", "std", "p05", "median", "p95")}
                )
                writer.writerow(row)

    derivatives = sigmoid_prime(preactivations)
    neuron_diagnostics: dict[str, object] = {}
    for neuron in range(10):
        for subset_name, mask in (
            ("all_validation", np.ones(len(labels), dtype=bool)),
            ("true_digit_4", labels == 4),
        ):
            neuron_diagnostics[f"neuron_{neuron}_{subset_name}"] = {
                "preactivation": summarize(preactivations[mask, neuron]),
                "sigmoid_derivative": summarize(derivatives[mask, neuron]),
                "derivative_below_1e-5_fraction": float(
                    np.mean(derivatives[mask, neuron] < 1e-5)
                ),
            }
    (OUTPUT_DIR / f"{name}_output_neuron_diagnostics.json").write_text(
        json.dumps(neuron_diagnostics, indent=2) + "\n", encoding="utf-8"
    )

    result: dict[str, object] = {
        "validation_accuracy": float(np.mean(predictions == labels)),
        "validation_correct": int(np.count_nonzero(predictions == labels)),
        "validation_examples": int(len(labels)),
        "class_metrics": class_metrics,
        "confusion_matrix": confusion.tolist(),
        "digit_4_recall": float(recall[4]),
        "digit_4_true_count": int(support[4]),
        "digit_4_predicted_count": int(predicted_counts[4]),
        "output_neuron_diagnostics": neuron_diagnostics,
    }
    return result, predictions, outputs


def save_checkpoint(network: Network, path: Path) -> None:
    """Store the experimental parameters and initialization metadata separately."""
    arrays: dict[str, np.ndarray] = {
        "sizes": np.asarray(network.sizes, dtype=np.int64),
        "seed": np.asarray(SEED, dtype=np.int64),
        "initialization": np.asarray(network.initialization),
    }
    for layer, (weight, bias) in enumerate(zip(network.weights, network.biases)):
        arrays[f"weight_{layer}"] = weight
        arrays[f"bias_{layer}"] = bias
    np.savez_compressed(path, **arrays)


def main() -> None:
    if not BASELINE_MODEL.exists() or not BASELINE_METRICS.exists():
        raise SystemExit("Baseline checkpoint and metrics are required under outputs/")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # load_data returns all three splits; intentionally bind and use only train/validation.
    training_data, validation_data, _held_out = load_data()
    train_images, train_labels = training_data
    validation_images, validation_labels = validation_data
    if len(train_labels) != 50_000 or len(validation_labels) != 10_000:
        raise SystemExit("Expected the existing 50,000/10,000 training/validation splits")

    baseline = restore_baseline(BASELINE_MODEL)
    baseline_metrics, _, _ = evaluate_validation(
        baseline, validation_images, validation_labels, "standard_normal"
    )
    with BASELINE_METRICS.open(newline="", encoding="utf-8") as file:
        baseline_history = list(csv.DictReader(file))
    baseline_costs = [float(row["training_cost"]) for row in baseline_history]
    baseline_accuracy_history = [float(row["validation_accuracy"]) for row in baseline_history]

    glorot = Network(ARCHITECTURE, seed=SEED, initialization="glorot_normal")
    print(
        f"Glorot-normal experiment: architecture={ARCHITECTURE}, train={len(train_labels)}, "
        f"validation={len(validation_labels)}, epochs={EPOCHS}, batch={MINI_BATCH_SIZE}, "
        f"learning_rate={LEARNING_RATE}, seed={SEED}. Test split is not evaluated.",
        flush=True,
    )
    started = time.perf_counter()
    glorot_history = glorot.stochastic_gradient_descent(
        training_data,
        epochs=EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        evaluation_data=validation_data,
        seed=SEED,
    )
    training_seconds = time.perf_counter() - started
    print(f"Glorot training and epoch evaluation time: {training_seconds:.2f}s", flush=True)

    with (OUTPUT_DIR / "glorot_training_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(glorot_history[0]))
        writer.writeheader()
        writer.writerows(glorot_history)

    glorot_metrics, _, _ = evaluate_validation(
        glorot, validation_images, validation_labels, "glorot_normal"
    )
    comparison_metrics = {
        "experiment": {
            "architecture": ARCHITECTURE,
            "training_examples": len(train_labels),
            "validation_examples": len(validation_labels),
            "epochs": EPOCHS,
            "mini_batch_size": MINI_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "test_split_evaluated": False,
            "glorot_training_seconds_including_epoch_metrics": training_seconds,
        },
        "standard_normal_baseline": {
            **baseline_metrics,
            "training_cost_by_epoch": baseline_costs,
            "validation_accuracy_by_epoch": baseline_accuracy_history,
            "checkpoint": str(BASELINE_MODEL.relative_to(ROOT)),
        },
        "glorot_normal": {
            **glorot_metrics,
            "training_cost_by_epoch": [float(row["training_cost"]) for row in glorot_history],
            "validation_accuracy_by_epoch": [float(row["validation_accuracy"]) for row in glorot_history],
            "training_seconds": training_seconds,
            "checkpoint": "outputs/glorot_comparison/glorot_network.npz",
        },
    }
    (OUTPUT_DIR / "initialization_comparison.json").write_text(
        json.dumps(comparison_metrics, indent=2) + "\n", encoding="utf-8"
    )
    save_checkpoint(glorot, OUTPUT_DIR / "glorot_network.npz")

    epochs = np.arange(1, EPOCHS + 1)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(epochs, baseline_costs, marker="o", label="Standard normal")
    axis.plot(epochs, [row["training_cost"] for row in glorot_history], marker="o", label="Glorot normal")
    axis.set(title="Training cost by initialization", xlabel="Epoch", ylabel="Mean quadratic cost")
    axis.set_xticks(epochs)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_cost_comparison.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(epochs, np.asarray(baseline_accuracy_history) * 100, marker="o", label="Standard normal")
    axis.plot(epochs, [float(row["validation_accuracy"]) * 100 for row in glorot_history], marker="o", label="Glorot normal")
    axis.set(title="Validation accuracy by initialization", xlabel="Epoch", ylabel="Accuracy (%)")
    axis.set_xticks(epochs)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "validation_accuracy_comparison.png", dpi=160)
    plt.close(figure)

    print(f"Standard-normal validation accuracy: {baseline_metrics['validation_accuracy']:.2%}")
    print(f"Glorot-normal validation accuracy: {glorot_metrics['validation_accuracy']:.2%}")
    print(f"Digit-4 validation recall: standard={baseline_metrics['digit_4_recall']:.4%}; "
          f"Glorot={glorot_metrics['digit_4_recall']:.4%}")
    print(f"Comparison artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
