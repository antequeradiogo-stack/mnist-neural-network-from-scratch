"""Parameter initialization, forward propagation, cost, backprop, and SGD."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    """Apply the numerically stable sigmoid function elementwise."""
    return np.exp(-np.logaddexp(0, -np.asarray(z)))


def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    """Return the derivative of sigmoid, evaluated at the pre-activation ``z``."""
    activation = sigmoid(z)
    return activation * (1.0 - activation)


def quadratic_cost(activations: np.ndarray, target: np.ndarray) -> float:
    """Return the quadratic cost, 0.5 * sum((activations - target) ** 2)."""
    activations = np.asarray(activations, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if activations.shape != target.shape:
        raise ValueError(
            f"Activations and target must have the same shape; "
            f"got {activations.shape} and {target.shape}"
        )
    return float(0.5 * np.sum((activations - target) ** 2))


class Network:
    """A neural network with selectable parameter initialization."""

    def __init__(
        self,
        sizes: list[int],
        seed: int | None = None,
        initialization: str = "glorot_normal",
    ) -> None:
        """Create biases and weights for each non-input layer."""
        if len(sizes) < 2:
            raise ValueError("A network must have at least an input and an output layer")
        if any(not isinstance(size, int) or size <= 0 for size in sizes):
            raise ValueError("Every layer size must be a positive integer")

        if initialization not in {"standard_normal", "glorot_normal"}:
            raise ValueError(
                "initialization must be 'standard_normal' or 'glorot_normal'"
            )

        self.sizes = list(sizes)
        self.num_layers = len(self.sizes)
        self.initialization = initialization
        rng = np.random.default_rng(seed)
        if initialization == "standard_normal":
            # Preserve the original draw order so existing seeded models remain exact.
            self.biases = [rng.standard_normal((n, 1)) for n in self.sizes[1:]]
            self.weights = [
                rng.standard_normal((next_n, previous_n))
                for previous_n, next_n in zip(self.sizes[:-1], self.sizes[1:])
            ]
        else:
            self.biases = [np.zeros((n, 1)) for n in self.sizes[1:]]
            self.weights = [
                rng.normal(
                    0.0,
                    np.sqrt(2.0 / (previous_n + next_n)),
                    size=(next_n, previous_n),
                )
                for previous_n, next_n in zip(self.sizes[:-1], self.sizes[1:])
            ]

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """Return the output activation for one input column vector."""
        activation = np.asarray(a, dtype=np.float64)
        expected_shape = (self.sizes[0], 1)
        if activation.shape != expected_shape:
            raise ValueError(
                f"Input must have shape {expected_shape}; got {activation.shape}"
            )
        for weight, bias in zip(self.weights, self.biases):
            activation = sigmoid(weight @ activation + bias)
        return activation

    def backprop(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return per-example gradients ``(nabla_b, nabla_w)`` for quadratic cost."""
        activation = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        expected_input_shape = (self.sizes[0], 1)
        expected_target_shape = (self.sizes[-1], 1)
        if activation.shape != expected_input_shape:
            raise ValueError(
                f"Input must have shape {expected_input_shape}; got {activation.shape}"
            )
        if target.shape != expected_target_shape:
            raise ValueError(
                f"Target must have shape {expected_target_shape}; got {target.shape}"
            )

        activations = [activation]
        weighted_inputs = []
        for weight, bias in zip(self.weights, self.biases):
            weighted_input = weight @ activations[-1] + bias
            weighted_inputs.append(weighted_input)
            activations.append(sigmoid(weighted_input))

        bias_gradients = [np.empty_like(bias) for bias in self.biases]
        weight_gradients = [np.empty_like(weight) for weight in self.weights]
        delta = (activations[-1] - target) * sigmoid_prime(weighted_inputs[-1])
        bias_gradients[-1] = delta
        weight_gradients[-1] = delta @ activations[-2].T

        for layer_index in range(2, self.num_layers):
            weighted_input = weighted_inputs[-layer_index]
            delta = (self.weights[-layer_index + 1].T @ delta) * sigmoid_prime(
                weighted_input
            )
            bias_gradients[-layer_index] = delta
            weight_gradients[-layer_index] = delta @ activations[-layer_index - 1].T
        return bias_gradients, weight_gradients

    def _input_column(self, x: np.ndarray) -> np.ndarray:
        """Convert one flattened image or column vector to the network's input shape."""
        value = np.asarray(x, dtype=np.float64)
        if value.shape == (self.sizes[0],):
            value = value.reshape(self.sizes[0], 1)
        if value.shape != (self.sizes[0], 1):
            raise ValueError(
                f"Each input must have shape ({self.sizes[0]},) or "
                f"({self.sizes[0]}, 1); got {value.shape}"
            )
        return value

    def _target_column(self, label: int | np.ndarray) -> np.ndarray:
        """Convert an integer class label or one-hot label into a target column."""
        value = np.asarray(label)
        if value.ndim == 0:
            class_index = int(value)
            if class_index != value or not 0 <= class_index < self.sizes[-1]:
                raise ValueError(f"Class label must be in [0, {self.sizes[-1] - 1}]")
            target = np.zeros((self.sizes[-1], 1), dtype=np.float64)
            target[class_index, 0] = 1.0
            return target
        if value.shape == (self.sizes[-1],):
            value = value.reshape(self.sizes[-1], 1)
        if value.shape != (self.sizes[-1], 1):
            raise ValueError(
                f"Each target must be a class index or shape "
                f"({self.sizes[-1]}, 1); got {value.shape}"
            )
        return np.asarray(value, dtype=np.float64)

    def update_mini_batch(
        self, mini_batch: Iterable[tuple[np.ndarray, int | np.ndarray]], learning_rate: float
    ) -> None:
        """Apply one averaged gradient update from a mini-batch of examples."""
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number")
        examples = list(mini_batch)
        if not examples:
            raise ValueError("mini_batch must contain at least one example")

        bias_sums = [np.zeros_like(bias) for bias in self.biases]
        weight_sums = [np.zeros_like(weight) for weight in self.weights]
        for x, label in examples:
            bias_gradients, weight_gradients = self.backprop(
                self._input_column(x), self._target_column(label)
            )
            bias_sums = [total + grad for total, grad in zip(bias_sums, bias_gradients)]
            weight_sums = [total + grad for total, grad in zip(weight_sums, weight_gradients)]

        scale = learning_rate / len(examples)
        self.weights = [weight - scale * grad for weight, grad in zip(self.weights, weight_sums)]
        self.biases = [bias - scale * grad for bias, grad in zip(self.biases, bias_sums)]

    @staticmethod
    def _validate_dataset(
        data: tuple[np.ndarray, np.ndarray], input_size: int, output_size: int, name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate loader-style ``(images, integer labels)`` or one-hot data."""
        if not isinstance(data, (tuple, list)) or len(data) != 2:
            raise ValueError(f"{name} must be a pair of (images, labels)")
        images = np.asarray(data[0], dtype=np.float64)
        labels = np.asarray(data[1])
        if images.ndim != 2 or images.shape[1] != input_size:
            raise ValueError(f"{name} images must have shape (N, {input_size}); got {images.shape}")
        if images.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one example")
        if labels.shape not in ((images.shape[0],), (images.shape[0], output_size)):
            raise ValueError(
                f"{name} labels must have shape ({images.shape[0]},) or "
                f"({images.shape[0]}, {output_size}); got {labels.shape}"
            )
        if not np.isfinite(images).all():
            raise ValueError(f"{name} images must contain only finite values")
        return images, labels

    def evaluate(self, evaluation_data: tuple[np.ndarray, np.ndarray]) -> int:
        """Return the number of correctly classified examples in a data split."""
        images, labels = self._validate_dataset(
            evaluation_data, self.sizes[0], self.sizes[-1], "evaluation_data"
        )
        expected_labels = np.argmax(labels, axis=1) if labels.ndim == 2 else labels
        if labels.ndim == 1 and (
            not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0)
            or np.any(labels >= self.sizes[-1])
        ):
            raise ValueError(f"Evaluation class labels must be integers in [0, {self.sizes[-1] - 1}]")
        correct = 0
        for image, label in zip(images, expected_labels):
            prediction = int(np.argmax(self.feedforward(image.reshape(-1, 1))))
            correct += prediction == int(label)
        return int(correct)

    def stochastic_gradient_descent(
        self,
        training_data: tuple[np.ndarray, np.ndarray],
        epochs: int,
        mini_batch_size: int,
        learning_rate: float,
        evaluation_data: tuple[np.ndarray, np.ndarray] | None = None,
        seed: int | None = None,
    ) -> list[dict[str, float | int]]:
        """Train with shuffled mini-batches and return one metrics record per epoch.

        ``training_cost`` is the mean quadratic cost across the training examples
        after that epoch's updates. When supplied, ``validation_accuracy`` is
        the fraction correct on the evaluation data.
        """
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if not isinstance(mini_batch_size, int) or mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be a positive integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number")
        images, labels = self._validate_dataset(
            training_data, self.sizes[0], self.sizes[-1], "training_data"
        )
        if labels.ndim == 1 and (
            not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0)
            or np.any(labels >= self.sizes[-1])
        ):
            raise ValueError(f"Training class labels must be integers in [0, {self.sizes[-1] - 1}]")
        if evaluation_data is not None:
            evaluation_data = self._validate_dataset(
                evaluation_data, self.sizes[0], self.sizes[-1], "evaluation_data"
            )

        rng = np.random.default_rng(seed)
        example_count = images.shape[0]
        history: list[dict[str, float | int]] = []
        for epoch in range(1, epochs + 1):
            shuffled_indices = rng.permutation(example_count)
            for start in range(0, example_count, mini_batch_size):
                indices = shuffled_indices[start : start + mini_batch_size]
                mini_batch = [(images[i], labels[i]) for i in indices]
                self.update_mini_batch(mini_batch, learning_rate)

            total_cost = 0.0
            for image, label in zip(images, labels):
                prediction = self.feedforward(image.reshape(-1, 1))
                target = self._target_column(label)
                total_cost += quadratic_cost(prediction, target)
            epoch_metrics: dict[str, float | int] = {
                "epoch": epoch,
                "training_cost": total_cost / example_count,
            }

            if evaluation_data is None:
                print(
                    f"Epoch {epoch}/{epochs}: training cost "
                    f"{epoch_metrics['training_cost']:.6f}"
                )
            else:
                evaluation_count = evaluation_data[0].shape[0]
                correct = self.evaluate(evaluation_data)
                percentage = 100.0 * correct / evaluation_count
                epoch_metrics["validation_correct"] = correct
                epoch_metrics["validation_accuracy"] = correct / evaluation_count
                print(
                    f"Epoch {epoch}/{epochs}: training cost "
                    f"{epoch_metrics['training_cost']:.6f}; validation accuracy "
                    f"{correct}/{evaluation_count} ({percentage:.1f}%)"
                )
            history.append(epoch_metrics)
        return history
