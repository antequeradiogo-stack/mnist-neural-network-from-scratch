"""Tests for neural network initialization, forward propagation, cost, and backprop."""

import unittest

import numpy as np

from src.network import Network, quadratic_cost, sigmoid, sigmoid_prime


class SigmoidTests(unittest.TestCase):
    def test_sigmoid_for_scalar_and_array_values(self) -> None:
        self.assertAlmostEqual(float(sigmoid(0.0)), 0.5)
        values = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        np.testing.assert_allclose(values, [0.0, 0.5, 1.0], atol=1e-15)

    def test_sigmoid_derivative_at_zero(self) -> None:
        self.assertAlmostEqual(float(sigmoid_prime(np.array([0.0]))[0]), 0.25)


class QuadraticCostTests(unittest.TestCase):
    def test_perfect_prediction_has_zero_cost(self) -> None:
        target = np.array([[0.0], [1.0], [0.0]])
        self.assertEqual(quadratic_cost(target, target), 0.0)

    def test_known_prediction_has_manually_computed_cost(self) -> None:
        activations = np.array([[0.8], [0.1]])
        target = np.array([[1.0], [0.0]])
        self.assertAlmostEqual(quadratic_cost(activations, target), 0.025)

    def test_rejects_incompatible_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            quadratic_cost(np.zeros((10, 1)), np.zeros((10,)))


class NetworkInitializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.network = Network([784, 30, 10], seed=2026)

    def test_stores_layer_sizes_and_count(self) -> None:
        self.assertEqual(self.network.sizes, [784, 30, 10])
        self.assertEqual(self.network.num_layers, 3)

    def test_creates_one_weight_matrix_and_bias_vector_per_non_input_layer(self) -> None:
        self.assertEqual(len(self.network.weights), 2)
        self.assertEqual(len(self.network.biases), 2)

    def test_parameter_shapes_follow_layer_sizes(self) -> None:
        self.assertEqual(self.network.weights[0].shape, (30, 784))
        self.assertEqual(self.network.weights[1].shape, (10, 30))
        self.assertEqual(self.network.biases[0].shape, (30, 1))
        self.assertEqual(self.network.biases[1].shape, (10, 1))

    def test_initial_parameters_are_finite(self) -> None:
        for parameter in self.network.weights + self.network.biases:
            self.assertTrue(np.isfinite(parameter).all())

    def test_seed_reproduces_initial_parameters(self) -> None:
        same_initialization = Network([784, 30, 10], seed=2026)
        for first, second in zip(self.network.weights, same_initialization.weights):
            np.testing.assert_array_equal(first, second)
        for first, second in zip(self.network.biases, same_initialization.biases):
            np.testing.assert_array_equal(first, second)

    def test_glorot_normal_is_the_default(self) -> None:
        default = Network([4, 3, 2], seed=91)
        explicit = Network([4, 3, 2], seed=91, initialization="glorot_normal")
        self.assertEqual(default.initialization, "glorot_normal")
        for actual, expected in zip(
            default.weights + default.biases, explicit.weights + explicit.biases
        ):
            np.testing.assert_array_equal(actual, expected)
        for bias in default.biases:
            np.testing.assert_array_equal(bias, np.zeros_like(bias))

    def test_explicit_standard_normal_is_reproducible_and_finite(self) -> None:
        first = Network([4, 3, 2], seed=91, initialization="standard_normal")
        second = Network([4, 3, 2], seed=91, initialization="standard_normal")
        self.assertEqual(first.initialization, "standard_normal")
        self.assertEqual([w.shape for w in first.weights], [(3, 4), (2, 3)])
        self.assertEqual([b.shape for b in first.biases], [(3, 1), (2, 1)])
        for actual, expected in zip(first.weights + first.biases, second.weights + second.biases):
            np.testing.assert_array_equal(actual, expected)
        self.assertTrue(
            all(np.isfinite(parameter).all() for parameter in first.weights + first.biases)
        )

    def test_glorot_normal_has_zero_biases_and_reproducible_weight_scales(self) -> None:
        network = Network([4, 3, 2], seed=13, initialization="glorot_normal")
        rng = np.random.default_rng(13)
        expected_weights = [
            rng.normal(0.0, np.sqrt(2.0 / 7.0), size=(3, 4)),
            rng.normal(0.0, np.sqrt(2.0 / 5.0), size=(2, 3)),
        ]
        for actual, expected in zip(network.weights, expected_weights):
            np.testing.assert_array_equal(actual, expected)
        for bias in network.biases:
            np.testing.assert_array_equal(bias, np.zeros_like(bias))
        self.assertEqual([w.shape for w in network.weights], [(3, 4), (2, 3)])
        self.assertEqual([b.shape for b in network.biases], [(3, 1), (2, 1)])
        self.assertTrue(
            all(np.isfinite(parameter).all() for parameter in network.weights + network.biases)
        )

    def test_rejects_unknown_initialization(self) -> None:
        with self.assertRaisesRegex(ValueError, "initialization must be"):
            Network([2, 1], initialization="uniform")


class FeedforwardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.network = Network([784, 30, 10], seed=17)

    def test_output_has_expected_shape_and_finite_activations(self) -> None:
        output = self.network.feedforward(np.zeros((784, 1)))
        self.assertEqual(output.shape, (10, 1))
        self.assertTrue(np.isfinite(output).all())

    def test_rejects_invalid_input_shapes(self) -> None:
        for invalid_input in (np.zeros(784), np.zeros((783, 1)), np.zeros((784, 2))):
            with self.subTest(shape=invalid_input.shape):
                with self.assertRaisesRegex(ValueError, r"Input must have shape \(784, 1\)"):
                    self.network.feedforward(invalid_input)

    def test_small_network_forward_pass(self) -> None:
        output = Network([2, 3, 1], seed=8).feedforward(np.array([[0.25], [0.75]]))
        self.assertEqual(output.shape, (1, 1))
        self.assertTrue(np.isfinite(output).all())


class BackpropTests(unittest.TestCase):
    def test_gradient_shapes_and_finite_values(self) -> None:
        network = Network([3, 2, 2], seed=31)
        bias_gradients, weight_gradients = network.backprop(
            np.array([[0.2], [0.4], [0.8]]), np.array([[1.0], [0.0]])
        )
        self.assertEqual([g.shape for g in bias_gradients], [(2, 1), (2, 1)])
        self.assertEqual([g.shape for g in weight_gradients], [(2, 3), (2, 2)])
        for gradient in bias_gradients + weight_gradients:
            self.assertTrue(np.isfinite(gradient).all())

    def test_rejects_invalid_input_and_target_shapes(self) -> None:
        network = Network([2, 2], seed=2)
        with self.assertRaisesRegex(ValueError, "Input must have shape"):
            network.backprop(np.zeros(2), np.zeros((2, 1)))
        with self.assertRaisesRegex(ValueError, "Target must have shape"):
            network.backprop(np.zeros((2, 1)), np.zeros(2))

    def test_hand_verifiable_single_neuron_gradient(self) -> None:
        network = Network([1, 1], seed=4)
        network.weights[0][:] = 0.0
        network.biases[0][:] = 0.0
        bias_gradients, weight_gradients = network.backprop(
            np.array([[2.0]]), np.array([[1.0]])
        )
        # At z=0, a=0.5 and sigmoid'(z)=0.25. So delta=(0.5-1)*0.25=-0.125.
        # The bias gradient is delta; the weight gradient is delta*x=-0.25.
        np.testing.assert_allclose(bias_gradients[0], [[-0.125]])
        np.testing.assert_allclose(weight_gradients[0], [[-0.25]])

    def test_gradients_match_finite_differences(self) -> None:
        network = Network([2, 2, 2], seed=14)
        x = np.array([[0.3], [0.7]])
        y = np.array([[1.0], [0.0]])
        bias_gradients, weight_gradients = network.backprop(x, y)
        epsilon = 1e-5

        def cost() -> float:
            return quadratic_cost(network.feedforward(x), y)

        for layer, bias in enumerate(network.biases):
            for index in np.ndindex(bias.shape):
                original = bias[index]
                bias[index] = original + epsilon
                cost_plus = cost()
                bias[index] = original - epsilon
                cost_minus = cost()
                bias[index] = original
                numerical = (cost_plus - cost_minus) / (2 * epsilon)
                self.assertAlmostEqual(
                    bias_gradients[layer][index], numerical, delta=1e-7
                )

        for layer, weight in enumerate(network.weights):
            for index in np.ndindex(weight.shape):
                original = weight[index]
                weight[index] = original + epsilon
                cost_plus = cost()
                weight[index] = original - epsilon
                cost_minus = cost()
                weight[index] = original
                numerical = (cost_plus - cost_minus) / (2 * epsilon)
                self.assertAlmostEqual(
                    weight_gradients[layer][index], numerical, delta=1e-7
                )



class SGDTests(unittest.TestCase):
    def test_tiny_mini_batch_update_matches_manual_gradient(self) -> None:
        network = Network([1, 1], seed=3)
        network.weights[0][:] = 0.0
        network.biases[0][:] = 0.0
        network.update_mini_batch([(np.array([2.0]), np.array([1.0]))], learning_rate=0.4)
        # For x=2, target=1, z=0: nabla_b=-0.125, nabla_w=-0.25.
        np.testing.assert_allclose(network.biases[0], [[0.05]])
        np.testing.assert_allclose(network.weights[0], [[0.1]])

    def test_sgd_uses_incomplete_final_mini_batch(self) -> None:
        class RecordingNetwork(Network):
            def __init__(self) -> None:
                super().__init__([1, 2], seed=5)
                self.batch_lengths = []

            def update_mini_batch(self, mini_batch, learning_rate) -> None:
                examples = list(mini_batch)
                self.batch_lengths.append(len(examples))
                super().update_mini_batch(examples, learning_rate)

        network = RecordingNetwork()
        data = (np.arange(5, dtype=float).reshape(5, 1), np.array([0, 1, 0, 1, 0]))
        network.stochastic_gradient_descent(
            data, epochs=1, mini_batch_size=2, learning_rate=0.1, seed=9
        )
        self.assertEqual(network.batch_lengths, [2, 2, 1])

    def test_shuffle_is_reproducible_with_local_seed(self) -> None:
        data = (
            np.array([[0.0], [0.2], [0.4], [0.6], [0.8]]),
            np.array([0, 1, 0, 1, 0]),
        )
        first = Network([1, 2], seed=42)
        second = Network([1, 2], seed=42)
        first.stochastic_gradient_descent(data, 2, 2, 0.3, seed=123)
        second.stochastic_gradient_descent(data, 2, 2, 0.3, seed=123)
        for left, right in zip(first.weights + first.biases, second.weights + second.biases):
            np.testing.assert_array_equal(left, right)

    def test_evaluate_counts_correct_predictions_for_integer_and_one_hot_labels(self) -> None:
        network = Network([1, 2], seed=1)
        network.weights[0][:] = 0.0
        network.biases[0][:] = np.array([[0.0], [2.0]])
        images = np.array([[0.0], [0.5], [1.0]])
        integer_labels = np.array([1, 1, 0])
        self.assertEqual(network.evaluate((images, integer_labels)), 2)
        one_hot_labels = np.eye(2)[integer_labels]
        self.assertEqual(network.evaluate((images, one_hot_labels)), 2)

    def test_training_preserves_parameter_shapes_and_finite_values(self) -> None:
        network = Network([2, 3, 2], seed=7)
        data = (
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            np.array([0, 1, 0]),
        )
        weight_shapes = [weight.shape for weight in network.weights]
        bias_shapes = [bias.shape for bias in network.biases]
        network.stochastic_gradient_descent(data, 2, 2, 0.2, seed=8)
        self.assertEqual([weight.shape for weight in network.weights], weight_shapes)
        self.assertEqual([bias.shape for bias in network.biases], bias_shapes)
        self.assertTrue(all(np.isfinite(parameter).all() for parameter in network.weights + network.biases))

    def test_rejects_invalid_hyperparameters(self) -> None:
        network = Network([1, 2], seed=1)
        data = (np.ones((2, 1)), np.array([0, 1]))
        invalid_parameters = (
            {"epochs": 0, "mini_batch_size": 1, "learning_rate": 0.1},
            {"epochs": 1, "mini_batch_size": 0, "learning_rate": 0.1},
            {"epochs": 1, "mini_batch_size": 1, "learning_rate": 0.0},
            {"epochs": 1, "mini_batch_size": 1, "learning_rate": float("nan")},
        )
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    network.stochastic_gradient_descent(data, **parameters)




    def test_sgd_returns_epoch_training_cost_and_validation_accuracy(self) -> None:
        network = Network([1, 2], seed=12)
        training = (np.array([[0.0], [1.0]]), np.array([0, 1]))
        evaluation = (np.array([[0.25], [0.75]]), np.array([0, 1]))
        history = network.stochastic_gradient_descent(
            training, epochs=2, mini_batch_size=1, learning_rate=0.2,
            evaluation_data=evaluation, seed=13
        )
        self.assertEqual(len(history), 2)
        for record in history:
            self.assertIn("training_cost", record)
            self.assertIn("validation_accuracy", record)
            self.assertTrue(np.isfinite(record["training_cost"]))
            self.assertGreaterEqual(record["validation_accuracy"], 0.0)
            self.assertLessEqual(record["validation_accuracy"], 1.0)





if __name__ == "__main__":
    unittest.main()
