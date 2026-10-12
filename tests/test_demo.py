"""Focused tests for the Streamlit demo's image preprocessing and inference."""

from pathlib import Path
import unittest

import numpy as np

from evaluate_candidate import load_checkpoint
from src.demo_utils import (
    predict_digit,
    preprocess_canvas_image,
    validate_candidate_network,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "mnist_glorot_candidate.npz"


class DrawingPreprocessingTests(unittest.TestCase):
    def test_blank_black_canvas_becomes_zero_input(self) -> None:
        canvas = np.zeros((320, 320, 4), dtype=np.uint8)
        canvas[:, :, 3] = 255

        result = preprocess_canvas_image(canvas)

        self.assertEqual(result.shape, (784, 1))
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_equal(result, np.zeros((784, 1)))

    def test_white_digit_is_normalized_and_has_expected_dimensions(self) -> None:
        canvas = np.zeros((320, 320, 4), dtype=np.uint8)
        canvas[:, :, 3] = 255
        canvas[70:250, 145:175, :3] = 255  # A simple vertical stroke.

        result = preprocess_canvas_image(canvas)

        self.assertEqual(result.shape, (784, 1))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(float(result.max()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)
        self.assertGreaterEqual(float(result.min()), 0.0)

    def test_rejects_invalid_canvas_shapes_and_pixel_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            preprocess_canvas_image(np.zeros((28, 28)))
        invalid = np.zeros((28, 28, 3), dtype=np.float64)
        invalid[0, 0, 0] = 300
        with self.assertRaisesRegex(ValueError, "range"):
            preprocess_canvas_image(invalid)


class CandidatePredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.network = load_checkpoint(CHECKPOINT)

    def test_saved_checkpoint_has_expected_shapes_and_predicts_ten_scores(self) -> None:
        validate_candidate_network(self.network)
        predicted_digit, scores = predict_digit(self.network, np.zeros((784, 1)))

        self.assertIn(predicted_digit, range(10))
        self.assertEqual(scores.shape, (10, 1))
        self.assertTrue(np.isfinite(scores).all())

    def test_rejects_invalid_network_input_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            predict_digit(self.network, np.zeros((28, 28)))

    def test_rejects_incompatible_checkpoint_architecture(self) -> None:
        from src.network import Network

        incompatible = Network([784, 16, 10], seed=2)
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_candidate_network(incompatible)


if __name__ == "__main__":
    unittest.main()
