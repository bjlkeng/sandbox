"""Vector errors, covariance traces, and finite-sample normalization."""

import unittest

import numpy as np

from common.gradient_metrics import summarize_gradients


class GradientMetricsTests(unittest.TestCase):
    def test_hand_calculated_vector_decomposition(self):
        estimates = np.array([[1., 2.], [3., 0.], [5., 4.]])
        reference = np.array([2., 1.])
        result = summarize_gradients(estimates, reference)
        np.testing.assert_array_equal(result["mean"], [3., 2.])
        np.testing.assert_array_equal(result["empirical_bias"], [1., 1.])
        self.assertEqual(result["squared_empirical_bias"], 2.)
        # Centered squared lengths are 4, 4, 8; reference errors are 2, 2, 18.
        self.assertAlmostEqual(result["empirical_variance_trace"], 16 / 3)
        self.assertEqual(result["sample_variance_trace"], 8.)
        self.assertAlmostEqual(result["mse"], 22 / 3)
        self.assertAlmostEqual(result["mse"], result["squared_empirical_bias"]
                               + result["empirical_variance_trace"])
        self.assertAlmostEqual(result["mse"], result["squared_empirical_bias"]
                               + (2 / 3) * result["sample_variance_trace"])
        self.assertAlmostEqual(result["mean_noise_floor"], 8 / 3)
        np.testing.assert_allclose(result["mean_standard_error"], np.sqrt([4 / 3, 4 / 3]))
        np.testing.assert_array_equal(estimates, [[1., 2.], [3., 0.], [5., 4.]])
        np.testing.assert_array_equal(reference, [2., 1.])

    def test_empirical_bias_and_variation_are_distinct(self):
        centered = summarize_gradients([[0., 2.], [2., 0.]], [1., 1.])
        self.assertEqual(centered["squared_empirical_bias"], 0.)
        self.assertEqual(centered["mse"], 2.)
        constant = summarize_gradients([[3., 4.]] * 3, [0., 0.])
        self.assertEqual(constant["empirical_variance_trace"], 0.)
        self.assertEqual(constant["mse"], 25.)
        self.assertEqual(constant["mean_noise_floor"], 0.)

    def test_require_repeated_finite_vectors_and_matching_reference(self):
        for estimates, reference in (
            ([], [1.]), ([[1., 2.]], [1., 2.]), ([1., 2.], [1.]),
            ([[], []], []), ([[1.], [2.]], [[1.]]), ([[1.], [2.]], [1., 2.]),
            ([[np.nan], [2.]], [1.]), ([[1.], [np.inf]], [1.]),
            ([[1.], [2.]], [np.nan]),
        ):
            with self.subTest(estimates=estimates, reference=reference):
                with self.assertRaises(ValueError):
                    summarize_gradients(estimates, reference)


if __name__ == "__main__":
    unittest.main()
