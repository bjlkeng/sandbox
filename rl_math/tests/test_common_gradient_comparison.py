"""Whole-vector bootstrap and paired squared-error comparisons."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from common.gradient_metrics import summarize_comparison


class GradientComparisonTests(unittest.TestCase):
    def test_all_four_resamples_of_two_known_vectors(self):
        # Every length-two resample of two observations, with replacement.
        rng = Mock()
        rng.integers.return_value = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        with patch("common.gradient_metrics.np.random.default_rng", return_value=rng):
            result = summarize_comparison({"a": [[0., 0.], [2., 4.]]}, [0., 0.],
                                          comparison="a", bootstrap_seed=7, resamples=4)
        a = result["estimators"]["a"]
        self.assertEqual(a["metrics"]["mse"], 10.)
        self.assertEqual(a["metrics"]["sample_variance_trace"], 10.)
        # Resampled MSEs are (0,10,10,20); sample traces are (0,10,10,0).
        np.testing.assert_allclose(a["intervals"]["mse"], [.75, 19.25])
        np.testing.assert_allclose(a["intervals"]["sample_variance_trace"], [0., 10.])
        np.testing.assert_allclose(a["intervals"]["empirical_bias"], [[.075, .15], [1.925, 3.85]])
        self.assertNotIn("squared_empirical_bias", a["intervals"])
        self.assertEqual(a["mse_difference"]["interval"], [0., 0.])

    def test_pairing_preserves_constant_error_difference_and_vector_coordinates(self):
        base = [[0., 0.], [2., 0.], [4., 0.]]
        shifted = [[0., 1.], [2., 1.], [4., 1.]]
        result = summarize_comparison({"base": base, "shifted": shifted, "same": base}, [0., 0.],
                                      comparison="base", bootstrap_seed=123, resamples=100)
        # Each paired squared error differs by exactly 1, despite large variation
        # in individual squared errors. Independent method resampling breaks this.
        shifted_result = result["estimators"]["shifted"]
        self.assertAlmostEqual(shifted_result["mse_difference"]["estimate"], 1.)
        np.testing.assert_allclose(shifted_result["mse_difference"]["interval"], [1., 1.], atol=1e-14)
        self.assertEqual(result["estimators"]["same"]["mse_difference"]["interval"], [0., 0.])
        self.assertEqual(shifted_result["intervals"]["empirical_bias"][0][1], 1.)
        self.assertEqual(shifted_result["intervals"]["empirical_bias"][1][1], 1.)
        replay = summarize_comparison({"base": base, "shifted": shifted, "same": base}, [0., 0.],
                                      comparison="base", bootstrap_seed=123, resamples=100)
        self.assertEqual(result, replay)
        self.assertEqual(base, [[0., 0.], [2., 0.], [4., 0.]])

    def test_invalid_pairing_or_resampling_settings(self):
        valid = {"a": [[0., 0.], [1., 1.]]}
        for data, comparison in (({}, "a"), (valid, "missing"),
                                 ({**valid, "b": [[1., 1.]] * 3}, "a"),
                                 ({"a": [[np.nan], [1.]]}, "a")):
            with self.subTest(data=data), self.assertRaises(ValueError):
                summarize_comparison(data, [0., 0.], comparison=comparison, bootstrap_seed=1)
        for kwargs in ({"resamples": 1}, {"resamples": True}, {"bootstrap_seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                summarize_comparison(valid, [0., 0.], comparison="a", **({"bootstrap_seed": 1} | kwargs))


if __name__ == "__main__":
    unittest.main()
