"""Measure repeated vector estimates against a known reference gradient."""

import numpy as np
from numpy.typing import ArrayLike


def summarize_gradients(estimates: ArrayLike, reference: ArrayLike) -> dict:
    """Summarize B independent batch estimates (rows) in a fixed D-logit basis.

    For x_b in R^D, mean m, and exact gradient g:
      squared empirical bias = ||m - g||^2
      empirical variance trace = sum_b ||x_b - m||^2 / B
      MSE = sum_b ||x_b - g||^2 / B
          = squared empirical bias + empirical variance trace.

    Trace means the SUM of coordinate variances, without division by D. It
    measures squared Euclidean vector error in these parameter coordinates;
    changing the parameterization can change this measure. No covariance matrix
    is needed. x_b must be a mean of whole episode gradient sums, not transitions.

    sample_variance_trace uses B-1 and estimates the population variance of ONE
    batch estimate. With that convention the finite-sample identity is instead
    MSE = squared empirical bias + (B-1)/B * sample_variance_trace.

    Finite B makes m noisy: E[||m-g||^2] = ||E[x]-g||^2 + tr(Cov(x))/B.
    mean_noise_floor estimates the latter contribution with sample variance/B;
    mean_standard_error gives coordinatewise standard errors of m. Neither is
    a confidence interval or proof of bias. Independence is a caller precondition
    (conditional on any separately fitted, frozen baseline).

    Use float64 NumPy at the reporting boundary; return JSON-compatible values.
    Require at least two finite, nonempty vectors so sample variances exist.
    """
    estimates = np.asarray(estimates, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if (estimates.ndim != 2 or estimates.shape[0] < 2 or estimates.shape[1] == 0
            or reference.shape != estimates.shape[1:]):
        raise ValueError("supply B >= 2 vectors of D > 0 coordinates and a matching reference")
    if not np.isfinite(estimates).all() or not np.isfinite(reference).all():
        raise ValueError("estimates and reference must be finite")

    batches = len(estimates)
    mean = estimates.mean(axis=0)
    bias = mean - reference
    centered_squares = np.square(estimates - mean).sum(axis=0)
    sample_variances = centered_squares / (batches - 1)
    return {
        "mean": mean.tolist(),
        "empirical_bias": bias.tolist(),
        "squared_empirical_bias": float(np.dot(bias, bias)),
        "empirical_variance_trace": float(centered_squares.sum() / batches),
        "sample_variance_trace": float(sample_variances.sum()),
        "mse": float(np.square(estimates - reference).sum(axis=1).mean()),
        "mean_standard_error": np.sqrt(sample_variances / batches).tolist(),
        "mean_noise_floor": float(sample_variances.sum() / batches),
    }
