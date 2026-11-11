"""Measure repeated vector estimates against a known reference gradient."""

from collections.abc import Mapping

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


def summarize_comparison(
    estimates: Mapping[str, ArrayLike],
    reference: ArrayLike,
    *,
    comparison: str,
    bootstrap_seed: int,
    resamples: int = 2000,
) -> dict:
    """Add paired whole-batch percentile intervals to the existing point metrics.

    Each method supplies B-by-D vectors from the SAME B independent batches.
    Draw B row indices with replacement for each bootstrap replicate, keeping
    all coordinates and all methods together. Recompute signed mean error,
    sample covariance trace (B-1), and MSE. Their 2.5/97.5 percentiles give
    approximate pointwise 95% intervals, conditional on the frozen policy/fit.

    For MSE differences, subtract the comparison method in EACH replicate.
    Independent method resampling would lose the covariance from shared data.
    Negative difference means lower MSE than comparison. The reference gradient
    is known, so it is neither reestimated nor resampled.

    Do not give squared empirical bias a naive percentile interval: its norm
    folds sampling noise upward even at zero true bias. Keep that point metric
    and its mean_noise_floor descriptive; report signed coordinate intervals.
    These are not simultaneous intervals across coordinates/settings, do not
    include baseline-fitting uncertainty, and have no guaranteed finite-B
    coverage. More resamples refine quantiles, not the amount of observed data.
    """
    if not estimates or comparison not in estimates:
        raise ValueError("supply paired estimates including the comparison method")
    if type(resamples) is not int or resamples < 2:
        raise ValueError("resamples must be an integer >= 2")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in estimates.items()}
    metrics = {name: summarize_gradients(values, reference) for name, values in arrays.items()}
    shape = arrays[comparison].shape
    if any(values.shape != shape for values in arrays.values()):
        raise ValueError("paired methods must have the same B-by-D shape and row alignment")
    batches = shape[0]
    reference = np.asarray(reference, dtype=np.float64)
    indices = np.random.default_rng(bootstrap_seed).integers(batches, size=(resamples, batches))

    def interval(values):
        return np.quantile(values, [.025, .975], axis=0, method="linear").tolist()

    results, mse_replicates = {}, {}
    for name, values in arrays.items():
        sampled = values[indices]  # (resample, whole batch vector, coordinate)
        means = sampled.mean(axis=1)
        traces = np.square(sampled - means[:, None, :]).sum(axis=(1, 2)) / (batches - 1)
        mse_replicates[name] = np.square(sampled - reference).sum(axis=2).mean(axis=1)
        results[name] = {"metrics": metrics[name], "intervals": {
            "empirical_bias": interval(means - reference),
            "sample_variance_trace": interval(traces),
            "mse": interval(mse_replicates[name]),
        }}
    for name in results:
        results[name]["mse_difference"] = {
            "estimate": metrics[name]["mse"] - metrics[comparison]["mse"],
            "interval": interval(mse_replicates[name] - mse_replicates[comparison]),
        }
    return {
        "bootstrap": {"method": "paired whole-batch percentile", "confidence_level": .95,
                      "resamples": resamples, "seed": bootstrap_seed, "comparison": comparison,
                      "scope": "approximate pointwise intervals, conditional on the frozen fit"},
        "estimators": results,
    }
