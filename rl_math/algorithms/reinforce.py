"""REINFORCE at a frozen collecting policy, with optional state baselines."""

from collections.abc import Sequence
from math import isfinite

import torch
from torch import Tensor

from common.returns import discounted_returns, reward_to_go_weights


def _validate_episode(log_probabilities: Tensor, rewards: Sequence[float], parameters: Sequence[Tensor]) -> None:
    if not parameters:
        raise ValueError("supply at least one parameter")
    if (log_probabilities.ndim != 1 or log_probabilities.numel() != len(rewards)
            or log_probabilities.numel() == 0):
        raise ValueError("supply one scalar log-probability per reward in a nonempty episode")
    if not torch.isfinite(log_probabilities).all():
        raise ValueError("episode returns and selected action log-probabilities must be finite")


def full_return_gradient(
    log_probabilities: Tensor,
    rewards: Sequence[float],
    parameters: Sequence[Tensor],
    gamma: float = 1.0,
) -> tuple[Tensor, ...]:
    """Compute one episode's REINFORCE sum G_0 * sum_t grad log pi(a_t | s_t).

    J = E[G_0], with G_0 = sum_t gamma**t * r_{t+1}. Every action receives
    the same full return; no additional gamma**t multiplies its score.
    Average these gradient sums over episodes to estimate dJ/dtheta, regardless
    of their lengths. There is no reward-to-go, baseline, or normalization.

    Supply a complete, nonempty episode's 1D tensor of selected action
    log-probabilities, with its graph under the unchanged collecting policy.
    rewards[t] is the recorded r_{t+1}. Autograd differentiates G_0 * sum(log pi),
    treating observed rewards as constants. This scalar is a gradient surrogate,
    not an estimate of J. The function is independent of environment types.
    Return detached derivatives in parameter order, with zeros for unused
    parameters. Do not update parameters or accumulate their .grad buffers.
    Each call consumes the supplied log-probability graphs: evaluate fresh
    log-probabilities for another probe. Parameters must require gradients.
    """
    _validate_episode(log_probabilities, rewards, parameters)
    full_return = discounted_returns(rewards, gamma)[0]
    if not isfinite(full_return):
        raise ValueError("episode returns and selected action log-probabilities must be finite")

    surrogate = full_return * log_probabilities.sum()
    derivatives = torch.autograd.grad(surrogate, tuple(parameters), materialize_grads=True)
    return tuple(derivative.detach() for derivative in derivatives)


def reward_to_go_gradient(
    log_probabilities: Tensor,
    rewards: Sequence[float],
    parameters: Sequence[Tensor],
    gamma: float = 1.0,
    *,
    baselines: Tensor | Sequence[float] | None = None,
) -> tuple[Tensor, ...]:
    """Sum gamma**t * (G_t - b(s_t)) * grad log pi(a_t | s_t) for an episode.

    G_t = sum_{k=t}^{T-1} gamma**(k-t) * r_{k+1}. Its inner discount is
    relative to action t; the outer gamma**t discounts back to episode start
    for J = E[G_0]. Here t starts at zero for each complete episode.

    G_0 = past discounted rewards + gamma**t * G_t. Conditioned on the
    history before action t, past rewards are fixed and the expected score
    over that action is zero. Removing that past-reward score contribution
    therefore preserves the expected gradient, but changes individual samples.
    Without a baseline and with only a terminal reward, gamma**t * G_t equals
    G_0 for every action, so the two sampled estimators coincide.

    baselines optionally supplies one finite scalar b(s_t) per selected action,
    in episode order. Omit it for plain reward-to-go. The caller must use a
    fixed state-only function, independent of the sampled action and subsequent
    returns; a baseline fitted on these same episodes need not satisfy this.
    Include remaining time in a finite-horizon state. Exact V and imperfect
    state-only baselines both preserve the expectation because the action's
    conditional mean score is zero. Their gradient variances can differ; exact
    V need not minimize variance. G_t - V(s_t) is a sampled advantage, with
    conditional mean Q(s_t,a_t) - V(s_t), not generally the exact advantage.

    Baseline values are detached even if computed from actor parameters, then
    placed on the log-probability device/dtype. Only log-probabilities are
    differentiated: gamma**t multiplies both G_t and b(s_t). No derivative of
    the baseline or return target belongs in this score-function estimator.

    Inputs and graph ownership follow full_return_gradient: a complete,
    nonempty episode under the unchanged collecting policy, recorded rewards
    held constant, and fresh selected log-probabilities for each call.
    Return detached ascent gradients in parameter order, including zeros for
    unused parameters, without changing parameters or their .grad buffers.
    Average episode sums, not transitions. No normalization or bootstrapping.
    """
    _validate_episode(log_probabilities, rewards, parameters)
    # Recorded rewards have no gradient. Copy weights onto the log-probability
    # device/dtype, retaining the policy graph only through log_probabilities.
    weights = log_probabilities.new_tensor(reward_to_go_weights(rewards, gamma))
    if baselines is not None:
        baseline_values = torch.as_tensor(baselines, dtype=log_probabilities.dtype,
                                          device=log_probabilities.device).detach()
        if baseline_values.shape != log_probabilities.shape or not torch.isfinite(baseline_values).all():
            raise ValueError("supply one finite scalar baseline per selected action")
        times = torch.arange(len(rewards), device=log_probabilities.device, dtype=log_probabilities.dtype)
        weights = weights - gamma**times * baseline_values
    if not torch.isfinite(weights).all():
        raise ValueError("episode return/baseline weights and selected action log-probabilities must be finite")

    surrogate = torch.dot(weights, log_probabilities)
    derivatives = torch.autograd.grad(surrogate, tuple(parameters), materialize_grads=True)
    return tuple(derivative.detach() for derivative in derivatives)
