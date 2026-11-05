"""Full-return REINFORCE derivatives at a frozen collecting policy."""

from collections.abc import Sequence
from math import isfinite

import torch
from torch import Tensor

from common.returns import discounted_returns


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
    if not parameters:
        raise ValueError("supply at least one parameter")
    if (log_probabilities.ndim != 1 or log_probabilities.numel() != len(rewards)
            or log_probabilities.numel() == 0):
        raise ValueError("supply one scalar log-probability per reward in a nonempty episode")
    full_return = discounted_returns(rewards, gamma)[0]
    if not isfinite(full_return) or not torch.isfinite(log_probabilities).all():
        raise ValueError("episode returns and selected action log-probabilities must be finite")

    surrogate = full_return * log_probabilities.sum()
    derivatives = torch.autograd.grad(surrogate, tuple(parameters), materialize_grads=True)
    return tuple(derivative.detach() for derivative in derivatives)
