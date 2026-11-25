"""Frozen-policy actor gradients using fixed, bootstrapped critic predictions."""

from collections.abc import Sequence

import torch
from torch import Tensor

from algorithms.advantages import one_step_td_advantages
from algorithms.reinforce import _validate_episode


def one_step_td_gradient(
    log_probabilities: Tensor,
    rewards: Tensor | Sequence[float],
    parameters: Sequence[Tensor],
    gamma: float = 1.0,
    *,
    values: Tensor | Sequence[float],
    next_values: Tensor | Sequence[float],
    terminated: Tensor | Sequence[bool],
) -> tuple[Tensor, ...]:
    """Sum gamma**t * delta_t * grad log pi(a_t | s_t) for one episode.

    delta_t is the detached one-step TD advantage from one_step_td_advantages.
    The outer gamma**t targets J = E[G_0]; t starts at episode start. Only the
    selected action log-probabilities are differentiated. Even exact V may
    depend on actor parameters, but its derivative is not part of this score
    estimator. Detachment does not remove bias from an imperfect bootstrap.

    Supply a complete nonempty episode under the unchanged collecting policy,
    fresh log-probability graphs for each call, and a fixed critic function
    chosen independently of probe data. Its next-state prediction can still
    depend on the sampled action through that action's outcome. Terminal
    semantics are defined by the target helper; this API does not handle partial
    rollout weighting. The caller ensures the episode and policy match.

    Return detached ascent gradients in parameter order, including zeros for
    unused parameters. Average these whole episode sums over episodes, without
    dividing by episode length or normalizing advantages. No actor or critic
    updates occur, and parameter values and .grad buffers remain unchanged.
    """
    _validate_episode(log_probabilities, rewards, parameters)
    current = torch.as_tensor(values, dtype=log_probabilities.dtype, device=log_probabilities.device)
    advantages = one_step_td_advantages(rewards, current, next_values, terminated, gamma)
    if advantages.shape != log_probabilities.shape:
        raise ValueError("supply one TD advantage per selected action")
    times = torch.arange(len(rewards), dtype=log_probabilities.dtype, device=log_probabilities.device)
    weights = gamma**times * advantages
    surrogate = torch.dot(weights, log_probabilities)
    derivatives = torch.autograd.grad(surrogate, tuple(parameters), materialize_grads=True)
    return tuple(derivative.detach() for derivative in derivatives)
