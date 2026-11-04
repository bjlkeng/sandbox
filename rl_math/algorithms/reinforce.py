"""Full-return REINFORCE derivatives at a frozen collecting policy."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor

from common.returns import discounted_returns


@dataclass(frozen=True)
class ReinforceEpisode:
    """Selected action log-probabilities and observed rewards for one episode.

    log_probabilities is a 1D tensor of log pi(a_t | s_t), evaluated under the
    same frozen policy that collected the episode, with its autograd graph.
    rewards[t] is the recorded r_{t+1}, treated as fixed data. The caller must
    supply complete, nonempty episodes, ending at a genuine task terminal state.
    This record is independent of environment observations and action labels.
    """

    log_probabilities: Tensor
    rewards: Sequence[float]


def full_return_gradient(
    episodes: Sequence[ReinforceEpisode],
    parameters: Sequence[Tensor],
    gamma: float = 1.0,
) -> tuple[Tensor, ...]:
    """Estimate dJ/dtheta = mean_episodes[G_0 * sum_t grad log pi(a_t | s_t)].

    J = E[G_0], with G_0 = sum_t gamma**t * r_{t+1}. Every action receives
    the same full return; no additional gamma**t multiplies its score.
    Sum within each complete episode, then average over episodes, regardless
    of their lengths. There is no reward-to-go, baseline, or normalization.

    Autograd differentiates mean(G_0 * sum(log pi)); observed returns are
    constants. This scalar is a gradient surrogate, not an estimate of J.
    Return detached derivatives in parameter order, with zeros for unused
    parameters. Do not update parameters or accumulate their .grad buffers.
    Each call consumes the supplied log-probability graphs: evaluate fresh
    log-probabilities for another probe. Parameters must require gradients.
    """
    if not episodes or not parameters:
        raise ValueError("supply at least one complete episode and one parameter")

    terms = []
    for episode in episodes:
        logp = episode.log_probabilities
        if logp.ndim != 1 or logp.numel() != len(episode.rewards) or logp.numel() == 0:
            raise ValueError("supply one scalar log-probability per reward in a nonempty episode")
        full_return = discounted_returns(episode.rewards, gamma)[0]
        if not isfinite(full_return) or not torch.isfinite(logp).all():
            raise ValueError("episode returns and selected action log-probabilities must be finite")
        terms.append(full_return * logp.sum())

    surrogate = torch.stack(terms).mean()
    derivatives = torch.autograd.grad(surrogate, tuple(parameters), materialize_grads=True)
    return tuple(derivative.detach() for derivative in derivatives)
