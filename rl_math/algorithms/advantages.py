"""Detached advantage targets, independent of environment and policy types."""

from collections.abc import Sequence
from numbers import Integral

import torch
from torch import Tensor


def one_step_td_advantages(
    rewards: Tensor | Sequence[float],
    values: Tensor,
    next_values: Tensor | Sequence[float],
    terminated: Tensor | Sequence[bool],
    gamma: float = 1.0,
) -> Tensor:
    """Return delta_t = r_{t+1} + gamma * Vhat(s_{t+1}) - Vhat(s_t).

    Supply one finite scalar reward, current value, and next value per
    transition. values is a nonempty 1D floating tensor; all calculations use
    its device/dtype. Freeze the critic while constructing these targets.
    Rewards and both value predictions are detached from the actor graph.

    terminated is a boolean mask for true task endings: replace the next
    value with zero there, even if the supplied prediction is nonzero. A goal
    and an explicitly defined finite task horizon both end the gridworld task.
    An external truncation or collection cutoff does not: supply its final
    observation's value and False, never the reset observation's value.

    These residuals have no outer gamma**t; that belongs in the actor sum.
    An exact critic makes their conditional means the true advantages. A
    fixed imperfect critic can bias those means through the next-state value.
    """
    values, rewards, next_values, terminal = _transition_inputs(
        rewards, values, next_values, terminated, gamma)

    future = torch.where(terminal, torch.zeros_like(next_values), next_values)
    advantages = rewards + gamma * future - values
    if not torch.isfinite(advantages).all():
        raise ValueError("TD advantages must be finite")
    return advantages


def _transition_inputs(rewards, values, next_values, terminated, gamma):
    """Validate and detach one scalar per transition on the values device."""
    if not 0. <= gamma <= 1.:
        raise ValueError("gamma must lie in [0, 1]")
    if values.ndim != 1 or not values.numel() or not values.is_floating_point():
        raise ValueError("values must be a nonempty 1D floating tensor")
    values = values.detach()
    rewards = torch.as_tensor(rewards, dtype=values.dtype, device=values.device).detach()
    next_values = torch.as_tensor(next_values, dtype=values.dtype, device=values.device).detach()
    terminal = torch.as_tensor(terminated, device=values.device)
    if any(x.shape != values.shape or not torch.isfinite(x).all()
           for x in (values, rewards, next_values)):
        raise ValueError("supply one finite reward, current value, and next value per transition")
    if terminal.shape != values.shape or terminal.dtype != torch.bool:
        raise ValueError("supply one boolean terminated flag per transition")

    return values, rewards, next_values, terminal


def _n_step_targets(rewards, next_values, terminal, truncated, gamma, n):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n must be a positive integer")
    truncated = torch.as_tensor(truncated, device=next_values.device)
    if truncated.shape != next_values.shape or truncated.dtype != torch.bool:
        raise ValueError("supply one boolean truncated flag per transition")

    bootstrap = torch.where(terminal, torch.zeros_like(next_values), next_values)
    targets = rewards + gamma * bootstrap  # One observed reward, then a value.
    boundary = terminal | truncated
    for _ in range(1, min(n, len(rewards))):
        # Extend by one observed reward at a time. The batch's last transition
        # always uses its own final observation; no later reward is available.
        following = torch.cat((targets[1:], bootstrap[-1:]))
        future = torch.where(boundary, bootstrap, following)
        targets = rewards + gamma * future
    if not torch.isfinite(targets).all():
        raise ValueError("n-step returns must be finite")
    return targets


def n_step_returns(
    rewards: Tensor | Sequence[float],
    next_values: Tensor,
    terminated: Tensor | Sequence[bool],
    truncated: Tensor | Sequence[bool],
    gamma: float = 1.0,
    *,
    n: int,
) -> Tensor:
    """Sum up to n rewards, then bootstrap, stopping at reset/batch boundaries.

    For k observed transitions before the first boundary (or n steps), return
    sum_{j=0}^{k-1} gamma**j * r_{t+j+1} + gamma**k * Vhat(s_{t+k}).
    A genuine termination replaces that final value with zero. A truncation
    stops reward accumulation but retains the final pre-reset state's value.
    If both flags are true, termination wins. At the batch end, a rollout
    cutoff also retains its final value, without requiring a truncation flag.

    Supply consecutive transitions in a nonempty 1D stream, with a flag at
    every reset boundary. next_values[t] MUST describe transition t's actual
    successor before any reset, never the next row's reset observation. This
    helper cannot detect an incorrectly supplied observation/value or flag.
    A task's own finite horizon is termination; external time limits truncate.

    next_values is a floating tensor defining output dtype/device. All scalar
    inputs must be finite and all targets are detached. Freeze the critic and
    collecting policy. n is a positive integer; it may exceed the batch length.
    This computes targets only, with no outer episode-time gamma**t weights.
    """
    _, rewards, next_values, terminal = _transition_inputs(
        rewards, torch.zeros_like(next_values), next_values, terminated, gamma)
    return _n_step_targets(rewards, next_values, terminal, truncated, gamma, n)


def n_step_advantages(
    rewards: Tensor | Sequence[float],
    values: Tensor,
    next_values: Tensor | Sequence[float],
    terminated: Tensor | Sequence[bool],
    truncated: Tensor | Sequence[bool],
    gamma: float = 1.0,
    *,
    n: int,
) -> Tensor:
    """Return detached n_step_returns(...) - Vhat(s_t) on values' dtype/device.

    Boundary semantics and caller requirements follow n_step_returns. With
    n=1 this is one_step_td_advantages. Covering a complete episode gives
    G_t - Vhat(s_t); covering a cutoff includes the final bootstrap instead.
    Current-state values are also detached. For a discounted start-state actor
    objective, multiply by gamma**t with t measured from episode start, sum
    each episode, then average episode sums. Partial rollout weighting and
    policy updates are not implemented here.
    """
    values, rewards, next_values, terminal = _transition_inputs(
        rewards, values, next_values, terminated, gamma)
    targets = _n_step_targets(rewards, next_values, terminal, truncated, gamma, n)
    advantages = targets - values
    if not torch.isfinite(advantages).all():
        raise ValueError("n-step advantages must be finite")
    return advantages
