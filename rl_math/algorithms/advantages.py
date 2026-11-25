"""Detached advantage targets, independent of environment and policy types."""

from collections.abc import Sequence

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

    future = torch.where(terminal, torch.zeros_like(next_values), next_values)
    advantages = rewards + gamma * future - values
    if not torch.isfinite(advantages).all():
        raise ValueError("TD advantages must be finite")
    return advantages
