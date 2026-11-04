"""Exact tabular policy gradients through PyTorch's Bellman computation graph."""

from dataclasses import dataclass

import torch
from torch import Tensor

from gridworld.model import Gridworld, State
from gridworld.policy import TabularSoftmaxPolicy
from gridworld.values import PolicyValues, evaluate_policy


@dataclass(frozen=True)
class PolicyGradient:
    """Exact values and dJ/dlogits in NORTH/EAST/SOUTH/WEST action order.

    Gradient keys match policy.logits. Each cell/time row is an independent
    parameter; unreachable and optional terminal rows have zero gradient.
    """

    values: PolicyValues
    gradient: dict[State, Tensor]


def differentiate_policy(
    world: Gridworld,
    policy: TabularSoftmaxPolicy,
    gamma: float = 1.0,
) -> PolicyGradient:
    """Differentiate J = V(start_state) without changing logits or their .grad.

    Autograd follows logits -> softmax -> Bellman expectations -> J. The gamma
    factors in those expectations retain all episode discount weights; no
    manual softmax derivative or visitation-weight recurrence is needed.
    materialize_grads gives zero tensors for parameters with no path to J.
    """
    values = evaluate_policy(world, policy, gamma=gamma)
    derivatives = torch.autograd.grad(
        values.start_return, tuple(policy.logits.values()), materialize_grads=True,
    )
    gradient = {state: derivative.detach() for state, derivative in zip(policy.logits, derivatives)}
    return PolicyGradient(values, gradient)
