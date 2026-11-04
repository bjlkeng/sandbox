"""Differentiable exact finite-horizon values under a fixed gridworld policy."""

from dataclasses import dataclass

import torch
from torch import Tensor

from common.policy import Policy
from gridworld.model import Action, Gridworld, State


@dataclass(frozen=True)
class PolicyValues:
    """V(s), Q(s, .), A(s, .), and J = V(start_state) for one evaluation.

    Keys include every non-wall cell/time pair, even those unreachable from
    the start. Q and advantage rows use NORTH/EAST/SOUTH/WEST order.
    Goal and zero-time states have zero V, Q, and advantage; no action is
    taken there. Values are float64 CPU tensors that retain the computation
    graph back to the policy's parameters; start_return is a scalar tensor.
    """

    v: dict[State, Tensor]
    q: dict[State, Tensor]
    advantage: dict[State, Tensor]
    start_return: Tensor


def evaluate_policy(
    world: Gridworld,
    policy: Policy[State],
    gamma: float = 1.0,
) -> PolicyValues:
    """Compute Q, V, and A by backward dynamic programming, without sampling.

    Q(s,a) = sum_s' P(s'|s,a) * [reward(s,a,s') + gamma * V(s')]
    V(s)   = sum_a pi(a|s) * Q(s,a)
    A(s,a) = Q(s,a) - V(s)

    The full state includes remaining time. Start with zero terminal values,
    then compute remaining=1, 2, ..., horizon: every nonterminal successor
    has one fewer step left, so its V is already known. This is a single
    finite backward pass; gamma=1 needs no iterative convergence argument.

    The policy supplies rows for every nonterminal state and stays fixed
    during evaluation. The expectation over its actions is explicit; a max
    would instead choose actions and solve the Bellman optimality equation.
    Model rewards are conditional means, so the sampler's independent
    zero-mean reward noise does not affect these expected values.

    Use torch.stack for previously computed values: torch.tensor(list_of_values)
    or conversion to Python floats would detach them from the autograd graph.
    Only fixed model probabilities and rewards are constructed as constants.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")

    v = {state: torch.tensor(0.0, dtype=torch.float64) for state in world.states()}
    q = {state: torch.zeros(len(Action), dtype=torch.float64) for state in v}
    advantage = {state: torch.zeros(len(Action), dtype=torch.float64) for state in v}
    # states() increases remaining time: zero-time boundary before longer horizons.
    for state in v:
        if world.is_terminal(state):
            continue
        action_values = []
        for action in Action:
            outcomes = world.transitions(state, action)
            probabilities = torch.tensor([o.probability for o in outcomes], dtype=torch.float64)
            rewards = torch.tensor([o.reward for o in outcomes], dtype=torch.float64)
            future_values = torch.stack([v[o.next_state] for o in outcomes])
            action_values.append(torch.dot(probabilities, rewards + gamma * future_values))
        q[state] = torch.stack(action_values)
        v[state] = torch.dot(policy.probabilities(state).to(q[state]), q[state])
        advantage[state] = q[state] - v[state]

    return PolicyValues(v, q, advantage, v[world.start_state])
