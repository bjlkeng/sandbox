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
    taken there. Values are float64 tensors on the policy's device, retaining
    the graph back to its parameters; start_return is a scalar tensor.
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

    # Model constants and terminal values follow the policy onto CPU or CUDA.
    device = policy.probabilities(world.start_state).device
    zero = torch.zeros((), dtype=torch.float64, device=device)
    v = {state: zero for state in world.states()}
    q = {state: zero.new_zeros(len(Action)) for state in v}
    advantage = {state: zero.new_zeros(len(Action)) for state in v}
    # states() increases remaining time: zero-time boundary before longer horizons.
    for state in v:
        if world.is_terminal(state):
            continue
        action_values = []
        for action in Action:
            outcomes = world.transitions(state, action)
            probabilities = zero.new_tensor([o.probability for o in outcomes])
            rewards = zero.new_tensor([o.reward for o in outcomes])
            future_values = torch.stack([v[o.next_state] for o in outcomes])
            action_values.append(torch.dot(probabilities, rewards + gamma * future_values))
        q[state] = torch.stack(action_values)
        v[state] = torch.dot(policy.probabilities(state).to(q[state]), q[state])
        advantage[state] = q[state] - v[state]

    return PolicyValues(v, q, advantage, v[world.start_state])
