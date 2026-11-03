"""Exact finite-horizon values under a fixed gridworld policy."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from common.policy import Policy
from gridworld.model import Action, Gridworld, State


@dataclass(frozen=True)
class PolicyValues:
    """V(s), Q(s, .), A(s, .), and J = V(start_state) for one evaluation.

    Keys include every non-wall cell/time pair, even those unreachable from
    the start. Q and advantage rows use NORTH/EAST/SOUTH/WEST order.
    Goal and zero-time states have zero V, Q, and advantage; no action is
    taken there. These tables are a snapshot of the policy at evaluation.
    """

    v: dict[State, float]
    q: dict[State, NDArray[np.float64]]
    advantage: dict[State, NDArray[np.float64]]
    start_return: float


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
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")

    v = {state: 0.0 for state in world.states()}
    q = {state: np.zeros(len(Action)) for state in v}
    advantage = {state: np.zeros(len(Action)) for state in v}
    # states() increases remaining time: zero-time boundary before longer horizons.
    for state in v:
        if world.is_terminal(state):
            continue
        for index, action in enumerate(Action):
            outcomes = world.transitions(state, action)
            probabilities = np.array([o.probability for o in outcomes])
            rewards = np.array([o.reward for o in outcomes])
            future_values = np.array([v[o.next_state] for o in outcomes])
            q[state][index] = np.dot(probabilities, rewards + gamma * future_values)
        v[state] = float(np.dot(policy.probabilities(state), q[state]))
        advantage[state] = q[state] - v[state]

    return PolicyValues(v, q, advantage, v[world.start_state])
