"""Exact tabular logit derivatives by reversing the Bellman computation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gridworld.model import Action, Gridworld, State
from gridworld.policy import TabularSoftmaxPolicy
from gridworld.values import PolicyValues, evaluate_policy


@dataclass(frozen=True)
class PolicyGradient:
    """Snapshot of exact values, dJ/dlogits, and reverse-pass value weights.

    Gradient keys match policy.logits, in NORTH/EAST/SOUTH/WEST action order.
    Each cell/time row is an independent parameter, even when rows have equal
    logits. Unreachable and optional terminal parameter rows have zero gradient.
    No policy parameters are changed.

    For nonterminal s, value_weights[s] = dJ/dV(s) in the Bellman computation
    = gamma**t * Pr(s_t=s), where t = horizon - s.remaining. These weights
    include the episode discount; they are not normalized visitation counts.
    Terminal values are constants, so their reverse weights are kept at zero.
    """

    values: PolicyValues
    gradient: dict[State, NDArray[np.float64]]
    value_weights: dict[State, float]


def differentiate_policy(
    world: Gridworld,
    policy: TabularSoftmaxPolicy,
    gamma: float = 1.0,
) -> PolicyGradient:
    """Differentiate J = V(start_state) through the finite Bellman graph.

    First evaluate V and Q with evaluate_policy. Then reverse its order:
    horizon, horizon-1, ..., 1. Start with w(start)=1 and all other w=0.
    The chain rule through V(s) = sum_a pi(a|s) Q(s,a) gives

        dJ/dtheta[s,b] = w(s) * pi(b|s) * [Q(s,b) - V(s)]
        w(s') += w(s) * gamma * sum_a pi(a|s) P(s'|s,a).

    The first line uses d pi(a)/d theta[b] = pi(a) * (1[a=b] - pi(b)).
    The second follows the dependency of Q on successor values. Accumulate
    all predecessor contributions before processing a successor's row.
    Goal and horizon values are fixed at zero, with no derivative to pass on.

    This is an analytic reverse derivative of the value computation, using
    NumPy without an autodiff dependency. Transition probabilities and reward
    means are fixed; independent zero-mean reward noise affects neither J nor
    its gradient. gamma must lie in [0, 1]. There is no sampling or update.
    """
    values = evaluate_policy(world, policy, gamma=gamma)
    gradient = {state: np.zeros(len(Action)) for state in policy.logits}
    value_weights = {state: 0.0 for state in values.v}
    value_weights[world.start_state] = 1.0

    # evaluate_policy inserts states in increasing remaining-time order.
    for state in reversed(values.v):
        if world.is_terminal(state):
            continue
        weight = value_weights[state]
        probabilities = policy.probabilities(state)
        # Reverse the local softmax expectation, holding Q fixed at this node.
        gradient[state] = weight * probabilities * values.advantage[state]
        for action, probability in zip(Action, probabilities):
            for outcome in world.transitions(state, action):
                if not world.is_terminal(outcome.next_state):
                    value_weights[outcome.next_state] += (
                        weight * gamma * probability * outcome.probability
                    )

    return PolicyGradient(values, gradient, value_weights)
