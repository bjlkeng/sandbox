"""Discrete-action policy interface and a NumPy tabular softmax implementation."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from envs import Action, State


Observation = TypeVar("Observation")


class Policy(ABC, Generic[Observation]):
    """Map an observation to a categorical action distribution.

    Observation can be a gridworld State, a vector, or another representation.
    The probability vector follows the environment's agreed action order; the
    gridworld sampler uses NORTH, EAST, SOUTH, WEST. A one-hot vector represents
    a deterministic policy. Random draws belong to the sampler so it controls
    reproducibility independently of how probabilities are computed.

    This is an evaluation interface. Learning algorithms update a concrete
    policy's parameters separately; evaluating or sampling it does not learn.
    """

    @abstractmethod
    def probabilities(self, state: Observation) -> NDArray[np.float64]:
        """Return one finite, nonnegative probability per action, summing to 1."""


class TabularSoftmaxPolicy(Policy[State]):
    """pi(a | s) = exp(logit[s, a]) / sum_b exp(logit[s, b]).

    Rows use the full (cell, remaining-time) state. Copy the supplied rows;
    each must contain four finite logits in NORTH/EAST/SOUTH/WEST order.
    Missing states raise KeyError; terminal states need no row because no
    action is sampled there.

    Example: logits log(1), log(2), log(3), log(4) give probabilities
    (0.1, 0.2, 0.3, 0.4). These logits are parameters that a later algorithm
    can optimize; this class only converts their current values to probabilities.
    """

    def __init__(self, logits: Mapping[State, ArrayLike]):
        self.logits = {state: np.array(row, dtype=float, copy=True) for state, row in logits.items()}
        if any(row.shape != (len(Action),) or not np.isfinite(row).all()
               for row in self.logits.values()):
            raise ValueError("each policy row must contain four finite logits")

    def probabilities(self, state: State) -> NDArray[np.float64]:
        row = self.logits[state]
        # Subtracting a constant leaves softmax unchanged and avoids overflow.
        weights = np.exp(row - row.max())
        return weights / weights.sum()
