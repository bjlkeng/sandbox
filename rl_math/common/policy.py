"""Evaluation interface shared by discrete-action policies."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray


Observation = TypeVar("Observation")


class Policy(ABC, Generic[Observation]):
    """Map an observation to a categorical action distribution.

    Observation can be a gridworld State, a vector, or another representation.
    The probability vector follows the environment's agreed action order.
    A one-hot vector represents a deterministic policy. The sampler owns random
    draws, keeping reproducibility independent of how probabilities are computed.

    This is an evaluation interface. Learning algorithms update a concrete
    policy's parameters separately; evaluating or sampling it does not learn.
    """

    @abstractmethod
    def probabilities(self, state: Observation) -> NDArray[np.float64]:
        """Return one finite, nonnegative probability per action, summing to 1."""
