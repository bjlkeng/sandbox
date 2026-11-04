"""Evaluation interface shared by discrete-action policies."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import Tensor


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
    def probabilities(self, state: Observation) -> Tensor:
        """Return finite nonnegative probabilities summing to 1, retaining their graph."""

    def log_probabilities(self, state: Observation) -> Tensor:
        """Return log pi(. | state), retaining the graph for score derivatives.

        Parameterized policies should override this with a stable log-space
        computation when available. Zero-probability actions have log pi=-inf
        and cannot occur in an on-policy sample.
        """
        return self.probabilities(state).log()
