"""A differentiable tabular softmax policy over gridworld cell/time states."""

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from common.policy import Policy
from gridworld.model import Action, State


class TabularSoftmaxPolicy(Policy[State]):
    """pi(a | s) = exp(logit[s, a]) / sum_b exp(logit[s, b]).

    Rows use the full (cell, remaining-time) state. Copy the supplied rows;
    each must contain four finite logits in NORTH/EAST/SOUTH/WEST order.
    Missing states raise KeyError; terminal states need no row because no
    action is sampled there. Logits are owned float64 parameters. By default
    use CUDA when available; device="cpu" selects the CPU reference explicitly.

    Example: logits log(1), log(2), log(3), log(4) give probabilities
    (0.1, 0.2, 0.3, 0.4). These logits are parameters that a later algorithm
    can optimize; this class only converts their current values to probabilities.
    """

    def __init__(
        self,
        logits: Mapping[State, Sequence[float] | Tensor],
        *,
        device: str | torch.device | None = None,
    ):
        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.logits = {
            state: torch.nn.Parameter(torch.as_tensor(row, dtype=torch.float64, device=device).clone())
            for state, row in logits.items()
        }
        if any(row.shape != (len(Action),) or not torch.isfinite(row).all()
               for row in self.logits.values()):
            raise ValueError("each policy row must contain four finite logits")

    def probabilities(self, state: State) -> Tensor:
        # PyTorch's stable softmax also records its derivative in the graph.
        return torch.softmax(self.logits[state], dim=0)

    def log_probabilities(self, state: State) -> Tensor:
        # log(softmax(...)) can underflow before the logarithm; log_softmax avoids it.
        return torch.log_softmax(self.logits[state], dim=0)
