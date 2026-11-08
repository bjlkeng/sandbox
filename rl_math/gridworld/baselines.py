"""Fit state/time baselines from completed training episodes, then freeze them."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

import torch

from common.returns import discounted_returns
from gridworld.model import State
from gridworld.sampling import Episode


@dataclass(frozen=True)
class MonteCarloBaseline:
    """A frozen table of sample means, with zero for states unseen during fitting.

    Each key is the full (cell, remaining-time) state. values and counts are
    read-only snapshots; predictions are Python scalars with no autograd graph.
    reward_to_go_gradient copies these constants onto the actor's device/dtype.
    gamma records the discount used for the training targets.
    """

    values: Mapping[State, float]
    counts: Mapping[State, int]
    training_seeds: tuple[int, ...]
    gamma: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(self, "training_seeds", tuple(self.training_seeds))

    def __call__(self, state: State) -> float:
        """Predict from state alone; a missing state gets the fixed fallback 0."""
        return self.values.get(state, 0.)

    def for_episode(self, episode: Episode) -> list[float]:
        """Look up a separate probe's states without reading its actions/rewards.

        Reject a training seed to catch accidental replay or reuse of fitting
        episodes. This guard is not a proof of independence: callers must use
        independent sampling streams and must not relabel training episodes.
        Lookup never fits, updates counts, or uses a probe return as a target.
        """
        if episode.seed in self.training_seeds:
            raise ValueError("probe seed overlaps training data; use a separate episode")
        return [self(step.state) for step in episode.steps]


def fit_monte_carlo_baseline(
    training_episodes: Sequence[Episode],
    gamma: float = 1.,
    *,
    device: str | torch.device | None = None,
) -> MonteCarloBaseline:
    """Fit b(s) = mean{G_t : s_t=s} on complete, separately sampled episodes.

    This is the exact tabular minimizer of sum_t (b(s_t) - G_t)**2 on the
    supplied data: no optimizer, learning rate, or epochs are needed. Compute
    each G_t with discounted_returns, starting from zero at task termination.
    Targets use discounting relative to t, without the actor's outer gamma**t.
    No current prediction, next-state value, or exact model value enters them:
    this is Monte Carlo regression, not a bootstrapped update.

    Use the frozen collecting policy and disjoint seeds for training/probes.
    This function accepts only training data; freeze its result before probing.
    The same cell at different remaining times has a different table entry.
    Terminal states are not action states and receive no training target.
    Unseen states use zero at lookup, which is still action-independent.

    Recorded rewards are constants, including tensor-valued scalar observations.
    Average their return targets in float64 on the selected device (CUDA when
    available by default), then snapshot the results as immutable scalar data.
    Fitting has no actor parameter input and retains no target computation graph.
    """
    discounted_returns([], gamma)  # Validate gamma even before reading the data.
    if not training_episodes:
        raise ValueError("supply at least one complete training episode")
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    seeds = tuple(episode.seed for episode in training_episodes)
    if any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be distinct nonnegative integers")

    targets_by_state: dict[State, list[float]] = {}
    for episode in training_episodes:
        if (not episode.steps or not episode.steps[-1].terminated
                or any(step.terminated for step in episode.steps[:-1])):
            raise ValueError("training episodes must be nonempty and end at task termination")
        rewards = [r.detach().item() if isinstance(r, torch.Tensor) else r for r in episode.rewards()]
        targets = discounted_returns(rewards, gamma)
        if not all(isfinite(value) for value in targets):
            raise ValueError("training return targets must be finite")
        for step, target in zip(episode.steps, targets):
            targets_by_state.setdefault(step.state, []).append(target)

    # Each scalar mean solves that state's independent least-squares problem.
    values = {state: torch.tensor(targets, dtype=torch.float64, device=device).mean().item()
              for state, targets in targets_by_state.items()}
    if not all(isfinite(value) for value in values.values()):
        raise ValueError("fitted baseline values must be finite")
    counts = {state: len(targets) for state, targets in targets_by_state.items()}
    return MonteCarloBaseline(values, counts, seeds, float(gamma))
