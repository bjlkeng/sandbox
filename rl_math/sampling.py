"""A tabular softmax policy and seeded complete gridworld episodes.

NumPy handles probability vectors and random draws. Columns follow tuple(Action):
NORTH, EAST, SOUTH, WEST. Rows use the full (cell, remaining-time) state.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from envs import Action, Gridworld, State


class TabularSoftmaxPolicy:
    """pi(a | s) = exp(logit[s, a]) / sum_b exp(logit[s, b]).

    Copy the supplied rows; each must contain four finite logits. Missing
    states raise KeyError rather than silently using a different policy.
    Terminal states need no row because no action is sampled there.

    Example: logits log(1), log(2), log(3), log(4) give probabilities
    (0.1, 0.2, 0.3, 0.4) in NORTH/EAST/SOUTH/WEST order.
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


@dataclass(frozen=True)
class EpisodeStep:
    """(s_t, a_t, r_{t+1}, s_{t+1}); t counts from this episode's start.

    terminated marks goal/horizon task endings, with zero future value.
    There are no external truncations or rollout cutoffs in this sampler.
    """

    t: int
    state: State
    action: Action
    reward: float
    next_state: State
    terminated: bool


@dataclass(frozen=True)
class Episode:
    """Recorded seed, Gaussian noise scale, trajectory, and goal/horizon ending.

    Replay requires the same gridworld, policy, seed, noise scale, and
    Python/NumPy environment. Use distinct nonnegative integer seeds for new
    episodes; repeating a seed replays rather than adding independent data.
    """

    seed: int
    reward_noise_std: float
    steps: tuple[EpisodeStep, ...]
    end_reason: str


def sample_episode(
    world: Gridworld,
    policy: TabularSoftmaxPolicy,
    *,
    seed: int,
    reward_noise_std: float = 0.0,
) -> Episode:
    """Sample a_t ~ pi(. | s_t), s_{t+1} ~ P(. | s_t, a_t), then reward.

    r_{t+1} = model_reward(s_t, a_t, s_{t+1}) + epsilon_{t+1}, where
    epsilon is an independent N(0, reward_noise_std**2) draw at each step.
    This preserves the expected-reward model, including the goal reward.
    Stop immediately on reaching the goal or exhausting the task horizon.

    Three streams spawned by SeedSequence keep action, transition, and reward draws
    explicit and avoid touching global random state. Changing only the noise
    scale leaves the sampled state/action path unchanged for a fixed seed.
    The fixed policy depends on state, not on previously sampled rewards.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(reward_noise_std) or reward_noise_std < 0:
        raise ValueError("reward_noise_std must be finite and nonnegative")

    action_seed, transition_seed, reward_seed = np.random.SeedSequence(seed).spawn(3)
    action_rng = np.random.default_rng(action_seed)
    transition_rng = np.random.default_rng(transition_seed)
    reward_rng = np.random.default_rng(reward_seed)
    actions = tuple(Action)
    state = world.start_state
    steps = []
    while not world.is_terminal(state):
        action = actions[action_rng.choice(len(actions), p=policy.probabilities(state))]
        outcomes = world.transitions(state, action)
        outcome = outcomes[transition_rng.choice(len(outcomes), p=[o.probability for o in outcomes])]
        noise = float(reward_rng.normal(0.0, reward_noise_std)) if reward_noise_std else 0.0
        terminated = world.is_terminal(outcome.next_state)
        steps.append(EpisodeStep(len(steps), state, action, outcome.reward + noise,
                                 outcome.next_state, terminated))
        state = outcome.next_state

    end_reason = "goal" if state.cell == world.goal else "horizon"
    return Episode(seed, reward_noise_std, tuple(steps), end_reason)
