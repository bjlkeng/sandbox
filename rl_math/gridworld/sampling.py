"""Seeded complete gridworld episodes from a discrete-action Policy.

NumPy handles random draws. Actions follow tuple(Action): NORTH, EAST, SOUTH,
WEST. Policies receive the full (cell, remaining-time) state.
"""

from dataclasses import dataclass

import numpy as np

from gridworld.model import Action, Gridworld, State
from common.policy import Policy


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
    policy: Policy[State],
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
