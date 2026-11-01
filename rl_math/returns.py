"""Returns and policy-gradient weights for complete finite episodes.

rewards[t] is r_{t+1}, received after taking action a_t in state s_t.
There is no bootstrap value: the future value at the terminal state is zero.
"""

from collections.abc import Sequence


def discounted_returns(rewards: Sequence[float], gamma: float = 1.0) -> list[float]:
    """Compute G_t = r_{t+1} + gamma * G_{t+1}, starting with G_T = 0.

    Return one value per action, including the final transition's reward.
    An empty episode/terminal tail returns []; its total return is zero.
    gamma must lie in [0, 1]. The input sequence is left unchanged.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")

    returns = [0.0] * len(rewards)
    future_return = 0.0  # G_T: no rewards after termination.
    for t in reversed(range(len(rewards))):
        future_return = float(rewards[t]) + gamma * future_return
        returns[t] = future_return
    return returns


def reward_to_go_weights(rewards: Sequence[float], gamma: float = 1.0) -> list[float]:
    """Return gamma**t * G_t for the discounted start-state gradient.

    t starts at the beginning of this complete episode. Sum weight * score
    over its actions, then average those sums over episodes. These weights
    contain no episode-length, batch-size, or advantage normalization.
    """
    return [gamma**t * value for t, value in enumerate(discounted_returns(rewards, gamma))]
