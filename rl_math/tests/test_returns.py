"""Deterministic checks for the discounted start-state convention."""

import math
import unittest

from returns import discounted_returns, reward_to_go_weights


class ReturnTests(unittest.TestCase):
    def test_worked_example(self):
        for gamma, expected_returns, expected_weights in [
            (1.0, [6.0, 5.0, 3.0], [6.0, 5.0, 3.0]),
            (0.5, [2.75, 3.5, 3.0], [2.75, 1.75, 0.75]),
            (0.0, [1.0, 2.0, 3.0], [1.0, 0.0, 0.0]),
        ]:
            with self.subTest(gamma=gamma):
                self.assertEqual(discounted_returns([1, 2, 3], gamma), expected_returns)
                self.assertEqual(reward_to_go_weights([1, 2, 3], gamma), expected_weights)
                # This trajectory is deterministic, so J = G_0.
                self.assertEqual(expected_returns[0], 1 + gamma * 2 + gamma**2 * 3)

    def test_terminal_reward_is_included_with_zero_future_value(self):
        self.assertEqual(discounted_returns([7], 0.5), [7.0])
        self.assertEqual(discounted_returns([0, 0, 8], 0.5), [2.0, 4.0, 8.0])
        self.assertEqual(reward_to_go_weights([0, 0, 8], 0.5), [2.0, 2.0, 2.0])

    def test_empty_terminal_tail(self):
        self.assertEqual(discounted_returns([], 0.5), [])
        self.assertEqual(reward_to_go_weights([], 0.5), [])

    def test_negative_rewards_and_input_is_preserved(self):
        rewards = [2, -4, 1]
        self.assertEqual(discounted_returns(rewards, 0.5), [0.25, -3.5, 1.0])
        self.assertEqual(rewards, [2, -4, 1])
        self.assertEqual(discounted_returns((2, -4, 1), 0.5), [0.25, -3.5, 1.0])

    def test_invalid_discount(self):
        for gamma in [-0.1, 1.1, math.nan, math.inf, -math.inf]:
            for helper in [discounted_returns, reward_to_go_weights]:
                with self.subTest(gamma=gamma, helper=helper.__name__):
                    with self.assertRaises(ValueError):
                        helper([], gamma)


class EpisodeGradientTests(unittest.TestCase):
    """An exact two-action tree checks weights against a separate objective.

    Both actions use Bernoulli(p), p = sigmoid(theta). At t=0, action 0
    terminates with reward 2; action 1 gives -1 and continues. At t=1,
    action 0 ends with reward 1 and action 1 ends with reward 4.
    The scalar score d log pi(a) / d theta is a - p.
    """

    @staticmethod
    def objective(theta, gamma):
        p = 1 / (1 + math.exp(-theta))
        # Enumerate path probabilities and start-discounted rewards directly.
        return (1 - p) * 2 + p * ((1 - p) * (-1 + gamma) + p * (-1 + gamma * 4))

    def test_expected_episode_sum_matches_objective_derivative(self):
        for theta in [-0.7, 0.0, 0.8]:
            p = 1 / (1 + math.exp(-theta))
            paths = [
                (1 - p, [2], [-p]),
                (p * (1 - p), [-1, 1], [1 - p, -p]),
                (p * p, [-1, 4], [1 - p, 1 - p]),
            ]
            for gamma in [0.0, 0.5, 1.0]:
                with self.subTest(theta=theta, gamma=gamma):
                    expected_gradient = sum(
                        probability * sum(
                            weight * score
                            for weight, score in zip(reward_to_go_weights(rewards, gamma), scores)
                        )
                        for probability, rewards, scores in paths
                    )
                    step = 1e-6
                    finite_difference = (
                        self.objective(theta + step, gamma)
                        - self.objective(theta - step, gamma)
                    ) / (2 * step)
                    self.assertAlmostEqual(expected_gradient, finite_difference, places=8)

    def test_average_episode_sums_in_a_batch_with_different_lengths(self):
        # At theta=0, the three paths have probabilities 1/2, 1/4, 1/4.
        # Repeating the short path twice gives their exact population weights.
        episodes = [
            ([2], [-0.5]),
            ([2], [-0.5]),
            ([-1, 1], [0.5, -0.5]),
            ([-1, 4], [0.5, 0.5]),
        ]
        episode_sums = [
            sum(weight * score for weight, score in zip(reward_to_go_weights(rewards, 0.5), scores))
            for rewards, scores in episodes
        ]
        self.assertEqual(episode_sums, [-1.0, -1.0, -0.5, 1.5])
        gradient = sum(episode_sums) / len(episodes)
        self.assertEqual(gradient, -0.25)
        self.assertEqual(gradient, 0.25 * (-3 + 0.5 + 6 * 0.5 * 0.5))
        transition_count = sum(len(rewards) for rewards, _ in episodes)
        self.assertNotEqual(sum(episode_sums) / transition_count, gradient)
        per_episode_time_mean = sum(
            total / len(rewards) for total, (rewards, _) in zip(episode_sums, episodes)
        ) / len(episodes)
        self.assertNotEqual(per_episode_time_mean, gradient)


if __name__ == "__main__":
    unittest.main()
