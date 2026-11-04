"""Policy probabilities, complete episode boundaries, and independent noise."""

import random
import unittest

import numpy as np
import torch

from gridworld.model import Action, Gridworld, State
from common.policy import Policy
from gridworld.policy import TabularSoftmaxPolicy
from common.returns import discounted_returns
from gridworld.sampling import sample_episode


def east_policy(world):
    # Finite logits whose other probabilities underflow to zero: a test fixture
    # for hand-checkable paths, not the exploratory policy used in the notebook.
    return TabularSoftmaxPolicy({s: (-1000, 0, -1000, -1000) for s in world.states()}, device="cpu")


class PolicyTests(unittest.TestCase):
    def test_policy_requires_a_probability_implementation(self):
        class IncompletePolicy(Policy[State]):
            pass

        with self.assertRaisesRegex(TypeError, "probabilities"):
            IncompletePolicy()

    def test_worked_logits_and_action_order(self):
        state = State((0, 0), 2)
        row = np.log([1, 2, 3, 4])
        policy = TabularSoftmaxPolicy({state: row}, device="cpu")
        self.assertIsInstance(policy, Policy)
        row[:] = 0  # Caller-owned arrays must not change the policy after construction.
        self.assertEqual(tuple(Action), (Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST))
        self.assertIsInstance(policy.probabilities(state), torch.Tensor)
        self.assertTrue(policy.logits[state].is_leaf)
        self.assertTrue(policy.logits[state].requires_grad)
        self.assertTrue(policy.probabilities(state).requires_grad)
        torch.testing.assert_close(policy.probabilities(state),
                                   torch.tensor([.1, .2, .3, .4], dtype=torch.float64))
        self.assertAlmostEqual(policy.probabilities(state).sum().item(), 1.0)

    def test_stable_softmax_and_time_specific_rows(self):
        early, late = State((0, 0), 2), State((0, 0), 1)
        policy = TabularSoftmaxPolicy({early: (1000, 1001, 1002, 1003), late: (0, 0, 0, 0)}, device="cpu")
        shifted = TabularSoftmaxPolicy({early: (0, 1, 2, 3)}, device="cpu")
        torch.testing.assert_close(policy.probabilities(early), shifted.probabilities(early))
        torch.testing.assert_close(policy.probabilities(late), torch.full((4,), .25, dtype=torch.float64))
        self.assertFalse(torch.allclose(policy.probabilities(early), policy.probabilities(late)))

    def test_invalid_rows_and_missing_state(self):
        state = State((0, 0), 1)
        for row in ((), (0, 0, 0), (0,) * 5, [[0] * 4], (0, 0, 0, np.nan),
                    (0, 0, 0, np.inf), (0, 0, 0, -np.inf)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                TabularSoftmaxPolicy({state: row}, device="cpu")
        with self.assertRaises(KeyError):
            TabularSoftmaxPolicy({}, device="cpu").probabilities(state)


class SamplingTests(unittest.TestCase):
    def test_sampler_accepts_a_non_tabular_policy(self):
        class SafeRoutePolicy(Policy[State]):
            def probabilities(self, state):
                action = Action.SOUTH if state.cell[0] == 0 else Action.EAST
                probabilities = torch.zeros(len(Action), dtype=torch.float64)
                probabilities[tuple(Action).index(action)] = 1.0
                return probabilities

        world = Gridworld(layout=("S#", ".G"), horizon=4, slip=0)
        episode = sample_episode(world, SafeRoutePolicy(), seed=7)
        self.assertEqual([s.action for s in episode.steps], [Action.SOUTH, Action.EAST])
        self.assertEqual([s.next_state for s in episode.steps],
                         [State((1, 0), 3), State((1, 1), 2)])
        self.assertEqual([s.reward for s in episode.steps], [-1, 9])
        self.assertEqual(episode.end_reason, "goal")
        self.assertTrue(episode.steps[-1].terminated)

    def test_goal_episode_has_timestamps_rewards_and_no_absorbing_tail(self):
        world = Gridworld(layout=("SHG",), horizon=4, slip=0)
        episode = sample_episode(world, east_policy(world), seed=7)
        self.assertEqual(episode.seed, 7)
        self.assertEqual(episode.reward_noise_std, 0)
        self.assertEqual(episode.end_reason, "goal")
        self.assertEqual([s.t for s in episode.steps], [0, 1])
        self.assertEqual([s.state for s in episode.steps],
                         [State((0, 0), 4), State((0, 1), 3)])
        self.assertEqual([s.next_state for s in episode.steps],
                         [State((0, 1), 3), State((0, 2), 2)])
        self.assertEqual([s.action for s in episode.steps], [Action.EAST] * 2)
        self.assertEqual([s.reward for s in episode.steps], [-6, 9])
        self.assertEqual([s.terminated for s in episode.steps], [False, True])
        self.assertEqual(discounted_returns([s.reward for s in episode.steps], 0.5)[0], -1.5)

    def test_horizon_is_task_termination_and_last_step_goal_keeps_reward(self):
        for layout, final_cell, end_reason, reward in [
            (("S.G",), (0, 1), "horizon", -1),
            (("SG",), (0, 1), "goal", 9),
        ]:
            world = Gridworld(layout=layout, horizon=1, slip=0)
            with self.subTest(layout=layout):
                episode = sample_episode(world, east_policy(world), seed=2)
                self.assertEqual(episode.end_reason, end_reason)
                step, = episode.steps
                self.assertEqual(step.t, 0)
                self.assertEqual(step.next_state, State(final_cell, 0))
                self.assertEqual(step.reward, reward)
                self.assertTrue(step.terminated)

    def test_seed_replays_stochastic_episode_without_using_global_random(self):
        world = Gridworld()
        policy = TabularSoftmaxPolicy({s: (0, 1, 0, 0) for s in world.states()}, device="cpu")
        global_state = random.getstate()
        numpy_global_state = np.random.get_state()
        first = sample_episode(world, policy, seed=42, reward_noise_std=2)
        self.assertEqual(first, sample_episode(world, policy, seed=42, reward_noise_std=2))
        self.assertEqual(random.getstate(), global_state)
        np.testing.assert_equal(np.random.get_state(), numpy_global_state)
        self.assertNotEqual(first.steps,
                            sample_episode(world, policy, seed=43, reward_noise_std=2).steps)
        for t, step in enumerate(first.steps):
            self.assertEqual(step.t, t)
            self.assertEqual(step.state.remaining, world.horizon - t)
            self.assertEqual(step.terminated, t == len(first.steps) - 1)
            self.assertIn(step.next_state,
                          [o.next_state for o in world.transitions(step.state, step.action)])
            if t:
                self.assertEqual(first.steps[t - 1].next_state, step.state)
        self.assertTrue(world.is_terminal(first.steps[-1].next_state))

    def test_noise_scale_changes_only_rewards_and_preserves_expected_model(self):
        world = Gridworld()
        policy = TabularSoftmaxPolicy({s: (0, 1, 0, 0) for s in world.states()}, device="cpu")
        model = tuple(world.transitions(world.start_state, a) for a in Action)
        episodes = [sample_episode(world, policy, seed=42, reward_noise_std=std)
                    for std in (0, 1, 2)]
        self.assertEqual([e.reward_noise_std for e in episodes], [0, 1, 2])
        self.assertEqual(len({len(e.steps) for e in episodes}), 1)
        for clean, noisy, twice in zip(*(e.steps for e in episodes)):
            self.assertEqual((clean.t, clean.state, clean.action, clean.next_state, clean.terminated),
                             (noisy.t, noisy.state, noisy.action, noisy.next_state, noisy.terminated))
            self.assertEqual((clean.state, clean.action, clean.next_state),
                             (twice.state, twice.action, twice.next_state))
            outcome, = (o for o in world.transitions(clean.state, clean.action)
                        if o.next_state == clean.next_state)
            self.assertEqual(clean.reward, outcome.reward)
            self.assertAlmostEqual(twice.reward - clean.reward,
                                   2 * (noisy.reward - clean.reward))
        self.assertTrue(any(a.reward != b.reward for a, b in zip(episodes[0].steps, episodes[1].steps)))
        self.assertEqual(model, tuple(world.transitions(world.start_state, a) for a in Action))

    def test_each_episode_restarts_t_at_zero(self):
        world = Gridworld(horizon=2)
        policy = east_policy(world)
        for seed in (1, 2):
            episode = sample_episode(world, policy, seed=seed)
            self.assertEqual([s.t for s in episode.steps], [0, 1])
            self.assertEqual(episode.steps[0].state, world.start_state)
            self.assertTrue(episode.steps[-1].terminated)

    def test_noise_has_zero_mean_configured_variance_and_no_time_correlation(self):
        world = Gridworld(layout=("SH.G",), horizon=2, slip=0)
        policy = east_policy(world)
        samples = [sample_episode(world, policy, seed=seed, reward_noise_std=2)
                   for seed in range(2000)]
        # The deterministic path pays -6 then -1 before the added noise.
        noise = np.array([[step.reward for step in e.steps] for e in samples]) - [-6, -1]
        # Broad, fixed-seed Monte Carlo tolerances: six standard errors.
        self.assertTrue(np.all(np.abs(noise.mean(axis=0)) < 6 * 2 / np.sqrt(len(samples))))
        self.assertTrue(np.all(np.abs(noise.var(axis=0) - 4) < 6 * 4 * np.sqrt(2 / len(samples))))
        covariance = np.cov(noise, rowvar=False, ddof=0)[0, 1]
        self.assertLess(abs(covariance), 6 * 4 / np.sqrt(len(samples)))

    def test_invalid_noise_and_seed(self):
        world = Gridworld()
        policy = east_policy(world)
        for std in (-1, np.nan, np.inf, -np.inf):
            with self.subTest(std=std), self.assertRaises(ValueError):
                sample_episode(world, policy, seed=0, reward_noise_std=std)
        for seed in (None, -1, 0.5, True, "7"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                sample_episode(world, policy, seed=seed)


if __name__ == "__main__":
    unittest.main()
