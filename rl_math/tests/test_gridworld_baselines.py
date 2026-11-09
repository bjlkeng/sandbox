"""Fit complete-return targets, freeze the table, and probe on separate data."""

from dataclasses import FrozenInstanceError, replace
import unittest

import numpy as np
import torch

from algorithms.reinforce import reward_to_go_gradient
from gridworld.baselines import fit_monte_carlo_baseline
from gridworld.gradients import differentiate_policy
from gridworld.model import Action, Gridworld, State
from gridworld.policy import TabularSoftmaxPolicy
from gridworld.sampling import Episode, EpisodeStep, sample_episode


def worked_policy(world, device="cpu"):
    return TabularSoftmaxPolicy({
        s: np.log([1, 2, 3, 4]) for s in world.states() if not world.is_terminal(s)
    }, device=device)


def recorded_episode(world, actions, seed, rewards=None):
    """A complete no-slip path with optional recorded reward overrides."""
    state, steps = world.start_state, []
    for t, action in enumerate(actions):
        outcome, = world.transitions(state, action)
        reward = outcome.reward if rewards is None else rewards[t]
        steps.append(EpisodeStep(t, state, action, reward, outcome.next_state,
                                 world.is_terminal(outcome.next_state)))
        state = outcome.next_state
    return Episode(seed, 0., tuple(steps), "goal" if state.cell == world.goal else "horizon")


class MonteCarloBaselineTests(unittest.TestCase):
    def test_hand_fit_averages_local_returns_by_full_state(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0)
        episodes = [recorded_episode(world, [Action.EAST, Action.EAST], 10),
                    recorded_episode(world, [Action.NORTH, Action.EAST], 11)]
        # G=(-1.5,9) and (-4,-6) at gamma=.5. Do not fit gamma**t * G_t.
        for gamma, start_mean in ((0, -3.5), (.5, -2.75), (1, -2.)):
            baseline = fit_monte_carlo_baseline(episodes, gamma=gamma, device="cpu")
            self.assertEqual(baseline(world.start_state), start_mean)
            self.assertEqual(baseline(State(world.start, 1)), -6.)
            self.assertEqual(baseline(State((0, 1), 1)), 9.)
            self.assertEqual(baseline.counts[world.start_state], 2)
            self.assertEqual(baseline.counts[State((0, 1), 1)], 1)
            self.assertEqual(sum(baseline.counts.values()), 4)
            self.assertEqual(baseline.training_seeds, (10, 11))
            self.assertEqual(baseline.gamma, gamma)
            # The fit neither invents a target at a terminal state nor reads exact V.
            self.assertNotIn(State(world.goal, 0), baseline.values)
            self.assertEqual(baseline(State(world.goal, 0)), 0.)
            self.assertEqual(baseline(State((0, 1), 2)), 0.)

    def test_early_goal_is_a_complete_target_even_with_time_remaining(self):
        world = Gridworld(layout=("SG",), horizon=4, slip=0)
        episode = recorded_episode(world, [Action.EAST], 10)
        self.assertEqual(episode.steps[-1].next_state.remaining, 3)
        baseline = fit_monte_carlo_baseline([episode], gamma=.5, device="cpu")
        self.assertEqual(dict(baseline.values), {world.start_state: 9.})
        self.assertEqual(dict(baseline.counts), {world.start_state: 1})
        self.assertEqual(baseline(episode.steps[-1].next_state), 0.)

    def test_probe_predictions_ignore_actions_rewards_and_do_not_update_the_fit(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0)
        policy = worked_policy(world)
        training = [sample_episode(world, policy, seed=s, reward_noise_std=2.)
                    for s in range(1000, 1016)]
        baseline = fit_monte_carlo_baseline(training, gamma=.5, device="cpu")
        original = (dict(baseline.values), dict(baseline.counts), baseline.training_seeds)
        probe = sample_episode(world, policy, seed=0, reward_noise_std=2.)
        changed = replace(probe, steps=tuple(
            replace(step, action=Action.WEST, reward=float("nan")) for step in probe.steps))
        # Lookup uses states alone, even when other probe fields are poisoned.
        self.assertEqual(baseline.for_episode(probe), baseline.for_episode(changed))
        prediction = baseline.for_episode(probe)
        prediction[0] = 12345.
        self.assertNotEqual(baseline.for_episode(probe)[0], 12345.)
        for _ in range(2):
            reward_to_go_gradient(probe.log_probabilities(policy.log_probabilities), probe.rewards(),
                                   tuple(policy.logits.values()), gamma=.5,
                                   baselines=baseline.for_episode(probe))
        self.assertEqual((dict(baseline.values), dict(baseline.counts), baseline.training_seeds), original)
        with self.assertRaises(TypeError):
            baseline.values[world.start_state] = 0.
        with self.assertRaises(TypeError):
            baseline.counts[world.start_state] = 0
        with self.assertRaises(FrozenInstanceError):
            baseline.gamma = 1.
        for episode in training:
            with self.assertRaisesRegex(ValueError, "training"):
                baseline.for_episode(episode)
        self.assertTrue(any(abs(baseline(s) - v.item()) > 1e-6
                            for s, v in differentiate_policy(world, policy, gamma=.5).values.v.items()
                            if s in baseline.values))

    def test_fitted_targets_and_actor_gradients_are_separate(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0)
        target = torch.nn.Parameter(torch.tensor([-6., 9.], dtype=torch.float64))
        training = recorded_episode(world, [Action.EAST, Action.EAST], 10, target)
        baseline = fit_monte_carlo_baseline([training], gamma=.5, device="cpu")
        self.assertEqual(baseline(world.start_state), -1.5)
        self.assertEqual(baseline(State((0, 1), 1)), 9.)
        # Changing source targets after fitting must not change the frozen snapshot.
        with torch.no_grad():
            target.add_(100)
        self.assertEqual(baseline(world.start_state), -1.5)
        policy = worked_policy(world)
        parameters = (*policy.logits.values(), target)
        originals = [p.detach().clone() for p in parameters]
        for parameter in parameters:
            parameter.grad = torch.full_like(parameter, 7.)
        probe = recorded_episode(world, [Action.EAST, Action.EAST], 20, [-4., 11.])
        gradient = reward_to_go_gradient(probe.log_probabilities(policy.log_probabilities),
                                         probe.rewards(), parameters, gamma=.5,
                                         baselines=baseline.for_episode(probe))
        # G_probe=(1.5,11), b=(-1.5,9), so score weights are (3,1).
        for state, row in zip(policy.logits, gradient):
            expected = ([-.3, 2.4, -.9, -1.2] if state == world.start_state else
                        [-.1, .8, -.3, -.4] if state == State((0, 1), 1) else [0.]*4)
            torch.testing.assert_close(row, row.new_tensor(expected), rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(gradient[-1], torch.zeros_like(target), rtol=0, atol=0)
        for parameter, original, row in zip(parameters, originals, gradient):
            torch.testing.assert_close(parameter, original, rtol=0, atol=0)
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 7.), rtol=0, atol=0)
            self.assertFalse(row.requires_grad)
        self.assertTrue(all(type(value) is float for value in baseline.values.values()))

    def test_fixed_learned_baseline_preserves_exact_expected_gradient(self):
        for layout, slip in ((("SHG",), 0), (("SG",), .3)):
            world = Gridworld(layout=layout, horizon=2, slip=slip)
            policy = worked_policy(world)
            parameters = tuple(policy.logits.values())
            # Fit once on a very small noisy dataset, independent of enumerated probes.
            training = [sample_episode(world, policy, seed=s, reward_noise_std=2.) for s in (1000, 1001)]

            def paths(state, probability=1., steps=()):
                if world.is_terminal(state):
                    yield probability, steps
                    return
                for action, pa in zip(Action, policy.probabilities(state).detach().tolist()):
                    for outcome in world.transitions(state, action):
                        yield from paths(outcome.next_state, probability * pa * outcome.probability,
                                         steps + ((state, action, outcome.reward),))

            trajectories = list(paths(world.start_state))
            self.assertAlmostEqual(sum(p for p, _ in trajectories), 1.)
            for gamma in (0, .5, 1):
                baseline = fit_monte_carlo_baseline(training, gamma=gamma, device="cpu")
                before = dict(baseline.values)
                expected = [torch.zeros_like(p) for p in parameters]
                for probability, steps in trajectories:
                    logp = torch.stack([policy.log_probabilities(s)[tuple(Action).index(a)]
                                        for s, a, _ in steps])
                    gradient = reward_to_go_gradient(logp, [r for _, _, r in steps], parameters,
                                                     gamma=gamma, baselines=[baseline(s) for s, _, _ in steps])
                    for total, sample in zip(expected, gradient):
                        total.add_(sample, alpha=probability)
                exact = differentiate_policy(world, policy, gamma=gamma).gradient
                for state, actual in zip(policy.logits, expected):
                    with self.subTest(layout=layout, gamma=gamma, state=state):
                        torch.testing.assert_close(actual, exact[state], rtol=1e-12, atol=1e-14)
                self.assertEqual(dict(baseline.values), before)

    def test_fitting_a_single_probe_return_erases_its_gradient(self):
        world = Gridworld(layout=("SG",), horizon=1, slip=0)
        policy = worked_policy(world)
        parameters = tuple(policy.logits.values())
        # Fixed separate training data: NORTH earns -1.
        independent = fit_monte_carlo_baseline(
            [recorded_episode(world, [Action.NORTH], 1000)], device="cpu")
        expected_separate, expected_reused = torch.zeros_like(parameters[0]), torch.zeros_like(parameters[0])
        for seed, (action, probability) in enumerate(zip(Action, [.1, .2, .3, .4])):
            probe = recorded_episode(world, [action], seed)
            separate, = reward_to_go_gradient(probe.log_probabilities(policy.log_probabilities),
                                               probe.rewards(), parameters,
                                               baselines=independent.for_episode(probe))
            leaked = fit_monte_carlo_baseline([probe], device="cpu")
            with self.assertRaisesRegex(ValueError, "training"):
                leaked.for_episode(probe)
            # Deliberately bypass the seed guard to exhibit the invalid experiment:
            # a state-only lookup fitted to this very return gives G - b = 0.
            reused, = reward_to_go_gradient(probe.log_probabilities(policy.log_probabilities),
                                             probe.rewards(), parameters, baselines=[leaked(world.start_state)])
            expected_separate += probability * separate
            expected_reused += probability * reused
        exact = differentiate_policy(world, policy).gradient[world.start_state]
        torch.testing.assert_close(expected_separate, exact, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(expected_reused, torch.zeros_like(exact), rtol=0, atol=0)
        self.assertGreater(torch.linalg.vector_norm(exact).item(), 1.)

    def test_incomplete_duplicate_and_nonfinite_training_data_are_rejected(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0)
        episode = recorded_episode(world, [Action.EAST, Action.EAST], 10)
        bad_batches = [
            [], [replace(episode, steps=())], [replace(episode, steps=episode.steps[:1])],
            [replace(episode, steps=(replace(episode.steps[0], terminated=True), episode.steps[1]))],
            [episode, episode],
            [replace(episode, steps=(replace(episode.steps[0], reward=float("nan")), episode.steps[1]))],
            [replace(episode, steps=(replace(episode.steps[0], reward=float("inf")), episode.steps[1]))],
            [recorded_episode(world, [Action.EAST, Action.EAST], 10, [1e308, 1e308])],
        ]
        for batch in bad_batches:
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                fit_monte_carlo_baseline(batch, device="cpu")
        for gamma in (-.1, 1.1, np.nan, np.inf):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                fit_monte_carlo_baseline([episode], gamma=gamma, device="cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device/driver unavailable")
    def test_cpu_cuda_fits_and_probe_gradients_agree(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=.3)
        policies = [worked_policy(world, device) for device in ("cpu", "cuda")]
        training = [sample_episode(world, policies[0], seed=s, reward_noise_std=2.)
                    for s in range(1000, 1032)]
        baselines = [fit_monte_carlo_baseline(training, gamma=.7, device=d) for d in ("cpu", "cuda")]
        self.assertEqual(baselines[0].counts, baselines[1].counts)
        self.assertEqual(baselines[0].training_seeds, baselines[1].training_seeds)
        np.testing.assert_allclose(list(baselines[0].values.values()), list(baselines[1].values.values()),
                                   rtol=1e-12, atol=1e-12)
        for seed in range(4):
            probe = sample_episode(world, policies[0], seed=seed, reward_noise_std=2.)
            gradients = [reward_to_go_gradient(probe.log_probabilities(p.log_probabilities),
                                               probe.rewards(), tuple(p.logits.values()), gamma=.7,
                                               baselines=b.for_episode(probe))
                         for p, b in zip(policies, baselines)]
            for cpu, gpu in zip(*gradients):
                self.assertEqual(gpu.device.type, "cuda")
                torch.testing.assert_close(cpu, gpu.cpu(), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
