"""Full-return and reward-to-go scores checked by hand and exact enumeration."""

import unittest

import numpy as np
import torch

from algorithms.reinforce import full_return_gradient, reward_to_go_gradient
from common.policy import Policy
from gridworld.gradients import differentiate_policy
from gridworld.model import Action, Gridworld, State
from gridworld.policy import TabularSoftmaxPolicy
from gridworld.sampling import sample_episode
from gridworld.values import evaluate_policy


ESTIMATORS = (full_return_gradient, reward_to_go_gradient)


def worked_policy(world, device="cpu"):
    return TabularSoftmaxPolicy({
        s: np.log([1, 2, 3, 4]) for s in world.states() if not world.is_terminal(s)
    }, device=device)


def selected_log_probabilities(policy, steps):
    return torch.stack([policy.log_probabilities(s)[tuple(Action).index(a)] for s, a, _ in steps])


class ReinforceTests(unittest.TestCase):
    def test_log_policy_score_and_extreme_logits(self):
        state = State((0, 0), 1)
        for logits, expected in ((np.log([1, 2, 3, 4]), [-.1, .8, -.3, -.4]),
                                 ([1000, -1000, 0, 0], [-1, 1, 0, 0])):
            policy = TabularSoftmaxPolicy({state: logits}, device="cpu")
            logp = policy.log_probabilities(state)
            self.assertTrue(torch.isfinite(logp).all())
            score, = torch.autograd.grad(logp[1], (policy.logits[state],))
            torch.testing.assert_close(score, score.new_tensor(expected), rtol=1e-12, atol=1e-14)

    def test_hand_worked_east_east_trajectory_uses_the_same_full_return(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0)
        policy = worked_policy(world)
        steps = [(world.start_state, Action.EAST, -6), (State((0, 1), 1), Action.EAST, 9)]
        gradient = full_return_gradient(selected_log_probabilities(policy, steps), [-6, 9],
                                        tuple(policy.logits.values()), gamma=.5)
        # G0 = -6 + .5*9 = -1.5; each visited row gets G0*(one_hot(E)-pi).
        expected = torch.tensor([.15, -1.2, .45, .6], dtype=torch.float64)
        for state, row in zip(policy.logits, gradient):
            torch.testing.assert_close(row, expected if state in (world.start_state, State((0, 1), 1))
                                       else torch.zeros_like(expected), rtol=1e-12, atol=1e-14)

    def test_reward_to_go_discards_past_reward_but_keeps_outer_discount(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0)
        policy = worked_policy(world)
        late = State((0, 1), 1)
        steps = [(world.start_state, Action.EAST, -6), (late, Action.EAST, 9)]
        # The first score gets -6 + gamma*9; the later score gets gamma*9.
        for gamma, early_row, late_row in (
            (0, [.6, -4.8, 1.8, 2.4], [0, 0, 0, 0]),
            (.5, [.15, -1.2, .45, .6], [-.45, 3.6, -1.35, -1.8]),
            (1, [-.3, 2.4, -.9, -1.2], [-.9, 7.2, -2.7, -3.6]),
        ):
            gradients = reward_to_go_gradient(selected_log_probabilities(policy, steps), [-6, 9],
                                               tuple(policy.logits.values()), gamma=gamma)
            for state, row in zip(policy.logits, gradients):
                expected = early_row if state == world.start_state else late_row if state == late else [0]*4
                with self.subTest(gamma=gamma, state=state):
                    torch.testing.assert_close(row, row.new_tensor(expected), rtol=1e-12, atol=1e-14)

    def test_terminal_reward_only_makes_sampled_estimators_coincide(self):
        parameter = torch.nn.Parameter(torch.tensor([-.3, .7], dtype=torch.float64))
        for actions in ([1], [0, 1, 0]):
            rewards = [0.] * (len(actions) - 1) + [8.]
            for gamma in (0, .5, 1):
                gradients = [estimator(torch.log_softmax(parameter, 0)[actions], rewards,
                                        (parameter,), gamma=gamma)[0] for estimator in ESTIMATORS]
                with self.subTest(actions=actions, gamma=gamma):
                    torch.testing.assert_close(*gradients, rtol=1e-12, atol=1e-14)

    def test_enumerated_expectation_matches_exact_gradient(self):
        for layout, slip in ((("SHG",), 0), (("SG",), .3)):
            world = Gridworld(layout=layout, horizon=2, slip=slip)
            policy = worked_policy(world)
            parameters = tuple(policy.logits.values())

            def trajectories(state, probability=1., steps=()):
                if world.is_terminal(state):
                    yield probability, steps
                    return
                probabilities = policy.probabilities(state).detach().tolist()
                for action, pa in zip(Action, probabilities):
                    for outcome in world.transitions(state, action):
                        yield from trajectories(outcome.next_state, probability * pa * outcome.probability,
                                                steps + ((state, action, outcome.reward),))

            paths = list(trajectories(world.start_state))
            self.assertAlmostEqual(sum(p for p, _ in paths), 1)
            for gamma in (0, .5, 1):
                exact = differentiate_policy(world, policy, gamma=gamma).gradient
                exact_values = evaluate_policy(world, policy, gamma=gamma).v
                imperfect = {s: row.new_tensor(3 + s.remaining + 2*s.cell[1])
                             for s, row in policy.logits.items()}
                variants = ((full_return_gradient, "full", None),
                            (reward_to_go_gradient, "RTG", None),
                            (reward_to_go_gradient, "exact V", exact_values),
                            (reward_to_go_gradient, "imperfect", imperfect))
                for estimator, label, baseline in variants:
                    expected = [torch.zeros_like(row) for row in parameters]
                    for probability, steps in paths:
                        kwargs = {} if baseline is None else {
                            "baselines": torch.stack([baseline[s] for s, _, _ in steps])}
                        gradient = estimator(selected_log_probabilities(policy, steps),
                                             [r for _, _, r in steps], parameters, gamma=gamma, **kwargs)
                        for row, sample in zip(expected, gradient):
                            row.add_(sample, alpha=probability)
                    for state, row in zip(policy.logits, expected):
                        with self.subTest(variant=label, layout=layout, gamma=gamma, state=state):
                            torch.testing.assert_close(row, exact[state], rtol=1e-12, atol=1e-14)

    def test_baseline_subtraction_keeps_outer_discount(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0)
        policy = worked_policy(world)
        late = State((0, 1), 1)
        steps = [(world.start_state, Action.EAST, -6), (late, Action.EAST, 9)]
        # Fixed b=(-2.9,-1), equal to exact V at gamma=.5. The weights are
        # (-6 + gamma*9 + 2.9, gamma*(9+1)), not gamma**t*G_t - b_t.
        for gamma, early_row, late_row in (
            (0, [.31, -2.48, .93, 1.24], [0, 0, 0, 0]),
            (.5, [-.14, 1.12, -.42, -.56], [-.5, 4, -1.5, -2]),
            (1, [-.59, 4.72, -1.77, -2.36], [-1, 8, -3, -4]),
        ):
            gradient = reward_to_go_gradient(selected_log_probabilities(policy, steps), [-6, 9],
                                             tuple(policy.logits.values()), gamma=gamma,
                                             baselines=[-2.9, -1.])
            for state, row in zip(policy.logits, gradient):
                expected = early_row if state == world.start_state else late_row if state == late else [0]*4
                with self.subTest(gamma=gamma, state=state):
                    torch.testing.assert_close(row, row.new_tensor(expected), rtol=1e-12, atol=1e-14)

    def test_baseline_graph_is_detached_and_reusable_without_actor_or_target_derivatives(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0)
        policy = worked_policy(world)
        states = [world.start_state, State((0, 1), 1)]
        values = evaluate_policy(world, policy, gamma=.5)
        target_parameter = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
        # The baseline graph depends on the actor and an otherwise unused target
        # parameter. Neither path belongs in the actor's score derivative.
        baselines = torch.stack([values.v[s] for s in states]) + target_parameter
        baselines.retain_grad()
        frozen_baselines = baselines.detach().clone()
        parameters = (*policy.logits.values(), target_parameter)
        originals = [p.detach().clone() for p in parameters]
        for p in parameters:
            p.grad = torch.full_like(p, 7.)
        steps = [(s, Action.EAST, r) for s, r in zip(states, [-6, 9])]
        # Reusing even the baseline graph is safe; log-probabilities need a fresh graph.
        for _ in range(2):
            gradient = reward_to_go_gradient(selected_log_probabilities(policy, steps), [-6, 9],
                                             parameters, gamma=.5, baselines=baselines)
            for state, row in zip(policy.logits, gradient):
                expected = ([-.14, 1.12, -.42, -.56] if state == states[0] else
                            [-.5, 4, -1.5, -2] if state == states[1] else [0]*4)
                torch.testing.assert_close(row, row.new_tensor(expected), rtol=1e-12, atol=1e-14)
            torch.testing.assert_close(gradient[-1], torch.zeros_like(target_parameter), rtol=0, atol=0)
            for p, original, row in zip(parameters, originals, gradient):
                torch.testing.assert_close(p, original, rtol=0, atol=0)
                torch.testing.assert_close(p.grad, torch.full_like(p, 7.), rtol=0, atol=0)
                self.assertFalse(row.requires_grad)
            torch.testing.assert_close(baselines, frozen_baselines, rtol=0, atol=0)
            self.assertIsNone(baselines.grad)

    def test_zero_baseline_shared_parameters_and_unequal_episode_lengths(self):
        parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        gradients = []
        for rewards, baselines in (([2.], [1.]), ([3., 4.], [1., 2.])):
            actions = [0] * len(rewards)
            gradient, = reward_to_go_gradient(torch.log_softmax(parameter, 0)[actions], rewards,
                                              (parameter,), gamma=.5, baselines=baselines)
            gradients.append(gradient)
            plain, = reward_to_go_gradient(torch.log_softmax(parameter, 0)[actions], rewards,
                                           (parameter,), gamma=.5)
            zero, = reward_to_go_gradient(torch.log_softmax(parameter, 0)[actions], rewards,
                                          (parameter,), gamma=.5, baselines=[0.]*len(rewards))
            torch.testing.assert_close(plain, zero, rtol=0, atol=0)
        # Weights are (1,) and (4,1); average episode sums: (1+4+1)*(.5,-.5)/2.
        torch.testing.assert_close(torch.stack(gradients).mean(0), parameter.new_tensor([1.5, -1.5]))

    def test_baseline_requires_one_finite_scalar_per_action(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        for baselines in ([], [1.], [[1., 2.]], 1., [np.nan, 0.], [np.inf, 0.], [1e100, 0.]):
            with self.subTest(baselines=baselines), self.assertRaises(ValueError):
                reward_to_go_gradient(torch.log_softmax(parameter, 0)[[0, 1]], [1., 2.],
                                       (parameter,), baselines=baselines)
        # Finite individual weights may still overflow when subtracting the baseline.
        with self.assertRaises(ValueError):
            reward_to_go_gradient(torch.log_softmax(parameter, 0)[[0]], [3e38],
                                   (parameter,), baselines=[-3e38])

    def test_mean_of_episode_sums_with_unequal_lengths_and_shared_parameters(self):
        # A shared two-action parameter also exercises the environment-independent API.
        parameter = torch.nn.Parameter(torch.tensor([0., 0.], dtype=torch.float64))
        parameter.grad = torch.tensor([7., 8.], dtype=torch.float64)
        original = parameter.detach().clone()
        unused = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))

        class TwoActionPolicy(Policy[int]):
            def probabilities(self, state):
                return torch.softmax(parameter, dim=0)

        policy = TwoActionPolicy()  # Exercise the default log_probabilities implementation.
        # (2*(.5,-.5) + 5*((.5,-.5)+(.5,-.5))) / 2 = (3,-3).
        # Reward-to-go uses (5, .5*4) instead of (5, 5): mean = (2.25,-2.25).
        for estimator, expected in zip(ESTIMATORS, ([3., -3.], [2.25, -2.25])):
            first = estimator(policy.log_probabilities(0)[[0]], [2.], (parameter, unused), gamma=.5)
            second = estimator(policy.log_probabilities(0)[[0, 0]], [3., 4.], (parameter, unused), gamma=.5)
            gradient, unused_gradient = (torch.stack(rows).mean(dim=0) for rows in zip(first, second))
            with self.subTest(estimator=estimator.__name__):
                torch.testing.assert_close(gradient, parameter.new_tensor(expected))
                torch.testing.assert_close(unused_gradient, torch.zeros_like(unused), rtol=0, atol=0)
                torch.testing.assert_close(parameter, original, rtol=0, atol=0)
                torch.testing.assert_close(parameter.grad, parameter.new_tensor([7., 8.]), rtol=0, atol=0)
                self.assertIsNone(unused.grad)
                self.assertFalse(gradient.requires_grad)
                self.assertFalse(unused_gradient.requires_grad)

    def test_zero_return_and_invalid_inputs(self):
        parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        for estimator in ESTIMATORS:
            zero = torch.log_softmax(parameter, 0)[[0]]
            gradient, = estimator(zero, [0.], (parameter,))
            torch.testing.assert_close(gradient, torch.zeros_like(parameter), rtol=0, atol=0)
            for gamma in (-.1, 1.1, np.nan, np.inf):
                with self.subTest(estimator=estimator.__name__, gamma=gamma), self.assertRaises(ValueError):
                    estimator(zero, [0.], (parameter,), gamma=gamma)
            for logp, rewards, parameters in ((zero, [0.], ()),
                    (torch.empty(0), [], (parameter,)),
                    (torch.zeros(2), [1.], (parameter,)),
                    (torch.zeros(1, 1), [1.], (parameter,)),
                    (torch.tensor([float('-inf')]), [1.], (parameter,)),
                    (torch.zeros(1), [float('nan')], (parameter,))):
                with self.subTest(estimator=estimator.__name__, rewards=rewards), self.assertRaises(ValueError):
                    estimator(logp, rewards, parameters)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device/driver unavailable")
    def test_cuda_sampled_batch_matches_cpu_and_stays_on_device(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=.3)
        policies = [worked_policy(world, device) for device in ("cpu", "cuda")]
        for estimator in ESTIMATORS:
            results, samples = [], []
            for policy in policies:
                episodes = [sample_episode(world, policy, seed=seed, reward_noise_std=2.) for seed in range(8)]
                samples.append(episodes)
                gradients = [estimator(e.log_probabilities(policy.log_probabilities), e.rewards(),
                                       tuple(policy.logits.values()), gamma=.7) for e in episodes]
                results.append(tuple(torch.stack(rows).mean(dim=0) for rows in zip(*gradients)))
            self.assertEqual(samples[0], samples[1])
            for cpu, gpu in zip(*results):
                with self.subTest(estimator=estimator.__name__):
                    self.assertEqual(gpu.device.type, "cuda")
                    torch.testing.assert_close(cpu, gpu.cpu(), rtol=1e-12, atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device/driver unavailable")
    def test_cuda_baselines_match_cpu_including_baseline_device_conversion(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=.3)
        policies = [worked_policy(world, device) for device in ("cpu", "cuda")]
        episodes = [sample_episode(world, policies[0], seed=seed, reward_noise_std=2.) for seed in range(8)]
        values = [evaluate_policy(world, policy, gamma=.7) for policy in policies]
        for episode in episodes:
            results = []
            for policy, table in zip(policies, values):
                # A CPU baseline also gets copied to the CUDA actor's dtype/device.
                for baseline_device in ("cpu", policy.logits[world.start_state].device):
                    baseline = torch.stack([table.v[step.state] for step in episode.steps]).to(baseline_device)
                    gradient = reward_to_go_gradient(episode.log_probabilities(policy.log_probabilities),
                                                     episode.rewards(), tuple(policy.logits.values()),
                                                     gamma=.7, baselines=baseline)
                    for row in gradient:
                        self.assertEqual(row.device, policy.logits[world.start_state].device)
                        self.assertEqual(row.dtype, torch.float64)
                    results.append(gradient)
            for gradient in results[1:]:
                for cpu, actual in zip(results[0], gradient):
                    torch.testing.assert_close(cpu, actual.cpu(), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
