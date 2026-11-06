"""Full-return score sums checked by hand and by enumerating whole episodes."""

import unittest

import numpy as np
import torch

from algorithms.reinforce import full_return_gradient
from common.policy import Policy
from gridworld.gradients import differentiate_policy
from gridworld.model import Action, Gridworld, State
from gridworld.policy import TabularSoftmaxPolicy
from gridworld.sampling import sample_episode


def worked_policy(world, device="cpu"):
    return TabularSoftmaxPolicy({
        s: np.log([1, 2, 3, 4]) for s in world.states() if not world.is_terminal(s)
    }, device=device)


def selected_log_probabilities(policy, steps):
    return torch.stack([policy.log_probabilities(s)[tuple(Action).index(a)] for s, a, _ in steps])


class FullReturnTests(unittest.TestCase):
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
                expected = [torch.zeros_like(row) for row in parameters]
                for probability, steps in paths:
                    gradient = full_return_gradient(selected_log_probabilities(policy, steps),
                                                    [r for _, _, r in steps], parameters, gamma=gamma)
                    for row, sample in zip(expected, gradient):
                        row.add_(sample, alpha=probability)
                exact = differentiate_policy(world, policy, gamma=gamma).gradient
                for state, row in zip(policy.logits, expected):
                    with self.subTest(layout=layout, gamma=gamma, state=state):
                        torch.testing.assert_close(row, exact[state], rtol=1e-12, atol=1e-14)

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
        first = full_return_gradient(policy.log_probabilities(0)[[0]], [2.], (parameter, unused), gamma=.5)
        second = full_return_gradient(policy.log_probabilities(0)[[0, 0]], [3., 4.],
                                      (parameter, unused), gamma=.5)
        gradient, unused_gradient = (torch.stack(rows).mean(dim=0) for rows in zip(first, second))
        # (2*(.5,-.5) + 5*((.5,-.5)+(.5,-.5))) / 2 = (3,-3).
        torch.testing.assert_close(gradient, parameter.new_tensor([3., -3.]))
        torch.testing.assert_close(unused_gradient, torch.zeros_like(unused), rtol=0, atol=0)
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, parameter.new_tensor([7., 8.]), rtol=0, atol=0)
        self.assertIsNone(unused.grad)
        self.assertFalse(gradient.requires_grad)
        self.assertFalse(unused_gradient.requires_grad)

    def test_zero_return_and_invalid_inputs(self):
        parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        zero = torch.log_softmax(parameter, 0)[[0]]
        gradient, = full_return_gradient(zero, [0.], (parameter,))
        torch.testing.assert_close(gradient, torch.zeros_like(parameter), rtol=0, atol=0)
        for gamma in (-.1, 1.1, np.nan, np.inf):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                full_return_gradient(zero, [0.], (parameter,), gamma=gamma)
        for logp, rewards, parameters in ((zero, [0.], ()),
                (torch.empty(0), [], (parameter,)),
                (torch.zeros(2), [1.], (parameter,)),
                (torch.zeros(1, 1), [1.], (parameter,)),
                (torch.tensor([float('-inf')]), [1.], (parameter,)),
                (torch.zeros(1), [float('nan')], (parameter,))):
            with self.subTest(logp=logp, rewards=rewards), self.assertRaises(ValueError):
                full_return_gradient(logp, rewards, parameters)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device/driver unavailable")
    def test_cuda_sampled_batch_matches_cpu_and_stays_on_device(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=.3)
        policies = [worked_policy(world, device) for device in ("cpu", "cuda")]
        results = []
        samples = []
        for policy in policies:
            episodes = [sample_episode(world, policy, seed=seed, reward_noise_std=2.) for seed in range(8)]
            samples.append(episodes)
            gradients = [full_return_gradient(e.log_probabilities(policy.log_probabilities), e.rewards(),
                                              tuple(policy.logits.values()), gamma=.7) for e in episodes]
            results.append(tuple(torch.stack(rows).mean(dim=0) for rows in zip(*gradients)))
        self.assertEqual(samples[0], samples[1])
        for cpu, gpu in zip(*results):
            self.assertEqual(gpu.device.type, "cuda")
            torch.testing.assert_close(cpu, gpu.cpu(), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
