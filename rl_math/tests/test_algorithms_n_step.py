"""n-step targets: hand arithmetic, reset boundaries, and policy gradients."""

import unittest

import numpy as np
import torch

from algorithms.advantages import n_step_advantages, n_step_returns, one_step_td_advantages
from common.returns import discounted_returns
from gridworld.gradients import differentiate_policy
from gridworld.model import Gridworld, State
from test_algorithms_actor_critic import paths, worked_policy


class NStepTests(unittest.TestCase):
    def test_worked_path_interpolates_between_td_and_mc(self):
        values = torch.tensor([-2.9, -1.], dtype=torch.float64)
        for n, targets, advantages in ((1, [-6.5, 9.], [-3.6, 10.]),
                                       (2, [-1.5, 9.], [1.4, 10.]),
                                       (20, [-1.5, 9.], [1.4, 10.])):
            args = ([-6., 9.], values.new_tensor([-1., 999.]), [False, True], [False]*2, .5)
            torch.testing.assert_close(n_step_returns(*args, n=n), values.new_tensor(targets))
            torch.testing.assert_close(n_step_advantages(args[0], values, *args[1:], n=n),
                                       values.new_tensor(advantages))

    def test_terminal_truncation_and_cutoff_before_reset(self):
        # Row 2 is a RESET episode with a huge reward: row 1 must not use it.
        future = torch.tensor([10., 20., 30., 40.], dtype=torch.float64)
        for terminated, truncated, expected in (
                ([False, True, False, False], [False]*4, [2., 2., 210., 220.]),
                ([False]*4, [False, True, False, False], [7., 12., 210., 220.]),
                ([False, True, False, False], [False, True, False, False], [2., 2., 210., 220.]),
                ([False, True, False, False], [False, False, True, False], [2., 2., 115., 220.])):
            actual = n_step_returns([1., 2., 100., 200.], future, terminated, truncated, .5, n=99)
            torch.testing.assert_close(actual, actual.new_tensor(expected))
        # Identical final observation and reward: termination removes value;
        # external truncation and rollout cutoff both retain it.
        for terminated, truncated, expected in ((True, False, 2.), (False, True, 6.),
                                                 (False, False, 6.)):
            actual = n_step_returns([2.], future.new_tensor([8.]), [terminated], [truncated], .5, n=5)
            self.assertEqual(actual.item(), expected)

    def test_one_step_and_completed_episode_endpoints(self):
        values = torch.tensor([1., -3., 2.], dtype=torch.float64)
        future = values.new_tensor([-3., 2., 500.])
        for gamma in (0., .5, 1.):
            args = ([1., 2., 3.], values, future, [False, False, True])
            torch.testing.assert_close(n_step_advantages(*args, [False]*3, gamma, n=1),
                                       one_step_td_advantages(*args, gamma))
            expected = values.new_tensor(discounted_returns(args[0], gamma)) - values
            torch.testing.assert_close(n_step_advantages(*args, [False]*3, gamma, n=3), expected)

    def test_matches_independent_forward_sum_for_many_boundaries(self):
        rng = np.random.default_rng(1729)
        for size in (1, 3, 11):
            rewards, future = rng.normal(size=(2, size))
            terminal = rng.random(size) < .25
            truncated = rng.random(size) < .25
            for gamma in (0., .3, 1.):
                for n in (1, 2, 5, 30):
                    expected = []
                    for t in range(size):
                        total = 0.
                        for j in range(t, min(t+n, size)):
                            total += gamma**(j-t) * rewards[j]
                            if terminal[j] or truncated[j]:
                                break
                        if not terminal[j]:
                            total += gamma**(j-t+1) * future[j]
                        expected.append(total)
                    actual = n_step_returns(rewards, torch.tensor(future), terminal, truncated, gamma, n=n)
                    torch.testing.assert_close(actual, actual.new_tensor(expected), rtol=1e-12, atol=1e-14)

    def test_detached_actor_weights_and_unchanged_inputs_and_grad_buffers(self):
        actor = torch.nn.Parameter(torch.tensor([.2, -.1], dtype=torch.float64))
        critic = torch.nn.Parameter(torch.tensor([1., 2., 3.], dtype=torch.float64))
        for p in (actor, critic):
            p.grad = torch.full_like(p, 7.)
        # Both critic and actor graphs can occur in the supplied targets.
        rewards = critic + actor.sum()
        values, future = 2*critic + actor.sum(), 3*critic + actor.sum()
        snapshots = [v.detach().clone() for v in (actor, critic, rewards, values, future)]
        for n in (1, 2, 9):
            adv = n_step_advantages(rewards, values, future, [False, False, True], [False]*3, .5, n=n)
            returns = n_step_returns(rewards, future, [False, False, True], [False]*3, .5, n=n)
            self.assertFalse(adv.requires_grad)
            self.assertFalse(returns.requires_grad)
            logp = torch.log_softmax(actor, 0)[[0, 1, 0]]
            weights = adv * adv.new_tensor([1., .5, .25])
            gradient = torch.autograd.grad(torch.dot(weights, logp), (actor, critic), materialize_grads=True)
            scores = torch.eye(2, dtype=actor.dtype)[[0, 1, 0]] - actor.softmax(0).detach()
            torch.testing.assert_close(gradient[0], (weights[:, None] * scores).sum(0))
            torch.testing.assert_close(gradient[1], torch.zeros_like(critic))
        for value, snapshot in zip((actor, critic, rewards, values, future), snapshots):
            torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
        for p in (actor, critic):
            torch.testing.assert_close(p.grad, torch.full_like(p, 7.), rtol=0, atol=0)

    def test_exact_critic_gradient_for_intermediate_n_and_early_termination(self):
        world = Gridworld(layout=("SG",), horizon=3, slip=.3)
        policy = worked_policy(world)
        episodes = list(paths(world, policy))
        for gamma in (0., .5, 1.):
            exact = differentiate_policy(world, policy, gamma)
            for n in (1, 2, 3, 20):
                expected = torch.zeros(len(policy.logits), 4, dtype=torch.float64)
                for probability, episode in episodes:
                    adv = n_step_advantages(
                        episode.rewards(), torch.stack([exact.values.v[s.state] for s in episode.steps]),
                        torch.stack([exact.values.v[s.next_state] for s in episode.steps]),
                        [s.terminated for s in episode.steps], [False]*len(episode.steps), gamma, n=n)
                    times = torch.arange(len(episode.steps), dtype=adv.dtype)
                    surrogate = torch.dot(gamma**times * adv, episode.log_probabilities(policy.log_probabilities))
                    gradient = torch.autograd.grad(surrogate, tuple(policy.logits.values()), materialize_grads=True)
                    expected += probability * torch.stack(gradient)
                reference = torch.stack([exact.gradient[s] for s in policy.logits])
                torch.testing.assert_close(expected, reference, rtol=1e-12, atol=1e-14)

    def test_longer_return_removes_selected_bootstrap_bias(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0.)
        policy = worked_policy(world)
        exact = differentiate_policy(world, policy, .5)
        table = {s: v.detach().clone() for s, v in exact.values.v.items()}
        table[State((0, 1), 1)] += 4.
        reference = torch.stack([exact.gradient[s] for s in policy.logits])
        for n in (1, 2):
            expected = torch.zeros_like(reference)
            for probability, episode in paths(world, policy):
                adv = n_step_advantages(episode.rewards(), torch.stack([table[s.state] for s in episode.steps]),
                    torch.stack([table[s.next_state] for s in episode.steps]),
                    [s.terminated for s in episode.steps], [False]*len(episode.steps), .5, n=n)
                surrogate = torch.dot(adv * adv.new_tensor([1., .5]),
                                      episode.log_probabilities(policy.log_probabilities))
                gradient = torch.autograd.grad(surrogate, tuple(policy.logits.values()), materialize_grads=True)
                expected += probability * torch.stack(gradient)
            bias = torch.zeros_like(reference)
            if n == 1:
                bias[list(policy.logits).index(world.start_state)] = bias.new_tensor([-.04, .32, -.12, -.16])
            torch.testing.assert_close(expected-reference, bias, rtol=1e-12, atol=1e-14)

    def test_invalid_inputs(self):
        args = dict(rewards=[1., 2.], next_values=torch.tensor([3., 4.]),
                    terminated=[False, True], truncated=[False, False], gamma=.5, n=2)
        for changes in ({'n': 0}, {'n': -1}, {'n': 1.5}, {'n': True}, {'gamma': np.nan},
                        {'gamma': 1.1}, {'gamma': -.1}, {'truncated': [0, 1]},
                        {'truncated': [False]}, {'terminated': [0, 1]}, {'rewards': [1.]},
                        {'rewards': [1., np.inf]}, {'next_values': torch.tensor([np.nan, 2.])},
                        {'next_values': torch.tensor([1, 2])}, {'next_values': torch.tensor([])},
                        {'next_values': torch.ones(1, 2)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                n_step_returns(**(args | changes))
        with self.assertRaises(ValueError):
            n_step_advantages(**args, values=torch.tensor([1.]))
        huge = torch.tensor([torch.finfo(torch.float32).max])
        with self.assertRaises(ValueError):
            n_step_returns(huge, huge, [False], [False], n=1)
        with self.assertRaises(ValueError):
            n_step_advantages(huge, -huge, [0.], [True], [False], n=1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device/driver unavailable")
    def test_cuda_dtype_device_and_cpu_agreement(self):
        values = torch.tensor([1., 2., -3., 4.], dtype=torch.float64)
        future = torch.tensor([2., 99., 8., 5.])  # Deliberately CPU/float32.
        for n in (1, 2, 9):
            args = ([1., -2., 3., 4.], [False, True, False, False], [False, False, True, False])
            cpu = n_step_advantages(args[0], values, future, *args[1:], .7, n=n)
            gpu = n_step_advantages(args[0], values.cuda(), future, *args[1:], .7, n=n)
            self.assertEqual(gpu.device.type, 'cuda')
            self.assertEqual(gpu.dtype, torch.float64)
            torch.testing.assert_close(cpu, gpu.cpu(), rtol=1e-12, atol=1e-14)
            returns = n_step_returns(args[0], future.cuda(), *args[1:], .7, n=n)
            self.assertEqual(returns.device.type, 'cuda')
            self.assertEqual(returns.dtype, future.dtype)


if __name__ == '__main__':
    unittest.main()
