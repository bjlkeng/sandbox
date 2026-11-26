"""GAE residual sums, finite n-step mixtures, and reset-safe bootstrapping."""

import unittest

import numpy as np
import torch

from algorithms.advantages import gae_advantages, n_step_advantages, one_step_td_advantages
from common.returns import discounted_returns
from gridworld.gradients import differentiate_policy
from gridworld.model import Gridworld, State
from test_algorithms_actor_critic import paths, worked_policy


class GAETests(unittest.TestCase):
    def test_worked_three_step_cutoff_and_expanded_residuals(self):
        values = torch.tensor([4., 5., 6.], dtype=torch.float64)
        # delta = (-.5, 0, 1); last row is a continuing-task cutoff.
        for lambda_, expected in ((0., [-.5, 0., 1.]),
                                  (.5, [-.4375, .25, 1.]), (1., [-.25, .5, 1.])):
            actual = gae_advantages([1., 2., 3.], values, [5., 6., 8.],
                                    [False]*3, [False]*3, .5, lambda_=lambda_)
            torch.testing.assert_close(actual, values.new_tensor(expected), rtol=0, atol=0)
        # -.5 + (.5*.5)*0 + (.5*.5)**2*1 = -.4375, not a plain sum.

    def test_lambda_endpoints_match_td_and_full_return_with_boundaries(self):
        values = torch.tensor([4., 5., 6., 7.], dtype=torch.float64)
        for terminal, truncated in (([False]*4, [False]*4),
                                   ([False, True, False, True], [False]*4),
                                   ([False, False, False, True], [False, True, False, False]),
                                   ([False, True, False, False], [False, True, False, False])):
            for gamma in (0., .5, 1.):
                # Replace inconsistent interior predictions with the SAME frozen
                # critic at the next row; boundary predictions can differ.
                future = [99. if terminal[t] or truncated[t] else values[t+1].item()
                          for t in range(3)] + [8.]
                args = ([1., 2., 3., 4.], values, future, terminal, truncated, gamma)
                torch.testing.assert_close(gae_advantages(*args, lambda_=0.),
                    one_step_td_advantages(*args[:4], gamma), rtol=0, atol=0)
                torch.testing.assert_close(gae_advantages(*args, lambda_=1.),
                    n_step_advantages(*args, n=4), rtol=1e-12, atol=1e-14)
        complete = gae_advantages([1., 2., 3.], values[:3], [5., 6., 999.],
                                  [False, False, True], [False]*3, .5, lambda_=1.)
        torch.testing.assert_close(complete, values.new_tensor(discounted_returns([1., 2., 3.], .5))-values[:3])

    def test_reset_stops_trace_but_truncation_keeps_final_observation_value(self):
        values = torch.tensor([4., 5., 6., 7.], dtype=torch.float64)
        for terminal, truncated, expected in (
                ([False, True, False, False], [False]*4, [-1.25, -3.]),
                ([False]*4, [False, True, False, False], [1.25, 7.]),
                ([False, True, False, False], [False, True, False, False], [-1.25, -3.])):
            for reset_reward in (100., 10000.):
                actual = gae_advantages([1., 2., reset_reward, 200.], values, [5., 20., 7., 40.],
                                        terminal, truncated, .5, lambda_=.5)
                # First episode unaffected by the reset episode's large reward.
                torch.testing.assert_close(actual[:2], values.new_tensor(expected[:2]))
            for term, trunc, expected_last in ((True, False, -2.), (False, True, 2.),
                                               (False, False, 2.), (True, True, -2.)):
                single = gae_advantages([2.], values[:1], [8.], [term], [trunc], .5, lambda_=1.)
                self.assertEqual(single.item(), expected_last)

    def test_explicit_forward_residual_sum_and_finite_n_step_mixture(self):
        rng = np.random.default_rng(2026)
        for size in (1, 3, 9):
            rewards, current = rng.normal(size=(2, size))
            terminal, truncated = rng.random((2, size)) < .2
            future = np.r_[current[1:], 8.]
            future[terminal | truncated] = 20.  # Final observation before reset.
            values = torch.tensor(current)
            for gamma in (0., .6, 1.):
                delta = rewards + gamma*np.where(terminal, 0., future) - current
                for lambda_ in (0., .3, .8, 1.):
                    expected = []
                    for t in range(size):
                        total = 0.
                        for j in range(t, size):
                            total += (gamma*lambda_)**(j-t) * delta[j]
                            if terminal[j] or truncated[j]:
                                break
                        expected.append(total)
                    args = (rewards, values, future, terminal, truncated, gamma)
                    actual = gae_advantages(*args, lambda_=lambda_)
                    torch.testing.assert_close(actual, values.new_tensor(expected), rtol=1e-12, atol=1e-14)
                    mixture = sum((1-lambda_)*lambda_**(n-1)*n_step_advantages(*args, n=n)
                                  for n in range(1, size))
                    mixture = mixture + lambda_**(size-1)*n_step_advantages(*args, n=size)
                    torch.testing.assert_close(actual, mixture, rtol=1e-12, atol=1e-14)

    def test_actor_detachment_and_unchanged_grad_buffers(self):
        actor = torch.nn.Parameter(torch.tensor([.2, -.1], dtype=torch.float64))
        critic = torch.nn.Parameter(torch.tensor([4., 5., 6.], dtype=torch.float64))
        rewards = critic.new_tensor([1., 2., 3.]) + actor.sum() + critic.sum()
        current = critic + actor.sum()
        future = torch.cat((current[1:], current.new_tensor([8.])))
        inputs = (actor, critic, rewards, current, future)
        copies = [x.detach().clone() for x in inputs]
        for p in (actor, critic):
            p.grad = torch.full_like(p, 7.)
        for _ in range(2):  # Target graph can be reused; action graph must be fresh.
            adv = gae_advantages(rewards, current, future, [False]*3, [False]*3, .5, lambda_=.7)
            self.assertFalse(adv.requires_grad)
            logp = actor.log_softmax(0)[[0, 1, 0]]
            weights = adv * adv.new_tensor([1., .5, .25])
            gradient = torch.autograd.grad(torch.dot(weights, logp), (actor, critic), materialize_grads=True)
            scores = torch.eye(2, dtype=actor.dtype)[[0, 1, 0]] - actor.softmax(0).detach()
            torch.testing.assert_close(gradient[0], (weights[:, None]*scores).sum(0))
            torch.testing.assert_close(gradient[1], torch.zeros_like(critic))
        for x, original in zip(inputs, copies):
            torch.testing.assert_close(x, original, rtol=0, atol=0)
        for p in (actor, critic):
            torch.testing.assert_close(p.grad, torch.full_like(p, 7.), rtol=0, atol=0)

    def test_enumerated_gradient_with_exact_and_imperfect_critics(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0.)
        policy = worked_policy(world)
        for gamma in (0., .5, 1.):
            exact = differentiate_policy(world, policy, gamma)
            reference = torch.stack([exact.gradient[s] for s in policy.logits])
            for imperfect in (False, True):
                table = {s:v.detach().clone() for s,v in exact.values.v.items()}
                if imperfect:
                    table[State((0, 1), 1)] += 4.
                for lambda_ in (0., .4, 1.):
                    expected = torch.zeros_like(reference)
                    for probability, episode in paths(world, policy):
                        adv = gae_advantages(episode.rewards(), torch.stack([table[s.state] for s in episode.steps]),
                            torch.stack([table[s.next_state] for s in episode.steps]),
                            [s.terminated for s in episode.steps], [False]*len(episode.steps), gamma, lambda_=lambda_)
                        times = torch.arange(len(episode.steps), dtype=adv.dtype)
                        surrogate = torch.dot(gamma**times * adv, episode.log_probabilities(policy.log_probabilities))
                        gradient = torch.autograd.grad(surrogate, tuple(policy.logits.values()), materialize_grads=True)
                        expected += probability*torch.stack(gradient)
                    bias = torch.zeros_like(reference)
                    if imperfect:
                        bias[list(policy.logits).index(world.start_state)] = (
                            (1-lambda_)*gamma*4.*.2*bias.new_tensor([-.1,.8,-.3,-.4]))
                    torch.testing.assert_close(expected-reference, bias, rtol=1e-12, atol=1e-14)

    def test_invalid_inputs_and_nonfinite_accumulation(self):
        args = dict(rewards=[1., 2.], values=torch.tensor([3., 4.]), next_values=[4., 8.],
                    terminated=[False, False], truncated=[False, False], gamma=.5, lambda_=.5)
        for changes in ({'lambda_': -.1}, {'lambda_': 1.1}, {'lambda_': np.nan},
                        {'lambda_': np.inf}, {'gamma': np.nan}, {'gamma': -.1}, {'gamma': 1.1},
                        {'rewards': [1.]}, {'rewards': [1., np.inf]}, {'next_values': [1.]},
                        {'values': torch.tensor([])}, {'values': torch.ones(1,2)},
                        {'values': torch.tensor([1, 2])}, {'values': torch.tensor([1., np.nan])},
                        {'next_values': [1., np.inf]}, {'terminated': [0, 1]},
                        {'terminated': [False]}, {'truncated': [0, 1]}, {'truncated': [False]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                gae_advantages(**(args | changes))
        # Each delta is finite, but their undiscounted accumulated sum is not.
        huge = torch.finfo(torch.float32).max
        with self.assertRaises(ValueError):
            gae_advantages([huge, huge], torch.zeros(2), [0., 0.],
                           [False, True], [False, False], 1., lambda_=1.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device/driver unavailable')
    def test_cuda_dtype_device_and_cpu_agreement(self):
        for dtype in (torch.float32, torch.float64):
            values = torch.tensor([4.,5.,6.,7.], dtype=dtype)
            future = torch.tensor([5.,20.,7.,40.], dtype=torch.float32)
            args = ([1.,2.,100.,200.], [False]*4, [False,True,False,False])
            for lambda_ in (0., .37, 1.):
                cpu = gae_advantages(args[0], values, future, *args[1:], .7, lambda_=lambda_)
                gpu = gae_advantages(args[0], values.cuda(), future, *args[1:], .7, lambda_=lambda_)
                self.assertEqual(gpu.device.type, 'cuda')
                self.assertEqual(gpu.dtype, dtype)
                torch.testing.assert_close(cpu, gpu.cpu(), rtol=1e-6 if dtype==torch.float32 else 1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
