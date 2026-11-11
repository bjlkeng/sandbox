"""One-step TD arithmetic, detached actor targets, and exact expectations."""

import unittest

import numpy as np
import torch

from algorithms.actor_critic import one_step_td_gradient
from algorithms.advantages import one_step_td_advantages
from algorithms.reinforce import reward_to_go_gradient
from gridworld.gradients import differentiate_policy
from gridworld.model import Action, Gridworld, State
from gridworld.policy import TabularSoftmaxPolicy
from gridworld.sampling import Episode, EpisodeStep, sample_episode
from gridworld.values import evaluate_policy


def worked_policy(world, device="cpu"):
    return TabularSoftmaxPolicy({s: np.log([1., 2., 3., 4.]) for s in world.states()
                                if not world.is_terminal(s)}, device=device)


def paths(world, policy, state=None, probability=1., steps=()):
    state = world.start_state if state is None else state
    if world.is_terminal(state):
        yield probability, Episode(0, 0., steps, "goal" if state.cell == world.goal else "horizon")
        return
    for action, pa in zip(Action, policy.probabilities(state).detach().tolist()):
        for outcome in world.transitions(state, action):
            step = EpisodeStep(len(steps), state, action, outcome.reward, outcome.next_state,
                               world.is_terminal(outcome.next_state))
            yield from paths(world, policy, outcome.next_state, probability * pa * outcome.probability,
                             steps + (step,))


def td_gradient(episode, policy, table, gamma, extra_parameters=()):
    return one_step_td_gradient(
        episode.log_probabilities(policy.log_probabilities), episode.rewards(),
        (*policy.logits.values(), *extra_parameters), gamma=gamma,
        values=torch.stack([table[s.state] for s in episode.steps]),
        next_values=torch.stack([table[s.next_state] for s in episode.steps]),
        terminated=[s.terminated for s in episode.steps],
    )


class ActorCriticTests(unittest.TestCase):
    def test_worked_transition_and_terminal_arithmetic(self):
        # S/2 -> H/1 -> goal: the terminal prediction is deliberately wrong.
        values = torch.tensor([-2.9, -1.], dtype=torch.float64, requires_grad=True)
        next_values = torch.tensor([-1., 999.], dtype=torch.float64, requires_grad=True)
        for gamma, expected in ((0., [-3.1, 10.]), (.5, [-3.6, 10.]), (1., [-4.1, 10.])):
            delta = one_step_td_advantages([-6., 9.], values, next_values, [False, True], gamma)
            torch.testing.assert_close(delta, delta.new_tensor(expected), rtol=1e-12, atol=1e-14)
            self.assertFalse(delta.requires_grad)
        # A collection cutoff is not a task terminal: retain its bootstrap.
        delta = one_step_td_advantages([2.], values[:1], [8.], [False], .5)
        torch.testing.assert_close(delta, delta.new_tensor([8.9]))

    def test_hand_gradient_keeps_outer_discount_and_unused_rows(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0.)
        policy = worked_policy(world)
        start, late = world.start_state, State((0, 1), 1)
        logp = torch.stack([policy.log_probabilities(s)[1] for s in (start, late)])
        gradient = one_step_td_gradient(logp, [-6., 9.], tuple(policy.logits.values()), .5,
                                        values=[-2.9, -1.], next_values=[-1., 999.],
                                        terminated=[False, True])
        # Weights (-3.6, .5*10) times the EAST score (-.1,.8,-.3,-.4).
        for state, row in zip(policy.logits, gradient):
            expected = ([.36, -2.88, 1.08, 1.44] if state == start else
                        [-.5, 4., -1.5, -2.] if state == late else [0.]*4)
            torch.testing.assert_close(row, row.new_tensor(expected), rtol=1e-12, atol=1e-14)

    def test_detach_rewards_and_both_value_predictions_without_changing_grad_buffers(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0.)
        policy = worked_policy(world)
        target = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
        table = evaluate_policy(world, policy, .5).v
        current = torch.stack([table[world.start_state], table[State((0, 1), 1)]]) + target
        future = torch.stack([table[State((0, 1), 1)], table[State(world.goal, 0)]]) + target
        rewards = current.new_tensor([-6., 9.]) + target
        parameters = (*policy.logits.values(), target)
        originals = [p.detach().clone() for p in parameters]
        for parameter in parameters:
            parameter.grad = torch.full_like(parameter, 7.)
        for _ in range(2):
            logp = torch.stack([policy.log_probabilities(s)[1]
                                for s in (world.start_state, State((0, 1), 1))])
            gradient = one_step_td_gradient(logp, rewards, parameters, .5, values=current,
                                            next_values=future, terminated=[False, True])
            for state, row in zip(policy.logits, gradient):
                expected = ([.36, -2.88, 1.08, 1.44] if state == world.start_state else
                            [-.5, 4., -1.5, -2.] if state == State((0, 1), 1) else [0.]*4)
                torch.testing.assert_close(row, row.new_tensor(expected), rtol=1e-12, atol=1e-14)
            torch.testing.assert_close(gradient[-1], torch.zeros_like(target), rtol=0, atol=0)
            for parameter, original, row in zip(parameters, originals, gradient):
                torch.testing.assert_close(parameter, original, rtol=0, atol=0)
                torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 7.), rtol=0, atol=0)
                self.assertFalse(row.requires_grad)

    def test_enumerated_exact_critic_matches_reference_with_slip_and_early_termination(self):
        for layout, horizon, slip in ((("SHG",), 2, 0.), (("SG",), 3, .3)):
            world = Gridworld(layout=layout, horizon=horizon, slip=slip)
            policy = worked_policy(world)
            episodes = list(paths(world, policy))
            self.assertAlmostEqual(sum(p for p, _ in episodes), 1.)
            if slip:
                self.assertGreater(len({len(e.steps) for _, e in episodes}), 1)
            for gamma in (0., .5, 1.):
                exact = differentiate_policy(world, policy, gamma)
                # Even nonzero goal/time-limit predictions must be ignored.
                table = {s: v if not world.is_terminal(s) else v.new_tensor(999.)
                         for s, v in exact.values.v.items()}
                expected = torch.zeros(len(policy.logits), 4, dtype=torch.float64)
                for probability, episode in episodes:
                    expected += probability * torch.stack(td_gradient(episode, policy, table, gamma))
                reference = torch.stack([exact.gradient[s] for s in policy.logits])
                with self.subTest(layout=layout, gamma=gamma):
                    torch.testing.assert_close(expected, reference, rtol=1e-12, atol=1e-14)

    def test_imperfect_bootstrap_bias_but_same_table_as_mc_baseline_stays_unbiased(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0.)
        policy = worked_policy(world)
        for gamma in (0., .5, 1.):
            exact = differentiate_policy(world, policy, gamma)
            table = {s: v.detach().clone() for s, v in exact.values.v.items()}
            table[State((0, 1), 1)] += 4.
            reference = torch.stack([exact.gradient[s] for s in policy.logits])
            td, mc = torch.zeros_like(reference), torch.zeros_like(reference)
            for probability, episode in paths(world, policy):
                td += probability * torch.stack(td_gradient(episode, policy, table, gamma))
                mc += probability * torch.stack(reward_to_go_gradient(
                    episode.log_probabilities(policy.log_probabilities), episode.rewards(),
                    tuple(policy.logits.values()), gamma,
                    baselines=torch.stack([table[s.state] for s in episode.steps]),
                ))
            # Only EAST at S/2 leads to the +4 error at H/1.
            bias = torch.zeros_like(reference)
            start_row = list(policy.logits).index(world.start_state)
            bias[start_row] = gamma * 4. * .2 * bias.new_tensor([-.1, .8, -.3, -.4])
            torch.testing.assert_close(td - reference, bias, rtol=1e-12, atol=1e-14)
            torch.testing.assert_close(mc, reference, rtol=1e-12, atol=1e-14)
            if gamma == .5:
                self.assertAlmostEqual(bias.square().sum().item(), .144)

    def test_shared_parameters_average_whole_episode_sums(self):
        parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        gradients = []
        for rewards, current, future, terminal in (([2.], [1.], [999.], [True]),
                ([3., 4.], [1., 2.], [2., 999.], [False, True])):
            logp = torch.log_softmax(parameter, 0)[[0]*len(rewards)]
            gradient, = one_step_td_gradient(logp, rewards, (parameter,), .5, values=current,
                                             next_values=future, terminated=terminal)
            gradients.append(gradient)
        # The episode weights sum to 1 and (3 + .5*2)=4, giving 2.5*(.5,-.5).
        torch.testing.assert_close(torch.stack(gradients).mean(0), parameter.new_tensor([1.25, -1.25]))

    def test_invalid_targets_and_episode_inputs(self):
        values = torch.tensor([1., 2.], dtype=torch.float64)
        for replacements in ({"gamma": -.1}, {"gamma": 1.1}, {"gamma": np.nan},
                {"rewards": [1.]}, {"rewards": [1., np.inf]}, {"values": values[:0]},
                {"values": values.reshape(1, 2)}, {"values": torch.tensor([1., np.nan])},
                {"next_values": [1.]}, {"next_values": [np.inf, 0.]},
                {"terminated": [False]}, {"terminated": [0, 1]}):
            args = dict(rewards=[1., 2.], values=values, next_values=[2., 0.],
                        terminated=[False, True], gamma=.5)
            args.update(replacements)
            with self.subTest(replacements=replacements), self.assertRaises(ValueError):
                one_step_td_advantages(**args)
        parameter = torch.nn.Parameter(torch.zeros(2))
        for logp, parameters in ((torch.zeros(1), (parameter,)),
                                 (torch.tensor([0., -np.inf]), (parameter,)), (torch.zeros(2), ())):
            with self.assertRaises(ValueError):
                one_step_td_gradient(logp, [1., 2.], parameters, values=values,
                                     next_values=[2., 0.], terminated=[False, True])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device/driver unavailable")
    def test_cuda_gradient_matches_cpu_with_cpu_critic_predictions(self):
        world = Gridworld(layout=("SHG",), horizon=3, slip=.3)
        policies = [worked_policy(world, device) for device in ("cpu", "cuda")]
        table = evaluate_policy(world, policies[0], .7).v
        for seed in range(8):
            episode = sample_episode(world, policies[0], seed=seed, reward_noise_std=2.)
            gradients = [td_gradient(episode, policy, table, .7) for policy in policies]
            for cpu, gpu in zip(*gradients):
                self.assertEqual(gpu.device.type, "cuda")
                self.assertEqual(gpu.dtype, torch.float64)
                torch.testing.assert_close(cpu, gpu.cpu(), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
