"""Exact logit derivatives checked by hand and central finite differences."""

import unittest

import numpy as np
import torch

from gridworld.gradients import differentiate_policy
from gridworld.model import Gridworld, State
from gridworld.policy import TabularSoftmaxPolicy
from gridworld.values import evaluate_policy


def worked_policy(world):
    return TabularSoftmaxPolicy({
        s: np.log([1, 2, 3, 4]) for s in world.states() if not world.is_terminal(s)
    })


class ExactGradientTests(unittest.TestCase):
    def test_two_step_hand_calculation_retains_episode_discount(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0)
        result = differentiate_policy(world, worked_policy(world), gamma=0.5)
        self.assertAlmostEqual(result.values.start_return.item(), -2.9)
        # w(s) = gamma**t * Pr(s_t=s), with t measured from episode start.
        expected = [
            (State(world.start, 2), [.09, -.72, .27, .36]),
            (State(world.start, 1), [.04, -.32, .12, .16]),
            (State((0, 1), 1), [-.05, .2, -.15, 0]),
            (State((0, 1), 2), [0, 0, 0, 0]),
        ]
        for state, gradient in expected:
            with self.subTest(state=state):
                torch.testing.assert_close(result.gradient[state],
                                           torch.tensor(gradient, dtype=torch.float64),
                                           rtol=1e-12, atol=1e-14)

    def test_stochastic_time_dependent_policy_matches_finite_differences(self):
        world = Gridworld(layout=("SHG", ".#."), horizon=3, slip=0.3)
        policy = TabularSoftmaxPolicy({
            s: [.2 * s.remaining, -.3 * (s.cell[0] + 1), .4 * (s.cell[1] - 1), -.1 * s.remaining]
            for s in world.states() if not world.is_terminal(s)
        })
        epsilon = 1e-5
        for gamma in (0, 0.7, 1):
            result = differentiate_policy(world, policy, gamma=gamma)
            for state in policy.logits:
                numerical = torch.zeros(4, dtype=torch.float64)
                for action_index in range(4):
                    returns = []
                    for sign in (1, -1):
                        perturbed = {s: row.detach().clone() for s, row in policy.logits.items()}
                        perturbed[state][action_index] += sign * epsilon
                        with torch.no_grad():
                            returns.append(evaluate_policy(
                                world, TabularSoftmaxPolicy(perturbed), gamma=gamma
                            ).start_return.item())
                    numerical[action_index] = (returns[0] - returns[1]) / (2 * epsilon)
                with self.subTest(gamma=gamma, state=state):
                    torch.testing.assert_close(
                        result.gradient[state], numerical, rtol=1e-6, atol=1e-8
                    )

    def test_values_agree_with_evaluator_and_parameters_and_grad_buffers_are_unchanged(self):
        world = Gridworld()
        policy = worked_policy(world)
        original = {s: row.detach().clone() for s, row in policy.logits.items()}
        # Preserve pre-existing .grad buffers as well as unset ones.
        policy.logits[world.start_state].grad = torch.ones(4, dtype=torch.float64)
        result = differentiate_policy(world, policy, gamma=0.9)
        expected = evaluate_policy(world, policy, gamma=0.9)
        torch.testing.assert_close(result.values.start_return, expected.start_return)
        self.assertEqual(set(result.gradient), set(policy.logits))
        for state in expected.v:
            torch.testing.assert_close(result.values.v[state], expected.v[state])
            torch.testing.assert_close(result.values.q[state], expected.q[state])
            torch.testing.assert_close(result.values.advantage[state], expected.advantage[state])
        for state in policy.logits:
            torch.testing.assert_close(policy.logits[state], original[state], rtol=0, atol=0)
            self.assertFalse(result.gradient[state].requires_grad)
            if state == world.start_state:
                torch.testing.assert_close(policy.logits[state].grad, torch.ones(4, dtype=torch.float64))
            else:
                self.assertIsNone(policy.logits[state].grad)
        # A new evaluation permits repeated probes without gradient accumulation.
        again = differentiate_policy(world, policy, gamma=0.9)
        for state in result.gradient:
            torch.testing.assert_close(again.gradient[state], result.gradient[state], rtol=0, atol=0)

    def test_zero_discount_removes_all_later_gradients(self):
        world = Gridworld(layout=("SHG",), horizon=3, slip=0)
        result = differentiate_policy(world, worked_policy(world), gamma=0)
        torch.testing.assert_close(result.gradient[world.start_state],
                                   torch.tensor([.1, -.8, .3, .4], dtype=torch.float64))
        for state in result.gradient:
            if state != world.start_state:
                torch.testing.assert_close(result.gradient[state], torch.zeros(4, dtype=torch.float64),
                                           rtol=0, atol=0)

    def test_early_goal_and_unreachable_rows_have_no_gradient(self):
        world = Gridworld(layout=("SG.",), horizon=3, slip=0)
        # Include optional terminal rows: they are never evaluated or differentiated.
        policy = TabularSoftmaxPolicy({s: [0, 0, 0, 0] for s in world.states()})
        result = differentiate_policy(world, policy, gamma=0.5)
        self.assertGreater(result.gradient[world.start_state][1].item(), 0)
        for state in world.states():
            if world.is_terminal(state) or state.cell == (0, 2):
                torch.testing.assert_close(result.gradient[state], torch.zeros(4, dtype=torch.float64),
                                           rtol=0, atol=0)

    def test_constant_logit_shifts_leave_return_and_gradient_unchanged(self):
        world = Gridworld(horizon=3)
        policy = worked_policy(world)
        shifted = TabularSoftmaxPolicy({s: row + 20 for s, row in policy.logits.items()})
        result = differentiate_policy(world, policy, gamma=0.8)
        other = differentiate_policy(world, shifted, gamma=0.8)
        self.assertAlmostEqual(result.values.start_return.item(), other.values.start_return.item())
        for state in result.gradient:
            # A row's constant direction changes no probabilities, so its derivative is zero.
            self.assertAlmostEqual(float(result.gradient[state].sum()), 0)
            torch.testing.assert_close(result.gradient[state], other.gradient[state], rtol=1e-12, atol=1e-14)

    def test_invalid_discount(self):
        world = Gridworld(layout=("SHG",), horizon=2)
        for gamma in (-0.1, 1.1, np.nan, np.inf, -np.inf):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                differentiate_policy(world, worked_policy(world), gamma=gamma)


if __name__ == "__main__":
    unittest.main()
