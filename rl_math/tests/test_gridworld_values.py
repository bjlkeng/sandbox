"""Hand calculations, path enumeration, and fixed-policy Bellman identities."""

import unittest

import numpy as np
import torch

from gridworld.model import Action, Gridworld, State
from gridworld.policy import TabularSoftmaxPolicy
from gridworld.values import evaluate_policy


def worked_policy(world):
    return TabularSoftmaxPolicy({
        s: np.log([1, 2, 3, 4]) for s in world.states() if not world.is_terminal(s)
    }, device="cpu")


class ExactValueTests(unittest.TestCase):
    def test_two_step_hand_calculation(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0)
        values = evaluate_policy(world, worked_policy(world), gamma=0.5)
        # pi=(.1,.2,.3,.4), gamma=.5, action order NORTH/EAST/SOUTH/WEST.
        expected = [
            (State((0, 0), 1), [-1, -6, -1, -1], -2, [1, -4, 1, 1]),
            (State((0, 1), 1), [-6, 9, -6, -1], -1, [-5, 10, -5, 0]),
            (State((0, 0), 2), [-2, -6.5, -2, -2], -2.9, [.9, -3.6, .9, .9]),
            (State((0, 1), 2), [-6.5, 9, -6.5, -2], -1.6, [-4.9, 10.6, -4.9, -.4]),
        ]
        for state, q, v, advantage in expected:
            with self.subTest(state=state):
                torch.testing.assert_close(values.q[state], torch.tensor(q, dtype=torch.float64))
                self.assertAlmostEqual(values.v[state].item(), v)
                torch.testing.assert_close(values.advantage[state],
                                           torch.tensor(advantage, dtype=torch.float64))
        self.assertAlmostEqual(values.start_return.item(), -2.9)
        # The maximum would give -2 here, evaluating a different decision rule.
        self.assertNotAlmostEqual(values.v[world.start_state].item(),
                                  values.q[world.start_state].max().item())

    def test_terminal_values_are_zero_without_terminal_policy_rows(self):
        world = Gridworld(layout=("SG",), horizon=2, slip=0)
        policy = worked_policy(world)
        values = evaluate_policy(world, policy, gamma=1)
        self.assertEqual(set(values.v), set(world.states()))
        self.assertEqual(set(values.q), set(values.v))
        self.assertEqual(set(values.advantage), set(values.v))
        for state in world.states():
            if world.is_terminal(state):
                self.assertNotIn(state, policy.logits)
                self.assertEqual(values.v[state].item(), 0)
                torch.testing.assert_close(values.q[state], torch.zeros(4, dtype=torch.float64))
                torch.testing.assert_close(values.advantage[state], torch.zeros(4, dtype=torch.float64))
                self.assertFalse(values.v[state].requires_grad)
        # Goal reward is still paid when reaching it on the final allowed move.
        self.assertEqual(values.q[State(world.start, 1)][1].item(), 9)

    def test_zero_discount_and_undiscounted_return(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0)
        for gamma in (0, 1):
            with self.subTest(gamma=gamma):
                values = evaluate_policy(world, worked_policy(world), gamma=gamma)
                self.assertAlmostEqual(values.start_return.item(), -2 - 1.8 * gamma)
                self.assertAlmostEqual(values.v[State((0, 1), 2)].item(), -1 - 1.2 * gamma)

    def test_policy_can_change_with_remaining_time(self):
        world = Gridworld(layout=("S.G",), horizon=2, slip=0)
        # First go east; with one step left choose north, even next to the goal.
        policy = TabularSoftmaxPolicy({
            s: (-1000, 0, -1000, -1000) if s.remaining == 2 else (0, -1000, -1000, -1000)
            for s in world.states() if not world.is_terminal(s)
        }, device="cpu")
        values = evaluate_policy(world, policy, gamma=0.5)
        self.assertEqual(values.start_return.item(), -1.5)
        self.assertEqual(values.v[State((0, 1), 1)].item(), -1)
        self.assertEqual(values.q[State((0, 1), 1)][1].item(), 9)

    def test_stochastic_values_match_complete_path_enumeration(self):
        world = Gridworld(layout=("SHG", ".#."), horizon=3, slip=0.3)
        policy = TabularSoftmaxPolicy({
            s: np.log([1, 2 + s.remaining, 3, 4])
            for s in world.states() if not world.is_terminal(s)
        }, device="cpu")
        gamma = 0.7
        values = evaluate_policy(world, policy, gamma=gamma)

        def paths(state):
            # Enumerate complete reward sequences, without a value recurrence/cache.
            if world.is_terminal(state):
                yield 1.0, ()
                return
            for action, action_probability in zip(Action, policy.probabilities(state).detach().tolist()):
                for outcome in world.transitions(state, action):
                    for tail_probability, rewards in paths(outcome.next_state):
                        yield (action_probability * outcome.probability * tail_probability,
                               (outcome.reward,) + rewards)

        for state in world.states():
            expected = sum(probability * np.dot(gamma ** np.arange(len(rewards)), rewards)
                           for probability, rewards in paths(state))
            with self.subTest(state=state):
                self.assertAlmostEqual(values.v[state].item(), expected)
        self.assertEqual(values.start_return.item(), values.v[world.start_state].item())

    def test_bellman_expectation_and_zero_mean_advantage_on_reachable_states(self):
        world = Gridworld(horizon=4)
        policy = worked_policy(world)
        gamma = 0.9
        values = evaluate_policy(world, policy, gamma=gamma)
        pending, visited = [world.start_state], set()
        while pending:
            state = pending.pop()
            if state in visited:
                continue
            visited.add(state)
            if world.is_terminal(state):
                continue
            probabilities = policy.probabilities(state)
            self.assertAlmostEqual(values.v[state].item(), torch.dot(probabilities, values.q[state]).item())
            self.assertAlmostEqual(torch.dot(probabilities, values.advantage[state]).item(), 0)
            for index, action in enumerate(Action):
                expected_q = 0.0
                for outcome in world.transitions(state, action):
                    successor = outcome.next_state
                    # Explicitly average the next action under the same policy.
                    future = (0 if world.is_terminal(successor) else
                              torch.dot(policy.probabilities(successor), values.q[successor]).item())
                    expected_q += outcome.probability * (outcome.reward + gamma * future)
                    pending.append(successor)
                self.assertAlmostEqual(values.q[state][index].item(), expected_q)
        self.assertGreater(len(visited), 1)

    def test_return_graph_reaches_logits_at_earlier_and_later_times(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0)
        policy = worked_policy(world)
        values = evaluate_policy(world, policy, gamma=0.5)
        self.assertEqual(values.start_return.dtype, torch.float64)
        # Direct autograd use verifies the evaluator itself retains the graph.
        # The H/1 derivative requires the dependency through successor values.
        early, late = world.start_state, State((0, 1), 1)
        gradients = torch.autograd.grad(values.start_return, (policy.logits[early], policy.logits[late]))
        for actual, expected in zip(gradients, ([.09, -.72, .27, .36], [-.05, .2, -.15, 0])):
            torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64),
                                       rtol=1e-12, atol=1e-14)

    def test_invalid_discount(self):
        world = Gridworld()
        for gamma in (-0.1, 1.1, np.nan, np.inf, -np.inf):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                evaluate_policy(world, worked_policy(world), gamma=gamma)


if __name__ == "__main__":
    unittest.main()
