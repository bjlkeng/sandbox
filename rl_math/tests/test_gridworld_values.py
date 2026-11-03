"""Hand calculations, path enumeration, and fixed-policy Bellman identities."""

import unittest

import numpy as np

from gridworld.model import Action, Gridworld, State
from gridworld.policy import TabularSoftmaxPolicy
from gridworld.values import evaluate_policy


def worked_policy(world):
    return TabularSoftmaxPolicy({
        s: np.log([1, 2, 3, 4]) for s in world.states() if not world.is_terminal(s)
    })


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
                np.testing.assert_allclose(values.q[state], q)
                self.assertAlmostEqual(values.v[state], v)
                np.testing.assert_allclose(values.advantage[state], advantage, atol=1e-14)
        self.assertAlmostEqual(values.start_return, -2.9)
        # The maximum would give -2 here, evaluating a different decision rule.
        self.assertNotAlmostEqual(values.v[world.start_state], max(values.q[world.start_state]))

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
                self.assertEqual(values.v[state], 0)
                np.testing.assert_array_equal(values.q[state], np.zeros(4))
                np.testing.assert_array_equal(values.advantage[state], np.zeros(4))
        # Goal reward is still paid when reaching it on the final allowed move.
        self.assertEqual(values.q[State(world.start, 1)][1], 9)

    def test_zero_discount_and_undiscounted_return(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0)
        for gamma in (0, 1):
            with self.subTest(gamma=gamma):
                values = evaluate_policy(world, worked_policy(world), gamma=gamma)
                self.assertAlmostEqual(values.start_return, -2 - 1.8 * gamma)
                self.assertAlmostEqual(values.v[State((0, 1), 2)], -1 - 1.2 * gamma)

    def test_policy_can_change_with_remaining_time(self):
        world = Gridworld(layout=("S.G",), horizon=2, slip=0)
        # First go east; with one step left choose north, even next to the goal.
        policy = TabularSoftmaxPolicy({
            s: (-1000, 0, -1000, -1000) if s.remaining == 2 else (0, -1000, -1000, -1000)
            for s in world.states() if not world.is_terminal(s)
        })
        values = evaluate_policy(world, policy, gamma=0.5)
        self.assertEqual(values.start_return, -1.5)
        self.assertEqual(values.v[State((0, 1), 1)], -1)
        self.assertEqual(values.q[State((0, 1), 1)][1], 9)

    def test_stochastic_values_match_complete_path_enumeration(self):
        world = Gridworld(layout=("SHG", ".#."), horizon=3, slip=0.3)
        policy = TabularSoftmaxPolicy({
            s: np.log([1, 2 + s.remaining, 3, 4])
            for s in world.states() if not world.is_terminal(s)
        })
        gamma = 0.7
        values = evaluate_policy(world, policy, gamma=gamma)

        def paths(state):
            # Enumerate complete reward sequences, without a value recurrence/cache.
            if world.is_terminal(state):
                yield 1.0, ()
                return
            for action, action_probability in zip(Action, policy.probabilities(state)):
                for outcome in world.transitions(state, action):
                    for tail_probability, rewards in paths(outcome.next_state):
                        yield (action_probability * outcome.probability * tail_probability,
                               (outcome.reward,) + rewards)

        for state in world.states():
            expected = sum(probability * np.dot(gamma ** np.arange(len(rewards)), rewards)
                           for probability, rewards in paths(state))
            with self.subTest(state=state):
                self.assertAlmostEqual(values.v[state], expected)
        self.assertAlmostEqual(values.start_return, values.v[world.start_state])

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
            self.assertAlmostEqual(values.v[state], np.dot(probabilities, values.q[state]))
            self.assertAlmostEqual(np.dot(probabilities, values.advantage[state]), 0)
            for index, action in enumerate(Action):
                expected_q = 0.0
                for outcome in world.transitions(state, action):
                    successor = outcome.next_state
                    # Explicitly average the next action under the same policy.
                    future = (0 if world.is_terminal(successor) else
                              np.dot(policy.probabilities(successor), values.q[successor]))
                    expected_q += outcome.probability * (outcome.reward + gamma * future)
                    pending.append(successor)
                self.assertAlmostEqual(values.q[state][index], expected_q)
        self.assertGreater(len(visited), 1)

    def test_invalid_discount(self):
        world = Gridworld()
        for gamma in (-0.1, 1.1, np.nan, np.inf, -np.inf):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                evaluate_policy(world, worked_policy(world), gamma=gamma)


if __name__ == "__main__":
    unittest.main()
