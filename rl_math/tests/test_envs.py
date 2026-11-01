"""Hand-checkable finite-horizon MDP examples; no sampling is involved."""

import math
import unittest

from envs import Action, Gridworld, State, Transition
from returns import discounted_returns


class GridworldTests(unittest.TestCase):
    def test_corner_slips_merge_blocked_moves_and_keep_expected_reward(self):
        world = Gridworld(layout=("SHG", ".#."), horizon=2, slip=0.3)
        # EAST succeeds with probability .7. NORTH/WEST hit the boundary;
        # SOUTH succeeds. Every move costs 1; entering H costs another 5.
        outcomes = {outcome.next_state: outcome for outcome in
                    world.transitions(world.start_state, Action.EAST)}
        self.assertEqual(set(outcomes), {State((0, 1), 1), State((0, 0), 1),
                                         State((1, 0), 1)})
        for state, probability, reward in [
            (State((0, 1), 1), 0.7, -6.0),
            (State((0, 0), 1), 0.2, -1.0),
            (State((1, 0), 1), 0.1, -1.0),
        ]:
            with self.subTest(state=state):
                self.assertAlmostEqual(outcomes[state].probability, probability)
                self.assertEqual(outcomes[state].reward, reward)
        self.assertAlmostEqual(world.expected_reward(world.start_state, Action.EAST), -4.5)

    def test_wall_and_boundary_consume_a_step(self):
        world = Gridworld(layout=("S#G", "..."), horizon=3, slip=0.0)
        for action in (Action.NORTH, Action.EAST, Action.WEST):
            with self.subTest(action=action):
                self.assertEqual(world.transitions(world.start_state, action),
                                 (Transition(1.0, State((0, 0), 2), -1.0),))
        self.assertEqual(world.transitions(world.start_state, Action.SOUTH),
                         (Transition(1.0, State((1, 0), 2), -1.0),))

    def test_goal_reward_is_paid_once_and_hazards_are_nonterminal(self):
        world = Gridworld(layout=("SHG",), horizon=4, slip=0.0,
                          step_reward=-2.0, hazard_penalty=-3.0, goal_reward=8.0)
        hazard, = world.transitions(world.start_state, Action.EAST)
        self.assertEqual(hazard, Transition(1.0, State((0, 1), 3), -5.0))
        self.assertFalse(world.is_terminal(hazard.next_state))
        # Remaining on a hazard still incurs the destination-cell penalty.
        self.assertEqual(world.expected_reward(hazard.next_state, Action.NORTH), -5.0)
        goal, = world.transitions(hazard.next_state, Action.EAST)
        self.assertEqual(goal, Transition(1.0, State((0, 2), 2), 6.0))
        self.assertTrue(world.is_terminal(goal.next_state))
        for action in Action:
            self.assertEqual(world.transitions(goal.next_state, action),
                             (Transition(1.0, goal.next_state, 0.0),))

    def test_same_cell_with_different_remaining_time_has_different_future(self):
        world = Gridworld(layout=("S.G",), horizon=10, slip=0.0)
        last_step, = world.transitions(State(world.start, 1), Action.EAST)
        more_time, = world.transitions(State(world.start, 10), Action.EAST)
        self.assertEqual(last_step, Transition(1.0, State((0, 1), 0), -1.0))
        self.assertEqual(more_time, Transition(1.0, State((0, 1), 9), -1.0))
        self.assertTrue(world.is_terminal(last_step.next_state))
        self.assertFalse(world.is_terminal(more_time.next_state))
        self.assertEqual(world.transitions(last_step.next_state, Action.EAST),
                         (Transition(1.0, last_step.next_state, 0.0),))
        self.assertEqual(world.expected_reward(more_time.next_state, Action.EAST), 9.0)
        # A goal reached on the last allowed move still pays its reward.
        self.assertEqual(world.transitions(State((0, 1), 1), Action.EAST),
                         (Transition(1.0, State(world.goal, 0), 9.0),))

    def test_short_hazardous_and_long_safe_routes_on_default_map(self):
        world = Gridworld(slip=0.0)
        paths = [
            ([Action.EAST] * 4, [-6.0, -6.0, -6.0, 9.0], -9.0),
            ([Action.NORTH] + [Action.EAST] * 4 + [Action.SOUTH],
             [-1.0] * 5 + [9.0], 4.0),
        ]
        for actions, expected_rewards, expected_return in paths:
            with self.subTest(actions=actions):
                state = world.start_state
                rewards = []
                for action in actions:
                    self.assertFalse(world.is_terminal(state))
                    outcome, = world.transitions(state, action)
                    self.assertEqual(outcome.probability, 1.0)
                    rewards.append(outcome.reward)
                    state = outcome.next_state
                self.assertEqual(state.cell, world.goal)
                self.assertTrue(world.is_terminal(state))
                self.assertEqual(rewards, expected_rewards)
                self.assertEqual(discounted_returns(rewards)[0], expected_return)

    def test_every_state_action_is_a_normalized_closed_distribution(self):
        for slip in (0.0, 0.3, 1.0):
            world = Gridworld(layout=("SHG", ".#."), horizon=3, slip=slip)
            states = set(world.states())
            self.assertEqual(len(states), 5 * 4)  # Five cells, times 0..3.
            self.assertIn(world.start_state, states)
            for state in states:
                for action in Action:
                    with self.subTest(slip=slip, state=state, action=action):
                        outcomes = world.transitions(state, action)
                        self.assertAlmostEqual(sum(o.probability for o in outcomes), 1.0)
                        self.assertEqual(len(outcomes), len({o.next_state for o in outcomes}))
                        for outcome in outcomes:
                            self.assertGreater(outcome.probability, 0.0)
                            self.assertIn(outcome.next_state, states)
                            expected_time = (
                                state.remaining if world.is_terminal(state) else state.remaining - 1
                            )
                            self.assertEqual(outcome.next_state.remaining, expected_time)

    def test_full_slip_excludes_the_intended_direction(self):
        world = Gridworld(layout=("S..", "...", "..G"), slip=1.0)
        outcomes = world.transitions(State((1, 1), 2), Action.NORTH)
        self.assertEqual({o.next_state.cell for o in outcomes}, {(1, 2), (2, 1), (1, 0)})
        for outcome in outcomes:
            self.assertAlmostEqual(outcome.probability, 1 / 3)

    def test_invalid_model_configuration(self):
        for layout in ((), ("",), ("SG", "."), ("S?G",), ("..G",),
                       ("S..",), ("SSG",), ("SGG",)):
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                Gridworld(layout=layout)
        for horizon in (0, -1, 1.5, True):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                Gridworld(horizon=horizon)
        for slip in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(slip=slip), self.assertRaises(ValueError):
                Gridworld(slip=slip)
        for name in ("step_reward", "goal_reward", "hazard_penalty"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    Gridworld(**{name: value})

    def test_invalid_state_or_action(self):
        world = Gridworld(layout=("S#G", "..."), horizon=3)
        for state in (State((-1, 0), 1), State((0, 3), 1), State((0, 1), 1),
                      State((0.0, 0), 1), State((False, 0), 1),
                      State((0, 0), -1), State((0, 0), 4), State((0, 0), 1.5)):
            with self.subTest(state=state), self.assertRaises(ValueError):
                world.transitions(state, Action.NORTH)
        with self.assertRaises(ValueError):
            world.transitions(world.start_state, "north")


if __name__ == "__main__":
    unittest.main()
