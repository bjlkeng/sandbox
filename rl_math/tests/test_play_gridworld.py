"""Check that the playable wrapper follows the same MDP and episode endings."""

import random
import unittest

from envs import Action
from play_gridworld import Game, render


class PlayTests(unittest.TestCase):
    def test_safe_route_and_terminal_moves(self):
        game = Game()
        game.reset(slip=0.0)
        for action in [Action.NORTH] + [Action.EAST] * 4 + [Action.SOUTH]:
            game.move(action)
        self.assertEqual(game.state.cell, game.world.goal)
        self.assertEqual(game.state.remaining, 6)
        self.assertEqual(game.total, 4.0)
        self.assertIn("Goal reached", render(game))
        before = (game.state, game.total, game.last)
        game.move(Action.WEST)
        self.assertEqual((game.state, game.total, game.last), before)

    def test_hazards_and_horizon_then_reset(self):
        game = Game()
        game.reset(slip=0.0)
        game.move(Action.EAST)
        self.assertEqual(game.total, -6.0)
        self.assertIn("reward -6", game.last)
        for _ in range(11):
            game.move(Action.SOUTH)  # Boundary keeps us on the hazard.
        self.assertEqual(game.state.remaining, 0)
        self.assertEqual(game.total, -72.0)
        self.assertIn("Time is up", render(game))
        game.reset(slip=0.2)
        self.assertEqual(game.state, game.world.start_state)
        self.assertEqual(game.total, 0.0)
        self.assertEqual(game.world.slip, 0.2)

    def test_slip_draws_follow_model_and_can_be_reproduced(self):
        games = [Game(random.Random(7)), Game(random.Random(7))]
        for action in [Action.NORTH, Action.EAST, Action.EAST, Action.SOUTH]:
            for game in games:
                outcomes = game.world.transitions(game.state, action)
                previous_total = game.total
                game.move(action)
                self.assertIn((game.state, game.total - previous_total),
                              [(o.next_state, o.reward) for o in outcomes])
            self.assertEqual((games[0].state, games[0].total),
                             (games[1].state, games[1].total))


if __name__ == "__main__":
    unittest.main()
