"""Check plotted coordinates and numbers against small independent examples."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gridworld.model import Action, Gridworld, State
from gridworld.policy import TabularSoftmaxPolicy
from gridworld.plotting import plot_advantages, plot_policy_values, plot_trajectories
from gridworld.sampling import sample_episode
from gridworld.values import evaluate_policy


class GridworldPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_hand_values_and_action_advantages_reach_the_artists(self):
        world = Gridworld(layout=("SHG", ".#."), horizon=2, slip=0)
        # At remaining=1, the second row cannot change these hand calculations.
        policy = TabularSoftmaxPolicy({
            s: np.log([1, 2, 3, 4]) for s in world.states() if not world.is_terminal(s)
        }, device="cpu")
        values = evaluate_policy(world, policy, gamma=.5)
        fig = plot_policy_values(world, policy, values, remaining=1, gamma=.5,
                                 policy_name="Hand example")
        image = fig.axes[0].images[0].get_array()
        # SOUTH now leaves S, but still pays -1; H has a wall to its south.
        np.testing.assert_allclose(image[0], [-2, -1, 0])
        self.assertTrue(image.mask[1, 1])
        self.assertIn("-2.00", [t.get_text() for t in fig.axes[0].texts])
        self.assertEqual(fig.axes[0].get_ylim(), (1.5, -.5))

        fig = plot_advantages(world, values, remaining=1, gamma=.5,
                              policy_name="Hand example")
        for ax, action, advantage in zip(fig.axes[:4], Action, [1, -4, 1, 1]):
            self.assertIn(action.name, ax.get_title())
            self.assertAlmostEqual(ax.images[0].get_array()[0, 0], advantage)
            self.assertIn(f"{advantage:+.2f}", [t.get_text() for t in ax.texts])
            self.assertEqual(ax.images[0].get_array()[0, 2], 0)
        self.assertTrue(all(ax.images[0].norm is fig.axes[0].images[0].norm
                            for ax in fig.axes[:4]))
        # Plotting detaches only at the display boundary; the evaluator survives.
        grad, = torch.autograd.grad(values.start_return, (policy.logits[world.start_state],))
        self.assertGreater(grad.abs().sum().item(), 0)
        self.assertTrue(all(row.grad is None for row in policy.logits.values()))

    def test_requested_time_controls_policy_and_terminal_rows(self):
        world = Gridworld(layout=("SG",), horizon=2, slip=0)
        policy = TabularSoftmaxPolicy({
            s: np.log([1, 2, 3, 4] if s.remaining == 1 else [4, 3, 2, 1])
            for s in world.states() if not world.is_terminal(s)
        }, device="cpu")
        values = evaluate_policy(world, policy)
        for remaining, north_probability in ((1, "0.10"), (2, "0.40")):
            fig = plot_policy_values(world, policy, values, remaining=remaining, gamma=1,
                                     policy_name="Time-dependent")
            # NORTH is above S: this detects a swapped row/column or wrong time.
            text = next(t for t in fig.axes[1].texts if t.get_position() == (0, -.36))
            self.assertEqual(text.get_text(), north_probability)
        fig = plot_policy_values(world, policy, values, remaining=0, gamma=1,
                                 policy_name="Time-dependent")
        np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), [[0, 0]])
        self.assertEqual([t.get_text() for t in fig.axes[1].texts].count("terminal"), 2)
        with self.assertRaises(ValueError):
            plot_policy_values(world, policy, values, remaining=3, gamma=1,
                               policy_name="Time-dependent")

    def test_paths_use_observed_successors_including_stays_and_slip(self):
        world = Gridworld(layout=("S.G",), horizon=4, slip=1)
        policy = TabularSoftmaxPolicy({
            s: [0, -1000, -1000, -1000]
            for s in world.states() if not world.is_terminal(s)
        }, device="cpu")
        episode = sample_episode(world, policy, seed=0)
        cells = [world.start] + [step.next_state.cell for step in episode.steps]
        self.assertTrue(any(a == b for a, b in zip(cells, cells[1:])))
        self.assertTrue(any(a != b for a, b in zip(cells, cells[1:])))
        self.assertTrue(all(step.action == Action.NORTH for step in episode.steps))
        fig = plot_trajectories(world, [episode], gamma=.9, policy_name="Always NORTH")
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), [c[1] for c in cells])
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [c[0] for c in cells])
        self.assertIn("seed=0", fig.axes[0].get_title())


if __name__ == "__main__":
    unittest.main()
