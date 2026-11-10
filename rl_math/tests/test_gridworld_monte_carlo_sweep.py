"""Declared budgets, a single frozen fit, and reports rebuilt from raw batches."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gridworld.baselines import fit_monte_carlo_baseline
from gridworld.monte_carlo_sweep import main, plot_sweep, run_sweep, save_sweep, summarize_sweep


class MonteCarloSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("gridworld.monte_carlo_sweep.fit_monte_carlo_baseline",
                   wraps=fit_monte_carlo_baseline) as fit:
            cls.raw = run_sweep(device="cpu", pilot=True)
            cls.fit_calls = fit.call_count
        cls.summary = summarize_sweep(cls.raw)

    def test_pilot_has_disjoint_equal_budgets_and_one_unchanged_fit(self):
        self.assertEqual(self.fit_calls, 1)
        self.assertEqual(len(self.raw["runs"]), 4)
        self.assertEqual(self.raw["settings"]["batch_count"], 8)
        seed_list = []
        reference = self.raw["runs"][0]
        for run in self.raw["runs"]:
            self.assertEqual(run["policy"], reference["policy"])
            self.assertEqual(run["exact_gradient"], reference["exact_gradient"])
            self.assertEqual(run["baselines"], reference["baselines"])
            for batch in run["sampling"]["batch_seeds"]:
                self.assertEqual(len(batch), run["sampling"]["episodes_per_batch"])
                seed_list.extend(batch)
            for vectors in run["batch_estimates"].values():
                self.assertEqual(np.shape(vectors), (8, 16))
        self.assertEqual(len(seed_list), len(set(seed_list)))
        self.assertTrue(set(seed_list).isdisjoint(range(1000, 1032)))
        self.assertEqual(self.raw["budget"]["probe_episodes"], 144)
        self.assertEqual(self.raw["budget"]["probe_transitions"], 288)
        self.assertEqual(self.raw["budget"]["training_transitions"], 64)
        self.assertEqual(self.raw["budget"]["total_transitions"], 352)

    def test_report_replays_raw_json_without_new_data_or_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, report_dir = Path(tmp) / "run", Path(tmp) / "report"
            save_sweep(self.raw, run_dir)
            expected = (run_dir / "summary.json").read_bytes()
            with patch("gridworld.monte_carlo_sweep.run_sweep", side_effect=AssertionError), \
                    patch("gridworld.monte_carlo_sweep.collect_gradient_batches", side_effect=AssertionError), \
                    patch("gridworld.monte_carlo_sweep.fit_monte_carlo_baseline", side_effect=AssertionError):
                main(["report", str(run_dir / "sweep.json"), "--output-dir", str(report_dir)])
            self.assertEqual((report_dir / "summary.json").read_bytes(), expected)
            for name in ("metrics.png", "paired_mse.png"):
                self.assertEqual((run_dir / name).read_bytes(), (report_dir / name).read_bytes())
            original = (run_dir / "sweep.json").read_bytes()
            with self.assertRaises(FileExistsError):
                save_sweep(self.raw, run_dir)
            self.assertEqual((run_dir / "sweep.json").read_bytes(), original)

    def test_plot_uses_saved_points_and_bias_coordinate(self):
        figures = plot_sweep(self.summary)
        self.assertEqual(set(figures), {"metrics.png", "paired_mse.png"})
        metrics = figures["metrics.png"]
        self.assertEqual(len(metrics.axes), 6)  # Three measures, two pilot noise levels.
        first = self.summary["conditions"][0]
        index = self.summary["bias_coordinate"]["index"]
        self.assertEqual(self.summary["bias_coordinate"]["action"], "EAST")
        self.assertEqual(self.summary["bias_coordinate"]["cell"], [0, 1])
        # First artist is the full-return signed bias (including its zero reference).
        self.assertAlmostEqual(metrics.axes[0].lines[0].get_ydata()[0],
                               first["comparison"]["estimators"]["full_return"]["metrics"]["empirical_bias"][index])
        for fig in figures.values():
            plt.close(fig)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device/driver unavailable")
    def test_cuda_pilot_matches_cpu(self):
        gpu = run_sweep(device="cuda", pilot=True)
        self.assertTrue(all(run["device"].startswith("cuda") for run in gpu["runs"]))
        for cpu_run, gpu_run in zip(self.raw["runs"], gpu["runs"]):
            np.testing.assert_allclose(cpu_run["exact_gradient"], gpu_run["exact_gradient"], atol=1e-13)
            for name in cpu_run["batch_estimates"]:
                np.testing.assert_allclose(cpu_run["batch_estimates"][name],
                                           gpu_run["batch_estimates"][name], atol=1e-13)


if __name__ == "__main__":
    unittest.main()
