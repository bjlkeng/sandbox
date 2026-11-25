"""Paired independent batches, whole-episode averaging, and raw-data replay."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common.returns import discounted_returns
from gridworld.baselines import fit_monte_carlo_baseline
from gridworld.gradient_probe import collect_gradient_batches, main, run_short_probe, save_probe, summarize_probe
from gridworld.model import Action, Gridworld, State
from gridworld.policy import TabularSoftmaxPolicy
from gridworld.sampling import sample_episode


def fixture(device="cpu"):
    world = Gridworld(layout=("SG",), horizon=3, slip=0.)
    # Reverse the usual row order to catch a reference/estimate ordering mismatch.
    states = list(reversed([s for s in world.states() if not world.is_terminal(s)]))
    policy = TabularSoftmaxPolicy({s: np.log([1., 2., 3., 4.]) for s in states}, device=device)
    training = [sample_episode(world, policy, seed=s, reward_noise_std=2.)
                for s in range(1000, 1008)]
    baseline = fit_monte_carlo_baseline(training, gamma=.5, device=device)
    return world, policy, baseline


class GradientProbeTests(unittest.TestCase):
    def test_batch_means_match_scores_with_unequal_episode_lengths(self):
        world, policy, baseline = fixture()
        seeds = [[0, 1, 2], [3, 4, 5]]
        parameters = tuple(policy.logits.values())
        originals = [p.detach().clone() for p in parameters]
        for parameter in parameters:
            parameter.grad = torch.full_like(parameter, 7.)
        baseline_snapshot = (dict(baseline.values), dict(baseline.counts))
        imperfect = {s: p.new_tensor(3. + s.remaining, requires_grad=True)
                     for s, p in policy.logits.items()}
        run = collect_gradient_batches(world, policy, batch_seeds=seeds,
                                       baseline=baseline, gamma=.5, reward_noise_std=2.,
                                       imperfect_critic=imperfect)
        # Independent score calculation: (one_hot(action) - pi) at each visited row.
        # Average whole episode sums, including zeros at unvisited state/time rows.
        states = list(policy.logits)
        pi = np.array([.1, .2, .3, .4])
        lengths = []
        exact_values = run["baselines"]["exact_v"]["values"]
        for b, batch in enumerate(seeds):
            expected = {name: [] for name in run["batch_estimates"]}
            for seed in batch:
                episode = sample_episode(world, policy, seed=seed, reward_noise_std=2.)
                lengths.append(len(episode.steps))
                returns = discounted_returns(episode.rewards(), .5)
                gradients = {name: np.zeros((len(states), 4)) for name in expected}
                for step, target in zip(episode.steps, returns):
                    row = states.index(step.state)
                    score = np.eye(4)[tuple(Action).index(step.action)] - pi
                    weights = {
                        "full_return": returns[0],
                        "reward_to_go": .5**step.t * target,
                        "exact_v": .5**step.t * (target - exact_values[row]),
                        "learned_mc": .5**step.t * (target - baseline(step.state)),
                        "imperfect_v": .5**step.t * (target - imperfect[step.state].item()),
                    }
                    for name, table in (("td_exact", dict(zip(states, exact_values))),
                                        ("td_imperfect", {s: v.item() for s, v in imperfect.items()})):
                        future = 0. if step.terminated else table[step.next_state]
                        weights[name] = .5**step.t * (step.reward + .5*future - table[step.state])
                    for name, weight in weights.items():
                        gradients[name][row] += weight * score
                for name in expected:
                    expected[name].append(gradients[name].ravel())
            for name in expected:
                np.testing.assert_allclose(run["batch_estimates"][name][b],
                                           np.mean(expected[name], axis=0), atol=1e-14)
        self.assertGreater(len(set(lengths)), 1)
        self.assertEqual(run["sampling"]["transitions_per_batch"],
                         [sum(lengths[:3]), sum(lengths[3:])])
        self.assertEqual(run["sampling"]["batch_seeds"], seeds)
        self.assertEqual(run["sampling"]["episodes_per_batch"], 3)
        self.assertEqual(run["policy"]["rows"][0]["remaining"], states[0].remaining)
        for actual, original in zip(parameters, originals):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
            torch.testing.assert_close(actual.grad, torch.full_like(actual, 7.), rtol=0, atol=0)
        self.assertEqual((dict(baseline.values), dict(baseline.counts)), baseline_snapshot)
        self.assertEqual(run["critics"]["td_imperfect"]["values"],
                         [imperfect[s].item() for s in states])
        self.assertEqual(run["baselines"]["imperfect_v"]["values"],
                         run["critics"]["td_imperfect"]["values"])
        for s, value in imperfect.items():
            self.assertEqual(value.item(), 3. + s.remaining)
            self.assertIsNone(value.grad)

    def test_raw_json_reproduces_summary_without_sampling_or_fitting(self):
        world, policy, baseline = fixture()
        imperfect = {s: 3. + s.remaining for s in policy.logits}
        run = collect_gradient_batches(world, policy, batch_seeds=[[10, 11], [12, 13]],
                                       baseline=baseline, gamma=.5, reward_noise_std=2.,
                                       imperfect_critic=imperfect)
        replay = collect_gradient_batches(world, policy, batch_seeds=[[10, 11], [12, 13]],
                                          baseline=baseline, gamma=.5, reward_noise_std=2.,
                                          imperfect_critic=imperfect)
        self.assertEqual(run["batch_estimates"], replay["batch_estimates"])
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "probe"
            save_probe(run, directory)
            expected = json.loads((directory / "summary.json").read_text())
            with patch("gridworld.gradient_probe.sample_episode", side_effect=AssertionError), \
                    patch("gridworld.gradient_probe.differentiate_policy", side_effect=AssertionError), \
                    patch("gridworld.gradient_probe.fit_monte_carlo_baseline", side_effect=AssertionError):
                output = StringIO()
                with redirect_stdout(output):
                    main(["summarize", str(directory / "probe.json")])
                self.assertEqual(json.loads(output.getvalue()), expected)
            original = (directory / "probe.json").read_bytes()
            with self.assertRaises(FileExistsError):
                save_probe(run, directory)
            self.assertEqual((directory / "probe.json").read_bytes(), original)

    def test_reject_dependent_or_unequal_batches_and_mismatched_discount(self):
        world, policy, baseline = fixture()
        for seeds in ([], [[0]], [[], []], [[0], [1, 2]], [[0, 0], [1, 2]],
                      [[0, 1], [1, 2]], [[0], [1000]], [[-1], [2]], [[True], [2]]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                collect_gradient_batches(world, policy, batch_seeds=seeds, baseline=baseline,
                                         gamma=.5, reward_noise_std=2.)
        with self.assertRaisesRegex(ValueError, "gamma"):
            collect_gradient_batches(world, policy, batch_seeds=[[0], [1]], baseline=baseline,
                                     gamma=.9, reward_noise_std=2.)

    def test_critic_requires_finite_scalar_for_each_nonterminal_state(self):
        world, policy, baseline = fixture()
        for table in ({}, {s: [1.] for s in policy.logits}, {s: np.nan for s in policy.logits}):
            with self.subTest(table=table), self.assertRaisesRegex(ValueError, "imperfect_critic"):
                collect_gradient_batches(world, policy, batch_seeds=[[0], [1]], baseline=baseline,
                                         gamma=.5, reward_noise_std=2., imperfect_critic=table)

    def test_short_td_probe_adds_methods_without_changing_mc_samples_or_fit(self):
        mc = run_short_probe(device="cpu")
        td = run_short_probe(device="cpu", include_td=True)
        self.assertEqual(len(td["batch_estimates"]), 7)
        for name, estimates in mc["batch_estimates"].items():
            self.assertEqual(estimates, td["batch_estimates"][name])
            self.assertEqual(mc["baselines"][name], td["baselines"][name])
        self.assertEqual(mc["sampling"], td["sampling"])
        np.testing.assert_allclose(td["exact_gradient"], mc["exact_gradient"], atol=0.)
        difference = np.array(td["critics"]["td_imperfect"]["values"]) - td["critics"]["td_exact"]["values"]
        states = [State(tuple(row["cell"]), row["remaining"]) for row in td["policy"]["rows"]]
        expected = np.zeros(len(states))
        expected[states.index(State((0, 1), 1))] = 4.
        np.testing.assert_allclose(difference, expected, atol=1e-14)
        self.assertEqual(td["critics"]["td_imperfect"]["training_transitions"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device/driver unavailable")
    def test_cuda_batches_agree_with_cpu_reference(self):
        runs = []
        for device in ("cpu", "cuda"):
            world, policy, baseline = fixture(device)
            run = collect_gradient_batches(world, policy, batch_seeds=[[0, 1], [2, 3]],
                                           baseline=baseline, gamma=.5, reward_noise_std=2.,
                                           imperfect_critic={s: 3. + s.remaining for s in policy.logits})
            self.assertTrue(all(p.device.type == device for p in policy.logits.values()))
            self.assertTrue(run["device"].startswith(device))
            runs.append(run)
        np.testing.assert_allclose(runs[0]["exact_gradient"], runs[1]["exact_gradient"], atol=1e-13)
        for name in runs[0]["batch_estimates"]:
            np.testing.assert_allclose(runs[0]["batch_estimates"][name],
                                       runs[1]["batch_estimates"][name], atol=1e-13)
            self.assertAlmostEqual(summarize_probe(runs[0])[name]["mse"],
                                   summarize_probe(runs[1])[name]["mse"], places=12)


if __name__ == "__main__":
    unittest.main()
