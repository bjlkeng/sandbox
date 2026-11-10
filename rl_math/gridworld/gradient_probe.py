"""One frozen-policy comparison; save raw batches and summarize without resampling.

From rl_math:
  python -m gridworld.gradient_probe run --output-dir runs/gradient-probe --device cuda
  python -m gridworld.gradient_probe summarize runs/gradient-probe/probe.json
"""

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch

from algorithms.reinforce import full_return_gradient, reward_to_go_gradient
from common.gradient_metrics import summarize_gradients
from gridworld.baselines import MonteCarloBaseline, fit_monte_carlo_baseline
from gridworld.gradients import differentiate_policy
from gridworld.model import Action, Gridworld
from gridworld.policy import TabularSoftmaxPolicy
from gridworld.sampling import sample_episode


def collect_gradient_batches(
    world: Gridworld,
    policy: TabularSoftmaxPolicy,
    *,
    batch_seeds: Sequence[Sequence[int]],
    baseline: MonteCarloBaseline,
    gamma: float,
    reward_noise_std: float,
) -> dict:
    """Compare four existing estimators using paired episodes in independent batches.

    Each row is (1/N) sum_i gradient(episode_i). Distinct seeds across all
    batches provide independent sampler streams; the same episodes within a
    batch are reused across estimators to pair the comparison. Validate seed
    separation from the learned baseline before collecting any probes. Hold
    both the supplied policy and that separately fitted baseline fixed.

    Flatten all policy rows in insertion order, then NORTH/EAST/SOUTH/WEST;
    retain zero/unvisited rows. Save this basis, the exact gradient, every batch
    vector, settings, and timing. Computation stays on the policy device until
    detached batch means are copied to CPU for serialization/statistics.
    No policy updates, refits, bootstrapping, or advantage normalization.

    Learned-baseline results are conditional on this ONE fit. Its training
    seeds/counts are recorded separately; equal probe episode budgets do not
    imply equal total sampling costs. Caller supplies the matching world/policy
    used for fitting and must not relabel training episodes as probes.
    """
    batches = [list(seeds) for seeds in batch_seeds]
    if len(batches) < 2 or not batches[0] or any(len(b) != len(batches[0]) for b in batches):
        raise ValueError("supply at least two nonempty batches with equal episode counts")
    seeds = [seed for batch in batches for seed in batch]
    if (any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
            or not set(seeds).isdisjoint(baseline.training_seeds)):
        raise ValueError("probe seeds must be distinct nonnegative integers, separate from training")
    if not 0 <= gamma <= 1 or gamma != baseline.gamma:
        raise ValueError("gamma must lie in [0, 1] and match the fitted baseline")
    if not np.isfinite(reward_noise_std) or reward_noise_std < 0:
        raise ValueError("reward_noise_std must be finite and nonnegative")

    states = tuple(policy.logits)
    parameters = tuple(policy.logits.values())
    device = parameters[0].device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    exact = differentiate_policy(world, policy, gamma)
    exact_values = {s: exact.values.v[s].detach() for s in states}
    estimates = {name: [] for name in ("full_return", "reward_to_go", "exact_v", "learned_mc")}
    transitions = []
    for batch in batches:
        episode_gradients = {name: [] for name in estimates}
        transition_count = 0
        for seed in batch:
            episode = sample_episode(world, policy, seed=seed, reward_noise_std=reward_noise_std)
            transition_count += len(episode.steps)
            rewards = episode.rewards()
            for name in estimates:
                # Each estimator consumes its graph, so evaluate fresh log pi.
                logp = episode.log_probabilities(policy.log_probabilities)
                if name == "full_return":
                    gradient = full_return_gradient(logp, rewards, parameters, gamma)
                else:
                    values = None
                    if name == "exact_v":
                        values = torch.stack([exact_values[step.state] for step in episode.steps])
                    elif name == "learned_mc":
                        values = baseline.for_episode(episode)
                    gradient = reward_to_go_gradient(logp, rewards, parameters, gamma, baselines=values)
                episode_gradients[name].append(torch.cat(gradient))
        for name in estimates:
            # Include every episode, even when it never visits a given logit row.
            estimates[name].append(torch.stack(episode_gradients[name]).mean(0).cpu().tolist())
        transitions.append(transition_count)

    result = {
        "schema_version": 1,
        "world": {"layout": list(world.layout), "horizon": world.horizon, "slip": world.slip,
                  "step_reward": world.step_reward, "goal_reward": world.goal_reward,
                  "hazard_penalty": world.hazard_penalty},
        "objective": {"name": "discounted start-state return", "gamma": gamma,
                      "exact_return": exact.values.start_return.item(),
                      "aggregation": "mean of whole episode gradient sums"},
        "policy": {
            "action_order": [action.name for action in Action],
            "gradient_order": "flatten rows in saved order, then action_order; include zero rows",
            "rows": [{"cell": list(s.cell), "remaining": s.remaining,
                      "logits": policy.logits[s].detach().cpu().tolist(),
                      "probabilities": policy.probabilities(s).detach().cpu().tolist()} for s in states],
        },
        "sampling": {"batch_count": len(batches), "episodes_per_batch": len(batches[0]),
                     "batch_seeds": batches, "reward_noise_std": reward_noise_std,
                     "transitions_per_batch": transitions, "paired_estimators": True},
        "baselines": {
            "full_return": {"method": "full_return_gradient", "baseline": None},
            "reward_to_go": {"method": "reward_to_go_gradient", "baseline": None},
            "exact_v": {"method": "reward_to_go_gradient", "baseline": "exact V at frozen policy",
                        "values": [exact_values[s].item() for s in states]},
            "learned_mc": {
                "method": "reward_to_go_gradient", "baseline": "frozen per-state mean of complete G_t",
                "values": [baseline(s) for s in states], "counts": [baseline.counts.get(s, 0) for s in states],
                "unseen_value": 0., "gamma": baseline.gamma,
                "training_seeds": list(baseline.training_seeds),
                "training_transitions": sum(baseline.counts.values()),
                "scope": "conditional on this single separately fitted table",
            },
        },
        "exact_gradient": torch.cat([exact.gradient[s] for s in states]).cpu().tolist(),
        "batch_estimates": estimates,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "device": str(device), "dtype": str(parameters[0].dtype),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    result["runtime_seconds"] = {"measurement": perf_counter() - started}
    return result


def run_short_probe(*, device: str = "cuda") -> dict:
    """Fixed pilot: SHG, 32 batches x 8 probes, one separate 32-episode fit.

    Reuse the worked example's settings: horizon 2, no slip, gamma .5, sigma 2,
    pi=(.1,.2,.3,.4). Probes use seeds 0..255; training uses 1000..1031. Fit only
    once, with the same world, frozen policy, discount, and noise as the probes.
    The fit costs 64 additional transitions beyond the 512 shared probe steps.
    """
    started = perf_counter()
    world = Gridworld(layout=("SHG",), horizon=2, slip=0.)
    policy = TabularSoftmaxPolicy({s: np.log([1., 2., 3., 4.]) for s in world.states()
                                  if not world.is_terminal(s)}, device=device)
    gamma, noise = .5, 2.
    fit_started = perf_counter()
    training = [sample_episode(world, policy, seed=s, reward_noise_std=noise) for s in range(1000, 1032)]
    baseline = fit_monte_carlo_baseline(training, gamma, device=device)
    fit_elapsed = perf_counter() - fit_started
    run = collect_gradient_batches(world, policy, batch_seeds=[list(range(i, i+8)) for i in range(0, 256, 8)],
                                   baseline=baseline, gamma=gamma, reward_noise_std=noise)
    run["baselines"]["learned_mc"]["training_settings"] = "same world, policy, gamma, and noise as probes"
    run["runtime_seconds"].update(baseline_collection_and_fit=fit_elapsed,
                                  total_before_serialization=perf_counter() - started)
    return run


def summarize_probe(run: dict) -> dict:
    """Recompute all metrics from saved batch vectors and the saved exact reference."""
    return {name: summarize_gradients(estimates, run["exact_gradient"])
            for name, estimates in run["batch_estimates"].items()}


def save_probe(run: dict, output_dir: Path) -> None:
    """Save raw data and its reproducible summary to a new directory; never overwrite."""
    raw = json.dumps(run, indent=2, allow_nan=False) + "\n"
    summary = json.dumps(summarize_probe(run), indent=2, allow_nan=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "probe.json").write_text(raw)
    (output_dir / "summary.json").write_text(summary)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("run", help="collect the fixed short comparison")
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    summarize = commands.add_parser("summarize", help="print summary JSON from raw data; no new episodes")
    summarize.add_argument("path", type=Path, help="saved probe.json")
    args = parser.parse_args(argv)
    if args.command == "summarize":
        print(json.dumps(summarize_probe(json.loads(args.path.read_text())), indent=2, allow_nan=False))
        return
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new path to preserve existing outputs")
    run = run_short_probe(device=args.device)
    save_probe(run, args.output_dir)
    print("estimator        squared bias   variance(B)       MSE   mean noise floor")
    for name, metrics in summarize_probe(run).items():
        print(f"{name:16s} {metrics['squared_empirical_bias']:12.6f} "
              f"{metrics['empirical_variance_trace']:13.6f} {metrics['mse']:9.6f} "
              f"{metrics['mean_noise_floor']:18.6f}")
    print(f"Saved probe.json and summary.json to {args.output_dir}")
    print(f"Device: {run['device']}; runtime (before serialization): "
          f"{run['runtime_seconds']['total_before_serialization']:.3f} s")


if __name__ == "__main__":
    main()
