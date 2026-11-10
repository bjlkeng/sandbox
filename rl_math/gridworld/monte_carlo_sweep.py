"""A declared small Monte Carlo sweep; replot saved raw batches without sampling.

From rl_math:
  python -m gridworld.monte_carlo_sweep run --pilot --output-dir runs/mc-pilot --device cuda
  python -m gridworld.monte_carlo_sweep run --output-dir runs/mc-sweep --device cuda
  python -m gridworld.monte_carlo_sweep report runs/mc-sweep/sweep.json --output-dir runs/mc-replot
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from common.gradient_metrics import summarize_comparison
from gridworld.baselines import fit_monte_carlo_baseline
from gridworld.gradient_probe import collect_gradient_batches
from gridworld.model import Gridworld
from gridworld.policy import TabularSoftmaxPolicy
from gridworld.sampling import sample_episode


METHODS = {
    "full_return": ("Full return", "#4477AA", "o"),
    "reward_to_go": ("Reward-to-go", "#228833", "s"),
    "exact_v": ("Exact V", "#AA7733", "^"),
    "learned_mc": ("Learned MC", "#CC6677", "D"),
}


def run_sweep(*, device: str = "cuda", pilot: bool = False) -> dict:
    """Freeze one policy and fit across all noise/episode-count conditions.

    Full: B=128, N=(1,8,32), sigma=(0,2,4). Collect 15,744 probe episodes.
    Pilot: B=8, N=(1,8), sigma=(0,4), 144 episodes, separate seed region.
    A single 32-episode fit at sigma=2 is held fixed even as probe noise changes;
    this isolates measurement noise from changes in baseline fitting quality.
    Probes have disjoint seeds across ALL conditions/batches; four estimators
    reuse the same episodes within each batch. No policy/baseline updates.
    """
    import matplotlib

    started = perf_counter()
    settings = {
        "pilot": pilot, "batch_count": 8 if pilot else 128,
        "batch_sizes": [1, 8] if pilot else [1, 8, 32],
        "noise_stds": [0., 4.] if pilot else [0., 2., 4.],
        "probe_seed_start": 100000 if pilot else 10000,
        "training_seeds": list(range(1000, 1032)), "training_noise_std": 2.,
        "bootstrap_resamples": 2000, "bootstrap_seed_start": 20261112,
    }
    world, gamma = Gridworld(layout=("SHG",), horizon=2, slip=0.), .5
    policy = TabularSoftmaxPolicy({s: np.log([1., 2., 3., 4.]) for s in world.states()
                                  if not world.is_terminal(s)}, device=device)
    fit_started = perf_counter()
    training = [sample_episode(world, policy, seed=s, reward_noise_std=settings["training_noise_std"])
                for s in settings["training_seeds"]]
    baseline = fit_monte_carlo_baseline(training, gamma, device=device)
    fit_seconds = perf_counter() - fit_started
    runs, next_seed = [], settings["probe_seed_start"]
    for noise in settings["noise_stds"]:
        for size in settings["batch_sizes"]:
            seeds = [list(range(next_seed + b*size, next_seed + (b+1)*size))
                     for b in range(settings["batch_count"])]
            next_seed += settings["batch_count"] * size
            runs.append(collect_gradient_batches(world, policy, batch_seeds=seeds, baseline=baseline,
                                                 gamma=gamma, reward_noise_std=noise))
    probe_transitions = sum(sum(run["sampling"]["transitions_per_batch"]) for run in runs)
    training_transitions = sum(baseline.counts.values())
    return {
        "schema_version": 1, "settings": settings, "runs": runs,
        "report_versions": {"matplotlib": matplotlib.__version__},
        "budget": {"probe_episodes": next_seed - settings["probe_seed_start"],
                   "probe_transitions": probe_transitions, "training_episodes": len(training),
                   "training_transitions": training_transitions,
                   "total_transitions": probe_transitions + training_transitions},
        "runtime_seconds": {"collection_and_fit": fit_seconds,
                            "total_before_analysis": perf_counter() - started},
    }


def summarize_sweep(raw: dict) -> dict:
    """Use only saved vectors/reference/settings; reproduce every plotted value."""
    first = raw["runs"][0]
    conditions = []
    for i, run in enumerate(raw["runs"]):
        comparison = summarize_comparison(
            run["batch_estimates"], run["exact_gradient"], comparison="reward_to_go",
            bootstrap_seed=raw["settings"]["bootstrap_seed_start"] + i,
            resamples=raw["settings"]["bootstrap_resamples"],
        )
        conditions.append({"noise_std": run["sampling"]["reward_noise_std"],
                           "episodes_per_batch": run["sampling"]["episodes_per_batch"],
                           "comparison": comparison})
    # Predetermined worked coordinate, not selected after looking for significance.
    row = next(i for i, s in enumerate(first["policy"]["rows"])
               if s["cell"] == [0, 1] and s["remaining"] == 1)
    action = first["policy"]["action_order"].index("EAST")
    index = row * len(first["policy"]["action_order"]) + action
    return {
        "schema_version": 1, "settings": raw["settings"], "budget": raw["budget"],
        "runtime_seconds": raw["runtime_seconds"],
        "world": first["world"], "objective": first["objective"], "policy": first["policy"],
        "baselines": first["baselines"], "versions": first["versions"] | raw["report_versions"],
        "device": first["device"], "gpu": first["gpu"],
        "bias_coordinate": {"index": index, "cell": [0, 1], "remaining": 1, "action": "EAST",
                            "exact_gradient": first["exact_gradient"][index]},
        "conditions": conditions,
        "interpretation": {
            "variance": "sample covariance trace (B-1); all logit coordinates; no division by dimension",
            "mse": "average squared error of one N-episode batch estimate, not error of the grand mean",
            "bias": "signed mean-minus-exact intervals for every coordinate; plot the predetermined H/1 EAST coordinate",
            "squared_bias": "empirical point and mean_noise_floor only; not a confidence interval for true squared bias",
            "intervals": "approximate pointwise 95% paired percentile intervals; no simultaneous coverage or baseline-fit uncertainty",
            "paired_difference": "method MSE minus reward-to-go MSE; below zero means lower error in this setting",
            "baseline": "one sigma=2 training fit reused at every probe noise; extra fit budget recorded separately",
            "seeds": "disjoint across probe settings/batches and training; shared between estimators within a batch",
        },
    }


def plot_sweep(summary: dict) -> dict:
    """Plot the saved statistics and interval endpoints, with consistent method colors."""
    import matplotlib.pyplot as plt

    settings = summary["settings"]
    noises, sizes = settings["noise_stds"], settings["batch_sizes"]
    coordinate = summary["bias_coordinate"]["index"]
    metrics, axes = plt.subplots(3, len(noises), figsize=(5*len(noises), 9), squeeze=False, sharey="row")
    paired, pair_axes = plt.subplots(1, len(noises), figsize=(5*len(noises), 4), squeeze=False, sharey=True)
    for column, noise in enumerate(noises):
        conditions = [c for c in summary["conditions"] if c["noise_std"] == noise]
        for method_index, (name, (label, color, marker)) in enumerate(METHODS.items()):
            # A slight horizontal offset separates the four intervals at each N.
            x = np.asarray(sizes) * 1.04**(method_index - 1.5)
            entries = [c["comparison"]["estimators"][name] for c in conditions]
            for row, key in enumerate(("empirical_bias", "sample_variance_trace", "mse")):
                points = [e["metrics"][key][coordinate] if row == 0 else e["metrics"][key] for e in entries]
                intervals = np.array([e["intervals"][key] for e in entries])
                if row == 0:
                    intervals = intervals[:, :, coordinate]
                ax = axes[row, column]
                ax.plot(x, points, marker=marker, color=color, label=label, markersize=4)
                # Draw endpoints directly: percentile intervals need not contain the point estimate.
                ax.vlines(x, intervals[:, 0], intervals[:, 1], color=color, alpha=.65)
            if name != "reward_to_go":
                ax = pair_axes[0, column]
                differences = [e["mse_difference"]["estimate"] for e in entries]
                intervals = np.array([e["mse_difference"]["interval"] for e in entries])
                ax.plot(x, differences, marker=marker, color=color, label=label, markersize=4)
                ax.vlines(x, intervals[:, 0], intervals[:, 1], color=color, alpha=.65)
        axes[0, column].axhline(0, color="black", linestyle=":", linewidth=1)
        pair_axes[0, column].axhline(0, color="black", linestyle=":", linewidth=1)
        axes[0, column].set_title(f"Reward noise sigma = {noise:g}")
        pair_axes[0, column].set_title(f"Reward noise sigma = {noise:g}")
        for ax in [*axes[:, column], pair_axes[0, column]]:
            ax.set_xscale("log", base=2)
            ax.set_xticks(sizes, [str(n) for n in sizes])
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
        axes[2, column].set_xlabel("Episodes per batch (N)")
        pair_axes[0, column].set_xlabel("Episodes per batch (N)")
    for row, key in ((1, "sample_variance_trace"), (2, "mse")):
        # The full run has positive bounds. A tiny pilot may have zero bounds;
        # keep those visible with a linear scale rather than silently clipping.
        if all(e["intervals"][key][0] > 0
               for c in summary["conditions"] for e in c["comparison"]["estimators"].values()):
            axes[row, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Mean error: H/1 EAST gradient")
    axes[1, 0].set_ylabel("Sample variance trace (all logits)")
    axes[2, 0].set_ylabel("MSE (all logits)")
    pair_axes[0, 0].set_ylabel("MSE difference from reward-to-go")
    mode = "PILOT / validation only — " if settings["pilot"] else ""
    subtitle = (f"{settings['batch_count']} independent batches per point; "
                "SHG, horizon 2, slip 0, gamma 0.5, fixed pi=(0.1,0.2,0.3,0.4)")
    metrics.suptitle(f"{mode}Monte Carlo gradient estimates\n{subtitle}", fontsize=12)
    paired.suptitle(f"{mode}Paired MSE comparison (below zero favors the named method)\n{subtitle}", fontsize=12)
    for fig, ax in ((metrics, axes[0, 0]), (paired, pair_axes[0, 0])):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .91), ncol=4, frameon=False)
        fig.text(.5, .012,
                 f"Bars: approximate pointwise 95% intervals, {settings['bootstrap_resamples']:,} paired bootstrap resamples. "
                 "One frozen 32-episode MC fit at sigma=2; methods offset slightly along x.",
                 ha="center", fontsize=8)
    metrics.subplots_adjust(left=.07, right=.985, bottom=.085, top=.85, wspace=.30, hspace=.30)
    paired.subplots_adjust(left=.07, right=.985, bottom=.22, top=.73, wspace=.30)
    return {"metrics.png": metrics, "paired_mse.png": paired}


def _write_report(summary: dict, output_dir: Path) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    figures = plot_sweep(summary)
    try:
        for name, fig in figures.items():
            fig.savefig(output_dir / name, dpi=160, facecolor="white",
                        metadata={"Software": f"Matplotlib {matplotlib.__version__}"})
    finally:
        for fig in figures.values():
            plt.close(fig)


def save_sweep(raw: dict, output_dir: Path) -> None:
    """Save raw batches, summary and figures to a new directory, never overwriting."""
    encoded = json.dumps(raw, indent=2, allow_nan=False) + "\n"
    summary = summarize_sweep(raw)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "sweep.json").write_text(encoded)
    _write_report(summary, output_dir)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="collect the declared sweep and save its report")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run.add_argument("--pilot", action="store_true", help="separate tiny validation run, not final evidence")
    report = commands.add_parser("report", help="rebuild summary and figures from raw JSON, without sampling")
    report.add_argument("path", type=Path, help="saved sweep.json")
    report.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new path to preserve outputs")
    import matplotlib
    matplotlib.use("Agg")
    started = perf_counter()
    if args.command == "run":
        raw = run_sweep(device=args.device, pilot=args.pilot)
        save_sweep(raw, args.output_dir)
        print(f"Collected {raw['budget']['probe_episodes']} shared probe episodes; "
              f"{raw['budget']['total_transitions']} transitions including the one baseline fit.")
    else:
        raw = json.loads(args.path.read_text())
        summary = summarize_sweep(raw)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _write_report(summary, args.output_dir)
        print("Rebuilt report from raw batch vectors; no collection, fitting or GPU computation.")
    print(f"Saved report to {args.output_dir}; command elapsed {perf_counter() - started:.3f} s")


if __name__ == "__main__":
    main()
