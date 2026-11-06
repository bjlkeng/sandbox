"""Reproduce the first notebook's fixed-policy figures without a display server.

From rl_math: python -m gridworld.figures --output-dir runs/gridworld-figures --device cuda
The output directory must be new, so a rerun never replaces unrelated outputs.
"""

import argparse
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch

from gridworld.model import Action, Gridworld, State
from gridworld.policy import TabularSoftmaxPolicy
from gridworld.sampling import sample_episode
from gridworld.values import evaluate_policy


POLICY_NAME = "safe-route preference (0.7 preferred, 0.1 each other action)"
FIGURE_SEEDS = (0, 1, 7)


def safe_route_policy(world: Gridworld, *, device=None) -> TabularSoftmaxPolicy:
    """The notebook's hand-chosen logits, held fixed at every remaining time.

    On the bottom row prefer NORTH; elsewhere prefer EAST until the last
    column, then SOUTH. On the default map this favors the longer safe route.
    A logit of log(7) versus three zeros yields probabilities (.7, .1, .1, .1)
    in the corresponding action order. This is a preference, not a safe-path
    guarantee: both policy sampling and environmental slip can enter hazards.
    """
    logits = {}
    for state in world.states():
        if world.is_terminal(state):
            continue
        row, column = state.cell
        preferred = (Action.NORTH if row == len(world.layout) - 1 else
                     Action.EAST if column < len(world.layout[0]) - 1 else Action.SOUTH)
        logits[state] = np.zeros(len(Action))
        logits[state][tuple(Action).index(preferred)] = np.log(7)
    return TabularSoftmaxPolicy(logits, device=device)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for PNGs and manifest.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new path to preserve existing outputs")

    # The command is noninteractive; importing the policy helper does not change
    # a notebook's backend. Import pyplot only after selecting Agg here.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gridworld.plotting import plot_advantages, plot_policy_values, plot_trajectories

    started = perf_counter()
    world, gamma, noise_std = Gridworld(), .9, 2.
    policy = safe_route_policy(world, device=args.device)
    with torch.no_grad():
        values = evaluate_policy(world, policy, gamma)
        episodes = [sample_episode(world, policy, seed=seed, reward_noise_std=noise_std)
                    for seed in FIGURE_SEEDS]
    figures = {"paths.png": plot_trajectories(world, episodes, gamma=gamma, policy_name=POLICY_NAME)}
    for remaining in (1, world.horizon):
        figures[f"values_policy_h{remaining:02d}.png"] = plot_policy_values(
            world, policy, values, remaining=remaining, gamma=gamma, policy_name=POLICY_NAME)
    figures["advantages_h12.png"] = plot_advantages(
        world, values, remaining=world.horizon, gamma=gamma, policy_name=POLICY_NAME)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, fig in figures.items():
        fig.savefig(args.output_dir / name, dpi=160, facecolor="white")
        plt.close(fig)
    manifest = {
        "world": {"layout": world.layout, "horizon": world.horizon, "slip": world.slip,
                  "step_reward": world.step_reward, "goal_reward": world.goal_reward,
                  "hazard_penalty": world.hazard_penalty},
        "policy": {"name": POLICY_NAME, "rule": safe_route_policy.__doc__.strip(),
                   "action_order": [action.name for action in Action]},
        "gamma": gamma, "reward_noise_std": noise_std, "seeds": FIGURE_SEEDS,
        "remaining_slices": [1, world.horizon], "start_return": values.start_return.item(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__, "matplotlib": matplotlib.__version__},
        "device": str(policy.probabilities(world.start_state).device),
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "figures": list(figures),
        "selected_cells": [
            {"cell": state.cell, "remaining": state.remaining, "v": values.v[state].item(),
             "q": values.q[state].tolist(), "advantage": values.advantage[state].tolist(),
             "probabilities": policy.probabilities(state).detach().cpu().tolist()}
            for state in (State(world.start, 1), world.start_state, State((3, 3), world.horizon))
        ],
        "episodes": [
            {"seed": e.seed, "end_reason": e.end_reason,
             "cells": [world.start] + [step.next_state.cell for step in e.steps],
             "actions": [step.action.name for step in e.steps], "rewards": e.rewards()}
            for e in episodes
        ],
        "elapsed_seconds": perf_counter() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(figures)} figures and manifest.json to {args.output_dir}")
    print(f"Device: {manifest['device']}; exact J={values.start_return.item():.9f}; "
          f"elapsed={manifest['elapsed_seconds']:.3f} s")


if __name__ == "__main__":
    main()
