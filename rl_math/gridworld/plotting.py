"""Spatial views of an existing fixed policy, exact evaluation, and samples.

Plotting copies tensors to CPU only at the display boundary. No policy updates
or new Bellman calculations happen here. Callers must pass values evaluated
under the displayed policy and gamma, without changing that policy afterwards.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np

from common.policy import Policy
from common.returns import discounted_returns
from gridworld.model import Action, Gridworld, State
from gridworld.sampling import Episode
from gridworld.values import PolicyValues


def _grid(ax, world: Gridworld, *, terrain: bool = False) -> None:
    """Use the model's row/column coordinates, including walls and cell symbols."""
    colors = {"#": "#344054", "H": "#fde8df", "G": "#d9f0e6", "S": "#e1edf9", ".": "white"}
    for row, line in enumerate(world.layout):
        for col, symbol in enumerate(line):
            face = colors[symbol] if terrain or symbol == "#" else "none"
            ax.add_patch(Rectangle((col - .5, row - .5), 1, 1, facecolor=face,
                                   edgecolor="#98a2b3", linewidth=.7))
            if symbol in "SHG":
                ax.text(col - .39, row - .32, symbol, ha="left", va="center",
                        fontsize=8, color="#263445",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=.5))
    ax.set(xlim=(-.5, len(world.layout[0]) - .5), ylim=(len(world.layout) - .5, -.5),
           xticks=range(len(world.layout[0])), yticks=range(len(world.layout)),
           xlabel="column", ylabel="row", aspect="equal")
    ax.tick_params(length=0, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _heading(fig: Figure, world: Gridworld, title: str, gamma: float, policy_name: str) -> None:
    fig.suptitle(title, fontsize=17, fontweight="bold", y=.98)
    fig.text(.5, .93, f"Fixed policy: {policy_name}", ha="center", fontsize=10)
    fig.text(.5, .895,
             f"Task horizon={world.horizon} · slip={world.slip:g} · γ={gamma:g} · "
             f"step={world.step_reward:g} · goal bonus={world.goal_reward:g} · "
             f"hazard penalty={world.hazard_penalty:g}", ha="center", fontsize=10)


def _slice(world: Gridworld, table: dict, remaining: int, action: int | None = None) -> np.ndarray:
    if type(remaining) is not int or not 0 <= remaining <= world.horizon:
        raise ValueError("remaining must be an integer in [0, horizon]")
    data = np.full((len(world.layout), len(world.layout[0])), np.nan)
    for cell in world.cells:
        value = table[State(cell, remaining)]
        data[cell] = (value if action is None else value[action]).detach().cpu().item()
    return data


def _heatmap(ax, world: Gridworld, data: np.ndarray, *, norm: Normalize,
             cmap: str, signed: bool = False):
    image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=norm, interpolation="nearest")
    _grid(ax, world)
    for row, col in world.cells:
        value = data[row, col]
        red, green, blue, _ = image.cmap(norm(value))
        color = "black" if .299 * red + .587 * green + .114 * blue > .55 else "white"
        label = f"{value:+.2f}" if signed else f"{value:.2f}"
        ax.text(col, row + .06, label, ha="center", va="center", color=color, fontsize=10)
    return image


def plot_policy_values(
    world: Gridworld, policy: Policy[State], values: PolicyValues, *,
    remaining: int, gamma: float, policy_name: str,
) -> Figure:
    """Show V and all four action probabilities at the same remaining time.

    Arrows indicate intended actions, before environmental slip. Their length
    and width increase with probability; printed probabilities are authoritative.
    Terminal states have value zero and no policy arrows (no action is taken).
    """
    data = _slice(world, values.v, remaining)
    fig, (value_ax, policy_ax) = plt.subplots(1, 2, figsize=(12, 7))
    fig.subplots_adjust(left=.06, right=.96, bottom=.21, top=.81, wspace=.28)
    time_label = f"{remaining} {'step' if remaining == 1 else 'steps'} left"
    _heading(fig, world, f"Values and action probabilities · {time_label}", gamma, policy_name)
    lo, hi = np.nanmin(data), np.nanmax(data)
    norm = Normalize(lo if lo < hi else lo - 1, hi if lo < hi else hi + 1)
    image = _heatmap(value_ax, world, data, norm=norm, cmap="viridis")
    value_ax.set_title("V(s): expected discounted return", fontsize=12, pad=12)
    fig.colorbar(image, ax=value_ax, fraction=.046, pad=.04, label="V")

    _grid(policy_ax, world, terrain=True)
    policy_ax.set_title("π(a | s): intended action probabilities", fontsize=12, pad=12)
    for row, col in world.cells:
        state = State((row, col), remaining)
        if world.is_terminal(state):
            policy_ax.text(col, row + .06, "terminal", ha="center", va="center", fontsize=8)
            continue
        probabilities = policy.probabilities(state).detach().cpu().numpy()
        for action, probability in zip(Action, probabilities):
            dr, dc = action.value
            length = .05 + .17 * probability
            policy_ax.annotate("", (col + dc * length, row + dr * length), (col, row),
                               arrowprops=dict(arrowstyle="-|>", color="#205b84",
                                               lw=.5 + 2 * probability,
                                               mutation_scale=6 + 5 * probability))
            policy_ax.text(col + dc * .30, row + dr * .36, f"{probability:.2f}",
                           ha="center", va="center", fontsize=7.5)
    fig.text(.5, .085, "S start · G terminal goal · H hazard · dark cells are walls\n"
             "All non-wall cells are evaluated, including states unreachable at this remaining time.\n"
             "Goal and zero-time values are 0; the goal bonus is paid on entry. Zero-mean reward noise leaves V unchanged.",
             ha="center", va="center", fontsize=10, linespacing=1.6)
    return fig


def plot_advantages(
    world: Gridworld, values: PolicyValues, *, remaining: int, gamma: float, policy_name: str,
) -> Figure:
    """One heatmap per action, with a shared, zero-centered advantage scale."""
    data = [_slice(world, values.advantage, remaining, i) for i in range(len(Action))]
    limit = max(float(np.nanmax(np.abs(data))), 1e-12)
    norm = Normalize(-limit, limit)
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    fig.subplots_adjust(left=.07, right=.86, bottom=.16, top=.83, hspace=.32, wspace=.24)
    time_label = f"{remaining} {'step' if remaining == 1 else 'steps'} left"
    _heading(fig, world, f"Action advantages · {time_label}", gamma, policy_name)
    for ax, action, array in zip(axes.flat, Action, data):
        image = _heatmap(ax, world, array, norm=norm, cmap="RdBu", signed=True)
        ax.set_title(f"{action.name} · A(s, a) = Q(s, a) − V(s)", fontsize=11, pad=10)
    color_ax = fig.add_axes((.90, .27, .022, .43))
    fig.colorbar(image, cax=color_ax, label="Advantage (shared scale; 0 is neutral)")
    fig.text(.5, .065, "Blue / positive: better than this policy's average. Red / negative: worse.\n"
             "Positive advantage does not require a positive V or Q. The policy-weighted mean advantage is 0.\n"
             "S start · G terminal (all A=0) · H hazard · dark cells are walls; all cell/time states shown.\n"
             "Independent zero-mean reward noise leaves these exact expectations unchanged.",
             ha="center", va="center", fontsize=10, linespacing=1.6)
    return fig


def plot_trajectories(
    world: Gridworld, episodes: Sequence[Episode], *, gamma: float, policy_name: str,
) -> Figure:
    """Draw observed movement and label visit times, including blocked moves.

    Use episodes collected in this world under the named fixed policy. The
    path follows next_state, which can differ from the intended action on slip.
    Visit-time labels preserve revisits and stays even when path segments overlap.
    """
    if not episodes or any(not episode.steps for episode in episodes):
        raise ValueError("provide at least one nonempty episode")
    fig, axes = plt.subplots(1, len(episodes), figsize=(5 * len(episodes), 7), squeeze=False)
    fig.subplots_adjust(left=.045, right=.98, bottom=.22, top=.77, wspace=.23)
    _heading(fig, world, "Sampled paths · remaining time decreases after every step", gamma, policy_name)
    for ax, episode in zip(axes.flat, episodes):
        _grid(ax, world, terrain=True)
        cells = [episode.steps[0].state.cell] + [step.next_state.cell for step in episode.steps]
        rows, cols = np.array(cells).T
        ax.plot(cols, rows, color="#287da4", alpha=.7, linewidth=2.5, marker=".")
        for before, after in zip(cells, cells[1:]):
            if before != after:
                ax.annotate("", (after[1], after[0]), (before[1], before[0]),
                            arrowprops=dict(arrowstyle="->", color="#287da4", lw=1.5))
        ax.scatter([cols[-1]], [rows[-1]], s=210, facecolors="none", edgecolors="#172b4d", lw=2)
        for cell in dict.fromkeys(cells):
            times = [str(t) for t, visited in enumerate(cells) if visited == cell]
            label = "\n".join(",".join(times[i:i + 4]) for i in range(0, len(times), 4))
            ax.text(cell[1], cell[0] + .2, label, ha="center", va="center", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1))
        total_return = discounted_returns(episode.rewards(), gamma)[0]
        ax.set_title(f"seed={episode.seed} · {episode.end_reason} · {len(episode.steps)} steps\n"
                     f"G₀={total_return:.2f} · reward noise σ={episode.reward_noise_std:g}",
                     fontsize=11, pad=12)
    fig.text(.5, .09, "S start · G goal · H hazard · dark cells are walls · ring marks the final cell\n"
             "Numbers are visit times t (remaining = task horizon − t); multiple times in a cell show stays/revisits.\n"
             "Arrows follow observed motion after slip. Each panel is one sample, not an estimate of V.",
             ha="center", va="center", fontsize=10, linespacing=1.6)
    return fig
