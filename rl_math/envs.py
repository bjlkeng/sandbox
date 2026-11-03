"""A finite-horizon gridworld with an explicit transition/reward model.

Cells are (row, column), starting at the upper left. The full Markov state
also includes remaining time: at the same cell, one step left may prevent
reaching a goal that is reachable with ten. The horizon is part of the task,
so remaining=0 is genuine termination with zero future value.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite


Cell = tuple[int, int]


class Action(Enum):
    NORTH = (-1, 0)
    EAST = (0, 1)
    SOUTH = (1, 0)
    WEST = (0, -1)


@dataclass(frozen=True)
class State:
    cell: Cell
    remaining: int


@dataclass(frozen=True)
class Transition:
    """One outcome: P(s' | s, a) and E[r_{t+1} | s, a, s']."""

    probability: float
    next_state: State
    reward: float


@dataclass(frozen=True)
class Gridworld:
    """S=start, G=terminal goal, #=wall, H=hazard, .=ordinary cell.

    With probability 1-slip, move in the intended direction; otherwise choose
    uniformly among the other three directions. A wall/boundary blocks motion
    but still consumes a step. Merge directions that reach the same next state.

    Every nonterminal transition pays step_reward, plus hazard_penalty when
    its destination is H (including a blocked move on H), or goal_reward when
    it reaches G. Hazards do not terminate. Goal and zero-time states have
    absorbing, zero-reward transitions, so goal reward is paid only once.

    Example: Gridworld(layout=("SHG", ".#."), horizon=2, slip=0.3).
    Taking EAST from S reaches H with probability .7 and reward -6, stays at S
    with probability .2 and reward -1, or goes SOUTH with probability .1 and
    reward -1. Each successor has one step left; E[r | S, EAST] = -4.5.
    """

    layout: tuple[str, ...] = (".....", "..#..", "..#..", ".....", "SHHHG")
    horizon: int = 12
    slip: float = 0.2
    step_reward: float = -1.0
    goal_reward: float = 10.0
    hazard_penalty: float = -5.0
    start: Cell = field(init=False)
    goal: Cell = field(init=False)
    cells: tuple[Cell, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", tuple(self.layout))
        if not self.layout or not self.layout[0]:
            raise ValueError("layout must be a nonempty rectangle")
        if any(len(row) != len(self.layout[0]) for row in self.layout):
            raise ValueError("layout must be a nonempty rectangle")
        if any(symbol not in "SG#H." for row in self.layout for symbol in row):
            raise ValueError("layout symbols must be S, G, #, H, or .")
        cells = tuple((r, c) for r, row in enumerate(self.layout)
                      for c, symbol in enumerate(row) if symbol != "#")
        for symbol, name in (("S", "start"), ("G", "goal")):
            positions = [cell for cell in cells if self.layout[cell[0]][cell[1]] == symbol]
            if len(positions) != 1:
                raise ValueError(f"layout must have exactly one {symbol}")
            object.__setattr__(self, name, positions[0])
        object.__setattr__(self, "cells", cells)
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not 0.0 <= self.slip <= 1.0:
            raise ValueError("slip must lie in [0, 1]")
        if not all(isfinite(value) for value in
                   (self.step_reward, self.goal_reward, self.hazard_penalty)):
            raise ValueError("rewards must be finite")

    @property
    def start_state(self) -> State:
        return State(self.start, self.horizon)

    def states(self) -> Iterator[State]:
        """Enumerate all non-wall cell/time pairs, including terminal states.

        Remaining time increases from 0 to horizon, convenient for backward
        dynamic programming. Some pairs need not be reachable from start_state.
        """
        for remaining in range(self.horizon + 1):
            for cell in self.cells:
                yield State(cell, remaining)

    def is_terminal(self, state: State) -> bool:
        self._validate_state(state)
        return state.remaining == 0 or state.cell == self.goal

    def transitions(self, state: State, action: Action) -> tuple[Transition, ...]:
        """Enumerate the nonzero outcomes of (s_t, a_t, r_{t+1}, s_{t+1}).

        reward is the conditional expected reward for that outcome. The model
        uses no random draws; expected_reward averages over these outcomes.
        """
        if not isinstance(action, Action):
            raise ValueError("action must be an Action")
        if self.is_terminal(state):
            return (Transition(1.0, state, 0.0),)

        probabilities: dict[Cell, float] = {}
        for direction in Action:
            probability = 1.0 - self.slip if direction == action else self.slip / 3
            if probability == 0.0:
                continue
            dr, dc = direction.value
            destination = (state.cell[0] + dr, state.cell[1] + dc)
            if destination not in self.cells:
                destination = state.cell
            probabilities[destination] = probabilities.get(destination, 0.0) + probability

        outcomes = []
        for cell, probability in probabilities.items():
            symbol = self.layout[cell[0]][cell[1]]
            reward = self.step_reward
            if symbol == "H":
                reward += self.hazard_penalty
            elif symbol == "G":
                reward += self.goal_reward
            outcomes.append(Transition(probability, State(cell, state.remaining - 1), reward))
        return tuple(outcomes)

    def expected_reward(self, state: State, action: Action) -> float:
        """E[r_{t+1} | s_t, a_t] = sum_s' P(s' | s_t, a_t) * reward(s')."""
        return sum(outcome.probability * outcome.reward
                   for outcome in self.transitions(state, action))

    def _validate_state(self, state: State) -> None:
        if state.cell not in self.cells or any(type(coord) is not int for coord in state.cell):
            raise ValueError("state cell must be an integer (row, column) on a non-wall cell")
        if type(state.remaining) is not int or not 0 <= state.remaining <= self.horizon:
            raise ValueError("remaining time must be an integer in [0, horizon]")
