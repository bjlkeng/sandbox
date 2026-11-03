"""Play locally: python3 play_gridworld.py [--port 8000], then open the URL.

Standard library only. One game is shared by all tabs connected to this server.
"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import random
from urllib.parse import parse_qs

from envs import Action, Gridworld


class Game:
    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random.Random()
        self.reset()

    def reset(self, slip=0.2):
        self.world = Gridworld(slip=slip)
        self.state = self.world.start_state
        self.total = 0.0
        self.last = "Use arrow keys or the buttons to reach G."

    def move(self, action):
        if self.world.is_terminal(self.state):
            return
        outcomes = self.world.transitions(self.state, action)
        outcome = self.rng.choices(outcomes, weights=[o.probability for o in outcomes])[0]
        self.state = outcome.next_state
        self.total += outcome.reward
        self.last = f"{action.name.title()} → {self.state.cell}, reward {outcome.reward:g}."


def render(game):
    world, state = game.world, game.state
    cells = []
    names = {"S": "start", "G": "goal", "H": "hazard", "#": "wall", ".": "floor"}
    for row, line in enumerate(world.layout):
        for column, symbol in enumerate(line):
            player = (row, column) == state.cell
            label = f"row {row}, column {column}, {names[symbol]}" + (", you" if player else "")
            tile = "●" if player else symbol if symbol != "." else ""
            cells.append(f'<span class="{names[symbol]}" aria-label="{label}">{tile}</span>')
    terminal = world.is_terminal(state)
    status = "Goal reached!" if state.cell == world.goal else "Time is up!" if terminal else ""
    disabled = "disabled" if terminal else ""
    checked = "checked" if world.slip else ""
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gridworld</title>
<style>
  body {{ font: 16px system-ui; max-width: 390px; margin: 32px auto; padding: 0 16px; color: #222; }}
  h1 {{ font-size: 24px; }}
  #grid {{ display: grid; grid-template-columns: repeat({len(world.layout[0])}, 1fr); gap: 4px; }}
  #grid span {{ display: grid; place-items: center; aspect-ratio: 1; background: #eee; font-size: 24px; }}
  #grid .wall {{ background: #555; color: white; }}
  #grid .hazard {{ background: #f7c7bf; }}
  #grid .goal {{ background: #bfe0ca; }}
  #grid .start {{ background: #d3dff0; }}
  button {{ font: inherit; padding: 8px 16px; cursor: pointer; }}
  button:disabled {{ cursor: default; }}
  fieldset {{ border: 0; padding: 0; text-align: center; margin: 16px 0; }}
  small {{ display: block; line-height: 1.5; margin: 16px 0; }}
</style>
<h1>Gridworld</h1>
<div id="grid" role="group" aria-label="Gridworld board">{''.join(cells)}</div>
<p><b>{state.remaining}</b> steps left · Total reward: <b>{game.total:g}</b> · Slip: {world.slip:.0%}</p>
<p aria-live="polite">{game.last} <b>{status}</b></p>
<form method="post" action="/move">
  <fieldset {disabled} aria-label="Move">
    <button id="NORTH" name="action" value="NORTH" aria-label="North">↑</button><br>
    <button id="WEST" name="action" value="WEST" aria-label="West">←</button>
    <button id="SOUTH" name="action" value="SOUTH" aria-label="South">↓</button>
    <button id="EAST" name="action" value="EAST" aria-label="East">→</button>
  </fieldset>
</form>
<form method="post" action="/reset">
  <label><input type="checkbox" name="slip" {checked}>20% slip on new game</label>
  <button>New game</button>
</form>
<small>● you · S start · G goal · H hazard · # wall.<br>
Each move: {world.step_reward:g}. Hazard: another {world.hazard_penalty:g}. Goal: adds {world.goal_reward:g}.<br>
Walls and edges block movement but use a step. End at G or after {world.horizon} moves.
Slip chooses one of the other three directions. Coordinates are (row, column), starting at 0.</small>
<script>
  const keys = {{ArrowUp: 'NORTH', ArrowRight: 'EAST', ArrowDown: 'SOUTH', ArrowLeft: 'WEST'}};
  document.addEventListener('keydown', event => {{
    if (!keys[event.key] || event.target.tagName === 'INPUT') return;
    event.preventDefault();
    if (!event.repeat) document.getElementById(keys[event.key]).click();
  }});
</script>
</html>"""


class Handler(BaseHTTPRequestHandler):
    game = Game()

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        page = render(self.game).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self):
        if self.path not in ("/move", "/reset"):
            self.send_error(404)
            return
        try:
            fields = parse_qs(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
            if self.path == "/reset":
                self.game.reset(slip=0.2 if "slip" in fields else 0.0)
            else:
                self.game.move(Action[fields["action"][0]])
        except (KeyError, ValueError):
            self.send_error(400, "Invalid move")
            return
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Gridworld: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
