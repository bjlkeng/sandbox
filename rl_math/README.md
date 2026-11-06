# A Reinforcement Learning Primer

Supporting code and notebooks for the in-progress blog post
[A Reinforcement Learning Primer](https://github.com/bjlkeng/bjlkeng.github.io/blob/post-rl-math/source/posts/rl-math.rst).

See [PLAN.md](PLAN.md) for the proposed experiments and implementation order.
Create implementation tasks from the plan only when requested, using this
subproject's own Beads workspace.

This directory is a starting point for experiments illustrating the post's
policy-gradient math:

* Markov decision processes, policies, trajectories, and discounted returns.
* Value and Q functions, Bellman equations, and advantages.
* Monte Carlo policy gradients (REINFORCE), reward-to-go, and baselines.
* Actor-critic, bootstrapping, and n-step advantage estimates.
* GAE and PPO-Clip as recommended additions to the draft.

`common/` holds the policy interface and return helpers shared across games.
`algorithms/` holds environment-independent gradient estimators.
`gridworld/` holds the explicit model, tabular policy, episode sampler, and local
player. Additional learning algorithms and game packages follow the
layout in [PLAN.md](PLAN.md) as their experiments are implemented.

From this directory, create the locked environment and open the worked notebook:

```bash
uv sync --locked
uv run --locked jupyter lab notebooks/01_gridworld.ipynb
```

The locked Linux environment includes CUDA-enabled PyTorch, NumPy, Matplotlib, JupyterLab,
and the Python kernel. In JupyterLab, use **Python 3 (ipykernel)** and choose
**Run All Cells**. Restart the kernel after changing the environment. In VS Code,
select this directory's `.venv/bin/python` as the notebook interpreter. The
`.python-version` file selects Python 3.10; `uv.lock` records exact dependencies.

The tabular policy stores float64 PyTorch parameters, using CUDA when available.
Pass `device="cuda"` to require the GPU, or `device="cpu"` for CPU reference checks.
The worked notebook explicitly requests CUDA and prints the GPU name.
Check GPU availability in the project environment with:

```bash
uv run --locked python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.cuda)"
```

`gridworld.values.evaluate_policy` computes V, Q, advantage, and the
scalar `start_return` tensor while retaining the autograd graph.
Model constants, terminal values, and derivatives follow the policy's device.
`gridworld.gradients.differentiate_policy` uses `torch.autograd.grad` to obtain
one gradient tensor per logit row, without updating parameters or accumulating
their `.grad` buffers. The notebook checks these derivatives against central
finite differences. NumPy owns the sampler's seeded random streams; the sampler
detaches and copies probabilities to CPU at that boundary. Pure evaluation can
use `torch.no_grad()`. The test suite includes CPU reference checks and CUDA
agreement/sampling checks; CUDA checks are skipped when no GPU is available.

`Policy.log_probabilities` exposes action log-probabilities with their graph;
the tabular policy uses `torch.log_softmax` for numerical stability.
`Episode.rewards()` extracts observed rewards, and
`Episode.log_probabilities(policy.log_probabilities)` evaluates the selected
action log-probabilities with a fresh autograd graph.
`algorithms.reinforce.full_return_gradient(log_probabilities, rewards, parameters, gamma)`
differentiates `G_0 * sum_t log pi(a_t | s_t)` for one complete episode collected
under the same frozen policy. Average the returned episode gradient sums for a
batch estimate. It returns detached gradients without parameter updates.
Every action gets the same discounted full return G_0, with no extra
outer discount or division by episode length. The notebook works one trajectory
by hand, checks the expected estimator over all 16 paths of a two-step model,
and compares one 256-episode seeded estimate with the exact gradient.

`algorithms.reinforce.reward_to_go_gradient` has the same episode-level API.
It differentiates `sum_t gamma**t * G_t * log pi(a_t | s_t)`, using the existing
`common.returns.reward_to_go_weights` helper. The outer `gamma**t` is required
for the discounted start-state objective; the discount inside `G_t` starts at
action t. Removing past rewards preserves the expected gradient because the
current action's expected score, conditional on its preceding history, is zero.
The notebook contrasts the two estimators on the same `S -> H -> G` path with
intermediate rewards, enumerates both exact expectations at gamma=0.5 and 1,
and reuses the same 256 sampled episodes for a paired illustration. Terminal
rewards alone make the sampled estimators coincide. Each gradient call needs
fresh log-probabilities under the unchanged collecting policy; both return
detached gradients without changing parameters or their `.grad` buffers.

The notebook also plots three sampled paths (seeds 0, 1, 7), exact V and policy
probabilities with 1 and 12 steps remaining, and four action-advantage maps
with one shared scale. Every figure identifies the fixed policy and model
settings. The worked `SHG` example shows why an action can have positive
advantage even with negative V: `Q(NORTH) - V = -2 - (-2.9) = +0.9`.
`gridworld.plotting` displays existing evaluator/sampler results; it detaches
tensors at the display boundary. The policy's hand-chosen preferences are
shared between the notebook and figure command in `gridworld.figures`.

Reproduce the selected [figures](figures/gridworld/) noninteractively from this
directory, using the same locked environment:

```bash
uv sync --locked
uv run --locked python -m gridworld.figures --output-dir runs/gridworld-figures --device cuda
```

This writes four PNGs and `manifest.json` containing the model/reward settings,
fixed policy rule, seeds, library versions, device, selected exact values,
sampled paths/rewards, and elapsed time. It uses the RTX/CUDA device when
requested; `--device cpu` selects the CPU reference. The output directory must
be new; choose another path for a rerun. Existing outputs are never overwritten.
`runs/` is ignored; `figures/gridworld/` holds the selected publication copies.
The notebook only displays figures, so rerunning its cells creates no files.

When a Tailscale proxy forwards its hostname, include that hostname in Jupyter's
allowed local names. Replace the example below with the hostname in your browser:

```bash
uv run --locked jupyter lab notebooks/01_gridworld.ipynb --no-browser \
  --ServerApp.ip=127.0.0.1 \
  --ServerApp.local_hostnames='["localhost","your-machine.your-tailnet.ts.net"]'
```

Point the proxy at Jupyter's loopback address and port. Use Jupyter's login token
when opening the proxied URL. See the
[Jupyter hostname setting](https://jupyter-server.readthedocs.io/en/latest/other/full-config.html#ServerApp.local_hostnames).

Run checks or the local player in the same environment:

```bash
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m gridworld.play
```

The player is available at <http://127.0.0.1:8000>; use `--port` to choose another port.
The worked example is in [notebooks/01_gridworld.ipynb](notebooks/01_gridworld.ipynb).
