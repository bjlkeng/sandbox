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

The locked Linux environment includes CUDA-enabled PyTorch, NumPy, JupyterLab,
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
`algorithms.reinforce.full_return_gradient` differentiates
`mean_episodes(G_0 * sum_t log pi(a_t | s_t))` for complete episodes collected
under the same frozen policy. It returns detached gradients without parameter
updates. Every action gets the same discounted full return G_0, with no extra
outer discount or division by episode length. The notebook works one trajectory
by hand, checks the expected estimator over all 16 paths of a two-step model,
and compares one 256-episode seeded estimate with the exact gradient.

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
