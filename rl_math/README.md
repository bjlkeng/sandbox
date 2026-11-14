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
`.python-version` file selects Python 3.11 (minimum 3.11.5); `uv.lock` records
exact dependencies. The patched Pillow, filelock, Tornado, and Soup Sieve
dependencies require this newer Python baseline. `uv sync --locked` creates
the matching environment; restart the notebook kernel after upgrading.

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

Pass `baselines=` to `reward_to_go_gradient` to subtract one fixed state-only
value per action in episode order. For example, obtain exact V with
`baseline_values = evaluate_policy(world, policy, gamma).v`, then pass
`torch.stack([baseline_values[step.state] for step in episode.steps])`.
The estimator differentiates only log-probabilities, with detached weights
`gamma**t * (G_t - b(s_t))`. Baselines follow the log-probability device/dtype;
omitting them preserves plain reward-to-go. The caller must keep the baseline
function fixed and independent of the sampled action and subsequent returns.
Fitting it on the same probe episodes can violate that condition.

The notebook extends the `SHG` example with exact V, checks cancellation over
all actions and complete paths for exact and intentionally imperfect baselines,
and reuses the existing 256 episodes without updating the policy. `G_t - V(s_t)`
is a sampled advantage whose conditional mean is `Q(s_t,a_t) - V(s_t)`.
A two-action counterexample shows why exact V need not minimize gradient
variance: it minimizes unweighted squared return error, whereas the gradient
also contains action-dependent scores. No value fitting or bootstrapping is
needed for this baseline example.

`gridworld.baselines.fit_monte_carlo_baseline(training_episodes, gamma, device=...)`
fits a tabular state/time baseline from complete recorded returns. Each fitted
value is the mean G_t at that state, the exact least-squares solution without an
optimizer or bootstrapping. Float64 means use the selected device; the result
stores read-only scalar values, visit counts, training seeds, and gamma.
Unseen states have the fixed fallback zero. Pass `learned.for_episode(probe)`
as `baselines=` to the existing reward-to-go estimator. Lookup reads only probe
states and rejects training seeds; callers must still collect independent data.

The notebook fits 32 noisy episodes with seeds 1000..1031, then freezes the
table for the existing 256 probes (seeds 0..255). Both use the same fixed `SHG`
policy, horizon 2, no slip, gamma=0.5, and reward-noise sigma=2. It displays
training targets/counts and errors against exact V, verifies the conditional
expected gradient by path enumeration, and reports the held-out probe estimates.
Fitting adds 64 transitions to the probes' 512. A one-step example shows how
fitting a baseline to the probe's own returns can bias the gradient even when
the fitted values are detached: detachment removes an autograd path, not the
statistical dependence on the sampled action. There are no policy updates.

`common.gradient_metrics.summarize_gradients` compares repeated **batch mean
gradient vectors** with the exact reference in a fixed logit coordinate order.
It sums coordinate variances (the covariance trace), without averaging over
coordinates. With B batches, the finite-sample identity is
`MSE = squared_empirical_bias + empirical_variance_trace`, where the variance
uses denominator B. The separately reported sample variance uses B-1, so its
contribution to that identity is `(B-1)/B * sample_variance_trace`.
The squared empirical bias includes noise in the mean of B batches; its expected
value is true squared bias plus population batch variance/B. The reported
`mean_noise_floor = sample_variance_trace/B` estimates that extra contribution,
and `mean_standard_error` gives coordinatewise uncertainty in the grand mean.
These are descriptive measurements, not confidence intervals or a test for bias.

Run the short frozen-policy comparison from this directory:

```bash
uv sync --locked
uv run --locked python -m gridworld.gradient_probe run --output-dir runs/gradient-probe --device cuda
uv run --locked python -m gridworld.gradient_probe summarize runs/gradient-probe/probe.json
```

The fixed pilot uses **32 independent batches of 8 episodes**, seeds **0..255**,
in the same `SHG` example: horizon 2, slip 0, gamma=0.5, noise sigma=2, and
probabilities (0.1, 0.2, 0.3, 0.4). Every estimator uses the same episodes within
each batch. Full-return, reward-to-go, exact-V baseline, and the learned Monte
Carlo baseline all average whole episode sums, including zero logit rows.
The learned table is fitted once using seeds **1000..1031** and held fixed;
the results are conditional on that fit. Its **64** training transitions are
additional to the **512** shared probe transitions. Equal probe budgets do not
mean equal total data costs. There is no sweep or policy training.

`run` saves `probe.json` (all batch vectors, the exact gradient, coordinate
order, policy logits, world/reward settings, batch seeds, baseline values and
training counts/seeds, versions, device, and wall-clock times) and `summary.json`.
Use a new output directory for each run; existing outputs are preserved.
`summarize` reads only the raw JSON and prints the same summary, without sampling,
fitting, exact-model evaluation, or GPU access. For example, verify replay with:

```bash
uv run --locked python -m gridworld.gradient_probe summarize runs/gradient-probe/probe.json > runs/gradient-probe/recomputed-summary.json
cmp runs/gradient-probe/summary.json runs/gradient-probe/recomputed-summary.json
```

The notebook's **Measure repeated batch gradients** section works a two-coordinate
example by hand, runs this pilot on CUDA, and plots the finite-sample MSE
decomposition. Notebook execution keeps raw data in memory and writes no files;
the command above saves it. `--device cpu` provides a reference run. The pilot's
observed MSEs are 3.0085 (full return), 1.9761 (reward-to-go), 0.9865 (exact V),
and 1.0686 (learned baseline). They describe this one setting and fit, not a
universal ranking. The earlier exact-expectation checks establish unbiasedness;
a nonzero finite-sample discrepancy does not overturn them.

`gridworld.monte_carlo_sweep` extends that measurement to a declared **3 by 3
sweep**: N=(1,8,32) episodes per batch, reward noise sigma=(0,2,4), and B=128
independent batches per setting. One policy and one separately learned baseline
stay fixed across all settings. `common.gradient_metrics.summarize_comparison`
adds paired whole-batch bootstrap intervals for signed bias, sample variance,
MSE, and MSE differences from reward-to-go. These are approximate pointwise
intervals, conditional on one fit; squared empirical bias remains descriptive.

The notebook's **Compare noise and episode count** section reads the
[selected figures and saved summary](figures/gridworld/monte_carlo/README.md),
so viewing the results does not recollect the sweep. That page gives captions,
interpretation, budgets, and all settings. To collect and then replay from this
directory (each output directory must be new):

```bash
uv sync --locked
uv run --locked python -m gridworld.monte_carlo_sweep run --pilot --output-dir runs/mc-pilot --device cuda
uv run --locked python -m gridworld.monte_carlo_sweep run --output-dir runs/mc-sweep --device cuda
uv run --locked python -m gridworld.monte_carlo_sweep report runs/mc-sweep/sweep.json --output-dir runs/mc-replot
cmp runs/mc-sweep/summary.json runs/mc-replot/summary.json
```

`run` saves raw vectors in `sweep.json`, all summary statistics and interval
endpoints in `summary.json`, and two PNGs. `report` rebuilds the summary and
figures using only the raw JSON; it needs no GPU, sampling, fitting, or exact
model evaluation. The pilot uses separate seeds and is excluded from the
selected results. Full collection uses 15,744 shared probe episodes (31,488
transitions), plus one 32-episode fit (64 transitions). The observed full command
took 35.8 seconds on the RTX 3090; this is wall-clock observation, not a hardware
benchmark. `runs/` remains ignored. No TD/GAE or neural training is included.

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
