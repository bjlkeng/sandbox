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
`gridworld/` holds the explicit model, tabular policy, episode sampler, and local
player. Shared learning algorithms and additional game packages follow the
layout in [PLAN.md](PLAN.md) as their experiments are implemented.

Install `requirements.txt` in your Python environment. From this directory, run:

```bash
python3 -m unittest discover -s tests -v
python3 -m gridworld.play
```

The player is available at <http://127.0.0.1:8000>; use `--port` to choose another port.
The worked example is in [notebooks/01_gridworld.ipynb](notebooks/01_gridworld.ipynb).
