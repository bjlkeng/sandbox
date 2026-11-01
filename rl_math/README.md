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

## Discounted returns and episode gradient weights

[returns.py](returns.py) provides two helpers for complete episodes:

* `discounted_returns(rewards, gamma)` computes each `G_t`: the sum of remaining
  rewards, discounted relative to time `t`.
* `reward_to_go_weights(rewards, gamma)` computes `gamma**t * G_t`, putting those
  returns on the episode-start clock for policy gradients.

Use transitions `(s_t, a_t, r_{t+1}, s_{t+1})`, so `rewards[t] = r_{t+1}`.
The terminal state's future return is zero; the final transition's reward is
included. For example:

```python
from returns import discounted_returns, reward_to_go_weights

rewards = [1, 2, 3]
assert discounted_returns(rewards, gamma=1.0) == [6.0, 5.0, 3.0]
assert discounted_returns(rewards, gamma=0.5) == [2.75, 3.5, 3.0]
assert reward_to_go_weights(rewards, gamma=0.5) == [2.75, 1.75, 0.75]
```

At `gamma=0.5`, these returns are `G_0 = 1 + 0.5*2 + 0.25*3`,
`G_1 = 2 + 0.5*3`, and `G_2 = 3`. For this deterministic episode, `J = G_0`;
in general, the objective is `J = E[G_0]`. To estimate its policy gradient,
sum `gamma**t * G_t * grad(log pi(a_t | s_t))` within each episode, then average
those sums over episodes.

The policy Bellman equation averages next-action Q values under the policy;
the maximum over next actions belongs to the optimality equation.

Requires Python 3.10+ and only the standard library. From `rl_math/`, run:

```bash
python3 -m unittest discover -s tests -v
```
