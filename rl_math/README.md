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

This first example uses only the Python standard library (Python 3.10+).
From `rl_math/`, run the focused checks with:

```bash
python3 -m unittest discover -s tests -v
```

### Reward indexing and the objective

Use transitions $(s_t, a_t, r_{t+1}, s_{t+1})$: action $a_t$ produces
`rewards[t] = r_{t+1}`. An episode has $T$ actions, rewards $r_1,\ldots,r_T$,
and terminal state $s_T$. Define the realized return and expected objective as

$$
G_t = \sum_{k=t}^{T-1}\gamma^{k-t}r_{k+1},\qquad
G_T = 0,\qquad
J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[G_0].
$$

The final reward is included: $G_{T-1}=r_T$. An empty episode has total return
zero. In [returns.py](returns.py), `discounted_returns` computes this backward
using `future_return = rewards[t] + gamma * future_return`. It accepts one
complete episode; a rollout cutoff with future rewards needs a bootstrap value
and is outside this helper's contract.

For the deterministic three-reward trajectory `[1, 2, 3]`, $J=G_0$:

| $t$ | `rewards[t]` ($r_{t+1}$) | $G_t$, $\gamma=1$ | $G_t$, $\gamma=0.5$ | $\gamma^tG_t$, $\gamma=0.5$ |
| --- | --- | --- | --- | --- |
| 0 | 1 | $1+2+3=6$ | $1+0.5(2)+0.25(3)=2.75$ | $2.75$ |
| 1 | 2 | $2+3=5$ | $2+0.5(3)=3.5$ | $0.5(3.5)=1.75$ |
| 2 | 3 | $3$ | $3$ | $0.25(3)=0.75$ |

Thus $J=6$ for $\gamma=1$, and $J=2.75$ for $\gamma=0.5$.
For stochastic trajectories, $G_0$ is one sample of $J$.

### Why reward-to-go still needs the outer discount

Let $z_t=\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ be the policy score.
Assume the environment and start-state distribution do not depend on $\theta$.
The likelihood-ratio gradient is $\nabla J=\mathbb{E}[\sum_t z_tG_0]$.
Split the return into rewards before and after action $a_t$:

$$
G_0 = \underbrace{\sum_{k<t}\gamma^kr_{k+1}}_{\text{past rewards}}
    + \gamma^tG_t.
$$

Given the history through $s_t$, past rewards are fixed and the conditional
mean score is zero: $\sum_a\pi_\theta(a\mid s_t)\nabla\log\pi_\theta(a\mid s_t)
=\nabla\sum_a\pi_\theta(a\mid s_t)=0$.
Their contribution therefore vanishes **in expectation**, giving

$$
\nabla J(\theta) = \mathbb{E}\left[
  \sum_{t=0}^{T-1}\gamma^tG_tz_t\right],\qquad
\hat g = \frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T_i-1}
  \gamma^tG_t^{(i)}z_t^{(i)}.
$$

`reward_to_go_weights` returns these $\gamma^tG_t$ coefficients. $G_t$ discounts
from time $t$; the outer factor puts it back on the episode-start clock.
For example, action $a_2$ above receives coefficient $0.75$, even though its
local return is $3$. At $\gamma=1$ the outer factor is one. At $\gamma=0$ only
the first action contributes to this objective.

### Sum within each episode, average over episodes

Here is an executable scalar-score example. A Bernoulli policy has
$p=\operatorname{sigmoid}(\theta)$ and score $z_t=a_t-p$. At the first step,
action 0 ends with reward 2; action 1 gives reward -1 and continues. At the
second step, action 0 ends with reward 1 and action 1 ends with reward 4.
At $\theta=0$, the paths have probabilities $1/2$, $1/4$, and $1/4$; the batch
below represents these exactly by repeating the short path twice.

```python
from returns import discounted_returns, reward_to_go_weights

assert discounted_returns([1, 2, 3], gamma=0.5) == [2.75, 3.5, 3.0]
assert reward_to_go_weights([1, 2, 3], gamma=0.5) == [2.75, 1.75, 0.75]

# (rewards, scalar policy scores), with one score per action.
episodes = [
    ([2], [-0.5]),
    ([2], [-0.5]),
    ([-1, 1], [0.5, -0.5]),
    ([-1, 4], [0.5, 0.5]),
]
episode_sums = [
    sum(w * z for w, z in zip(reward_to_go_weights(rewards, 0.5), scores))
    for rewards, scores in episodes
]
gradient = sum(episode_sums) / len(episodes)
assert episode_sums == [-1.0, -1.0, -0.5, 1.5]
assert gradient == -0.25
```

Independently, this tree has $J=(1-p)2+p[-1+\gamma((1-p)1+p4)]$ and
$dJ/d\theta=p(1-p)[-3+\gamma+6\gamma p]$. At $p=\gamma=0.5$ this is $-0.25$.
Dividing the batch sum by its six transitions instead of its four episodes
gives $-1/6$. Averaging each episode over its own length gives $-0.375$.
Neither matches this derivative. Vector policy scores follow the same sum.

The tests enumerate this tree at three policy settings and three discounts,
comparing the weighted score sum to finite differences of the independent
objective. They also check terminal and empty tails and the worked returns.
All checks are deterministic; no sampling seeds or training are needed.

### Policy Bellman expectation and local draft corrections

With remaining time included in the state for a finite-horizon task,
$V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]$ and
$V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)$. Consequently,

$$
Q^\pi(s,a)=\mathbb{E}\left[r_{t+1}
  +\gamma\sum_{a'}\pi(a'\mid s_{t+1})Q^\pi(s_{t+1},a')
  \mid s_t=s,a_t=a\right].
$$

The future value is zero at termination. For the optimal value function,
replace the policy average with $\max_{a'}Q^*(s_{t+1},a')$.
For example, next-action values $(2,10)$ under probabilities $(0.75,0.25)$
give policy value 4; their maximum is 10. These evaluate different behavior.
See the [Spinning Up Bellman equations](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#bellman-equations).

These conventions give the following local corrections to the
[reviewed draft](https://github.com/bjlkeng/bjlkeng.github.io/blob/05ef622b7efba63411f1f6c8322f99e6ffd9412d/source/posts/rl-math.rst):

* Equation 7 defines a realized return; its expectation defines $V^\pi$ or $J$.
  In Equation 8 and the one-step expressions in Equations 12–14, use $r_{t+1}$
  to match Equation 7 and the transition convention.
* Equation 13's policy-specific line needs the policy average above.
* Retain $\gamma^t$ when moving from Equation 20 to the episodic estimator in
  Equation 21. $N$ counts episodes, each contributing a sum over its actions.
  An occupancy measure $\sum_t\gamma^tP(s_t=s)$ is unnormalized; writing an
  expectation under its normalized distribution requires its total-mass factor.
* In Algorithm 1, iterate $t=T-1,\ldots,0$, starting with $G_T=0$, and use
  $r_{t+1}$ (stored at `rewards[t]`). Appendix A's past-reward sum is
  $\sum_{k<t}\gamma^k r_{k+1}$, as in the decomposition above.
