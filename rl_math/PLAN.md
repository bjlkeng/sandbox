# Reinforcement learning primer: supporting experiments

Supporting-code plan for the draft
[A Reinforcement Learning Primer](https://github.com/bjlkeng/bjlkeng.github.io/blob/post-rl-math/source/posts/rl-math.rst).

This file records the experiment design. Create implementation tasks only when
requested, in this subproject's own Beads workspace.

## Goal and scope

Build small, reproducible experiments that connect the post's equations to
observable behavior. Use a tiny stochastic gridworld to measure estimation
effects against exact answers, CartPole for the first neural implementations,
and LunarLander for a more substantial visual comparison.

Focus on stochastic policy gradients: REINFORCE with full return, reward-to-go,
and a state baseline; one-step and n-step actor-critic; generalized advantage
estimation (GAE); and PPO-Clip. GAE and PPO are recommended additions to the
draft, with experiments staged after the basic actor-critic implementation.
Keep V, Q, Bellman equations, and advantages as foundations for policy gradients
and critic learning. Standalone value-based control experiments are outside
this post's scope.

These experiments should fit comfortably on a machine with an RTX 3090. Start
with small networks and vector observations. Benchmark CPU execution before
moving training to CUDA; simulation and small batches may dominate runtime.
Training budgets below are starting points, not measured runtimes or guarantees
of convergence.

## Recommended Policy Gradients outline

The [reviewed draft](https://github.com/bjlkeng/bjlkeng.github.io/blob/05ef622b7efba63411f1f6c8322f99e6ffd9412d/source/posts/rl-math.rst)
already has the policy-gradient theorem, an estimator/credit-assignment overview,
REINFORCE with a baseline planned, and an empty Actor Critic subsection. Extend
that sequence around the following main ideas:

1. **Policy-gradient theorem and credit assignment.** Retain the derivation and
   estimator overview as the common framework.
2. **REINFORCE, reward-to-go, and baselines.** Explain causal credit assignment
   and why subtracting an action-independent state baseline preserves the
   expected gradient. A learned Monte Carlo baseline does not require
   bootstrapping.
3. **Actor-critic and temporal differences.** Introduce bootstrapping with a
   learned value function, then one-step and n-step advantage estimates. Mention
   A2C (synchronous advantage actor-critic) and A3C (asynchronous advantage
   actor-critic) here: their batching and worker-update arrangements fit in a
   short explanation, without separate derivations or implementations.
4. **Generalized advantage estimation.** Give GAE a short subsection or place it
   at the end of actor-critic. It connects the Monte Carlo/TD bias-variance
   trade-off through a weighted mixture of multi-step estimates. It estimates
   the learning signal and can be paired with different policy optimizers;
   it is not a separate policy-update algorithm. See the [GAE paper](https://arxiv.org/abs/1506.02438).
5. **Controlling policy updates: PPO.** Explain why an accurate gradient can
   still produce a damaging finite update. Briefly motivate natural policy
   gradients and [TRPO](https://arxiv.org/abs/1502.05477) through policy-space
   distance/KL divergence, then introduce the probability ratio and clipped
   surrogate of [PPO](https://arxiv.org/abs/1707.06347). This adds the distinct
   idea of controlling policy changes while taking several optimization steps
   on a fresh rollout batch. PPO deserves more than an implementation footnote;
   a full natural-gradient or TRPO derivation is unnecessary here.

The estimator overview describes choices of learning signal; it should not
imply that every algorithm differs only by substituting a different Psi_t.
GAE changes advantage estimation, PPO changes the actor objective, and A2C/A3C
also concern how experience and updates are organized.

Include brief context on exploration by sampling the policy, entropy bonuses,
and on-policy vs. off-policy data. Distinguish that last pair from online vs.
offline training. PPO still uses fresh on-policy rollouts, despite reusing each
batch for several epochs. Detailed off-policy corrections are outside scope.

For a short closing map, name the other substantial branches:
[deterministic policy gradients](https://proceedings.mlr.press/v32/silver14.html)
(leading to DDPG/TD3 for continuous actions), and maximum-entropy off-policy
actor-critic such as [SAC](https://proceedings.mlr.press/v80/haarnoja18b.html).
These introduce additional mathematical ideas, but neither needs a subsection
or experiment in this primer. A small entropy bonus in actor-critic does not
by itself cover SAC's maximum-entropy formulation.

## Algorithm-to-problem mapping

| Algorithm or concept | Problem | Main demonstration |
| --- | --- | --- |
| Bellman equations, V, Q, advantage | Stochastic gridworld | Exact values, policy arrows, and action advantages |
| Full-return REINFORCE vs. reward-to-go | Noisy gridworld; then CartPole | Gradient variance and the effect of past rewards |
| REINFORCE with a state baseline | Same problems | Variance reduction without changing the expected gradient |
| One-step and n-step actor-critic | Gridworld; then CartPole and LunarLander | Bootstrapping, critic error, and updates before an episode finishes |
| GAE | Gridworld; then the shared actor-critic implementation | Advantage-estimation bias/variance as lambda changes |
| PPO-Clip | CartPole smoke run; then LunarLander | Probability ratios, conservative updates, and rollout-batch reuse |

## Phase 1: exact answers in a stochastic gridworld

### Environment

Create a small, fully observed gridworld, initially around 5 by 5 cells, with:

* A start, terminal goal, walls, and hazardous cells.
* A short hazardous route and a longer safe route.
* A configurable probability of moving in an unintended direction.
* A per-step cost, goal reward, and hazard penalty.
* Configurable independent, zero-mean reward noise. This changes sampling
  variance while leaving expected rewards unchanged.

Use a short finite horizon and include remaining time in the state. This keeps
the problem Markov and permits exact backward dynamic programming. Display
value and policy maps at a specified remaining horizon.

Include intermediate rewards: with only a terminal reward, full-return and
correctly discounted reward-to-go gradient estimators coincide, hiding the
comparison of interest.

### Exact reference quantities

For a fixed tabular softmax policy, compute V, Q, advantage, and expected
discounted return using the known transition and reward model. Differentiate
the exact expected return with respect to the policy logits to obtain a
reference policy gradient.

Check Bellman consistency, the policy-weighted mean of the advantage, and
agreement between the exact gradient and finite differences on a small case.
These reference checks establish whether the sampling experiments are measuring
the intended quantities.

### Gradient-estimator experiment

Freeze the policy and repeatedly collect independent batches of episodes.
Compare estimators using:

1. Full trajectory return.
2. Reward-to-go.
3. Reward-to-go minus the exact state-value baseline.
4. Reward-to-go minus a learned state baseline.
5. One-step TD advantage using the exact value function.
6. One-step and n-step TD advantages using an imperfect learned critic.
7. GAE over a small lambda sweep, using exact and learned critics.

Measure empirical bias, variance, and mean squared error against the exact
gradient at equal episode counts. Vary batch size and reward-noise strength.
Hold each baseline and critic fixed during a measurement; fit learned baselines
on separate or earlier data so the probe does not introduce dependencies by
fitting to its own sampled returns.

An imperfect state-only baseline can preserve an unbiased Monte Carlo policy
gradient, while an imperfect bootstrapped value estimate can introduce bias.
The exact-critic TD estimator provides a useful control. Measure the variance
changes rather than assuming every estimator improves them in every setting.

Use `delta_t = r_{t+1} + gamma * V(s_{t+1}) - V(s_t)` and compute finite-rollout
GAE backward as `Ahat_t = delta_t + gamma * lambda * Ahat_{t+1}`, stopping the
recursion at episode boundaries. With complete episodes and zero terminal
value, lambda=0 gives the one-step TD residual and lambda=1 telescopes to
`G_t - V(s_t)`. With a rollout cutoff, lambda=1 includes the bootstrapped final
value instead. Probe a few intermediate values to show how critic error and
the estimation horizon interact.

### Blog artifacts

* Grid diagram with representative trajectories.
* Value heatmaps, policy arrows, and action-advantage maps.
* Estimator bias and variance against sample count and reward noise.
* A comparison showing how critic error affects bootstrapped estimates.
* A compact GAE bias/variance plot across lambda values.

## Phase 2: CartPole neural implementations

Use [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/),
which has four observed state variables, two discrete actions, and a default
reward of +1 per step.

Implement the algorithms in small, readable PyTorch modules, keeping the update
rules close to the equations in the post. Start with a small MLP, such as two
64-unit hidden layers, and a categorical action distribution for the policy.

Compare:

* REINFORCE with full trajectory return.
* REINFORCE with reward-to-go.
* REINFORCE with reward-to-go and a learned state baseline.
* One-step and n-step actor-critic.
* Actor-critic with GAE, reusing the actor and critic code.

Keep actor architectures fixed across variants. Use consistent settings within
estimator ablations, while allowing documented, limited tuning for different
algorithms. A small synchronous rollout batch is sufficient; implementing
asynchronous workers is outside scope.

Start with roughly 100,000 environment interactions per run and increase the
budget if necessary. Report the final budgets. Use several independent seeds;
aim for 5 to 10 for the main learning-curve comparisons.

### PPO after the actor-critic checks

Reuse the same categorical policy, value network, rollout collection, and GAE
helper for a readable PPO-Clip implementation. Save action log-probabilities
under the collecting policy and hold the old log-probabilities and computed
advantages fixed over the batch's optimization epochs. Define
`ratio_t = pi_theta(a_t | s_t) / pi_old(a_t | s_t)` and maximize the average of
`min(ratio_t * Ahat_t, clip(ratio_t, 1-epsilon, 1+epsilon) * Ahat_t)`.
Use the explicit time weights below when targeting the discounted start-state
objective.

First plot the clipped surrogate for positive and negative advantages. Then
run a short CartPole smoke experiment to verify collection, repeated minibatch
updates, and evaluation. Save the main optimizer comparison for LunarLander.
Use fresh rollouts after each optimization cycle; this is limited batch reuse,
not an arbitrary historical replay buffer.

Track approximate old-to-new policy KL, clipping fraction, and policy entropy.
Clipping removes incentives for some large probability changes; it does not
enforce a hard KL bound or guarantee monotonic improvement. An explicitly
documented KL early-stop threshold can provide an additional safeguard, as in
the [Spinning Up PPO implementation](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

### Blog artifacts

* Evaluation return vs. environment interactions, including variation across seeds.
* Gradient variability during training.
* Policy slices over pole angle and angular velocity, fixing other inputs.
* Short recordings before and after learning.
* A two-panel PPO surrogate plot for positive and negative advantages.

CartPole is useful for fast iteration, but its performance ceiling may hide
differences once several algorithms solve it. Carry the stronger comparison
to LunarLander.

## Phase 3: LunarLander comparison

Use the discrete-action
[LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
environment with its standard state-vector observations and default rewards.
It has eight observed state variables and four actions. Rewards account for
position, velocity, orientation, engine use, and landing or crashing.

Compare REINFORCE with a baseline, actor-critic, and PPO-Clip on the same task.
Use GAE with the same lambda and rollout length for the focused actor-critic
vs. PPO optimizer comparison, and document their actor-update counts. Keep the
one-step/n-step/lambda sweeps in the gridworld and CartPole rather than
repeating the full matrix here.
Optionally retain plain REINFORCE as a reference; successful training with that
variant is not a requirement for completing the article's core experiments.

Start with a budget around one million environment interactions per run, then
adjust based on pilot runs. Record any budget or hyperparameter changes before
running the final seed comparisons. Train from state vectors and render only
for evaluation or recordings.

### Blog artifacts

* Evaluation-return curves across seeds at equal interaction budgets.
* Recordings of early behavior, intermediate progress, and final policies.
* Landing success, crash frequency, and episode length alongside reward.
* Policy KL, clipping fraction, and entropy alongside PPO learning curves.
* If useful, one focused PPO clipping ablation with the same GAE estimates and
  optimization budget; report observed stability rather than assuming a winner.

## Optional extensions

### Applied alternative: inventory replenishment

A simulated store could replace or supplement the control tasks:

* State: stock, outstanding orders, and any demand information.
* Action: a discrete order quantity.
* Reward: revenue minus ordering, holding, and stockout costs.
* Dynamics: stochastic demand and delivery delays.

Include the full delivery pipeline in the state so the process is Markov.
A small bounded version permits exact dynamic programming and comparison with
an order-up-to heuristic and the same policy-gradient algorithms.
This needs more environment design and explanation, so keep it outside the
initial implementation.

## Mathematical and evaluation conventions

Before implementing the estimators, settle the reward indexing in the post.
Use one convention throughout code and figures: transitions
`(s_t, a_t, r_{t+1}, s_{t+1})`, with return G_t beginning at r_{t+1}.

For the discounted start-state objective, retain the outer gamma^t factor in
the episodic policy-gradient sum. Average episode-level gradient sums over
episodes. Apply the corresponding gamma^t weighting to PPO surrogate terms
when using this objective, with t measured from the episode start rather than
the rollout start. Common implementations instead use uniformly weighted
rollout transitions; document that choice if used for neural benchmarks and
do not treat it as an exact gradient of the discounted start-state objective.
PPO's surrogate uses states sampled under the collecting policy, so its
finite updates are not exact maximization of the true return either.

Correct Equation 13's policy-specific Bellman equation before translating it
into code: the next action is averaged under the policy. The maximum over
next actions belongs to the optimality equation. See the
[Bellman equation reference](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#bellman-equations).

Reconcile the draft's estimator table with its discounted definitions: exact
Q/advantage and the exact-value TD residual give unbiased policy-gradient
estimates under the stated sampling/weighting conventions. Approximate
bootstrapped critics can introduce bias. Variance rankings depend on the
problem and baseline quality; they are experimental questions, not universal
inequalities.

During training, keep baseline and advantage targets detached from the actor
gradient. Distinguish genuine environment termination from rollout cutoffs or
external time-limit truncation when deciding whether to bootstrap. An explicitly
defined finite task horizon, as in the gridworld, is part of the task itself.
Bootstrap at an external truncation from the final observation, but stop the
GAE recursion before a reset episode; at genuine termination, use zero future
value. Keep rollout-time critic predictions fixed while constructing targets.
Record any advantage normalization or entropy regularization, and disable
them in the exact-gradient probes so those comparisons retain their meaning.

Use separate evaluation episodes, fixed evaluation settings, and saved seed
lists. Plot results across runs rather than selecting the best seed. Report
environment interactions and wall-clock time separately, and retain raw metrics
so figures can be regenerated without retraining.

## Suggested directory layout

Create these files as their corresponding phases are implemented:

```text
rl_math/
  README.md
  PLAN.md
  requirements.txt
  envs.py                  # Custom gridworld and shared environment helpers
  reinforce.py             # Monte Carlo policy-gradient variants
  actor_critic.py           # One-step and n-step methods
  advantages.py             # Shared n-step and GAE target computation
  ppo.py                    # Clipped surrogate and rollout-batch optimization
  train.py                 # Common experiment and evaluation entry point
  notebooks/
    01_gridworld.ipynb
    02_cartpole.ipynb
    03_lunarlander.ipynb
  figures/                 # Selected figures for the post
  runs/                    # Generated metrics, checkpoints, and recordings
```

Keep generated training outputs out of Git; retain the small source files,
experiment configurations, notebooks, and selected publication artifacts.
Notebooks should explain and visualize experiments, with shared training logic
in Python modules.

## Task tracking

Create implementation tasks from the phases above in `rl_math/.beads/` when
requested. Use that workspace for live task status and dependencies. The
previous root-tracker epic and tasks were removed; they should not be recreated
automatically.

## Implementation references

* [Gymnasium environments](https://gymnasium.farama.org/environments/classic_control/)
* [SB3 evaluation and experiment guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
* [SB3 A2C CPU guidance](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)
* [GAE paper](https://arxiv.org/abs/1506.02438)
* [Asynchronous actor-critic paper (A3C)](https://arxiv.org/abs/1602.01783)
* [TRPO paper](https://arxiv.org/abs/1502.05477)
* [PPO paper](https://arxiv.org/abs/1707.06347)
* [Spinning Up PPO equations and implementation](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
* [RL Zoo PPO configurations](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml)

Use library configurations as scale references for pilot runs. The readable
implementations in this project may need different settings or budgets.
