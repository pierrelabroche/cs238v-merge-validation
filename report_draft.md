# Efficient Failure Mode Discovery in Autonomous Highway Merge Negotiation

**Pierre Labroche** | CS238v — Validation for Safety Critical Systems | Stanford University

---

## Abstract

We investigate efficient failure mode discovery for two autonomous driving agents — a rule-based IDM+MOBIL controller and a learned Deep Q-Network (DQN) — in a highway merge scenario. We parameterize a disturbance space over merging vehicle speed and position and compare three falsification approaches: Monte Carlo sampling, Importance Sampling (IS), and the Cross-Entropy Method (CEM). A coarse grid sweep reveals qualitatively different failure regions for each agent: IDM fails under very slow late-appearing mergers, while DQN fails when mergers appear early, requiring anticipation rather than reaction. IS with a manually specified proposal underperforms uniform MC due to the multi-modal failure structure of DQN, directly motivating CEM. CEM finds failures in every iteration and adaptively converges toward the correct failure region without prior knowledge. Our results show that for learned agents, adaptive falsification is most sample-efficient in early exploration, while for highly reliable rule-based agents, failure discovery remains inherently sample-intensive regardless of algorithm — with even 1000 episodes finding only one IDM crash, at a configuration the grid sweep missed entirely.

---

## 1. Introduction

Validating autonomous driving systems requires demonstrating safety not just under nominal conditions but under adversarial or unusual disturbances. Traditional testing approaches — fixed scenario suites or random simulation — scale poorly with the rarity of failure events. For a system with a 0.3% crash rate, naive Monte Carlo requires thousands of episodes to surface failures with statistical confidence.

This paper studies the highway merge scenario, where an ego vehicle must maintain high speed while yielding to vehicles entering from an access ramp. The competing objectives — speed and collision avoidance — produce rich failure modes that differ qualitatively between agent types. We evaluate two agents: IDM+MOBIL, a classical rule-based controller, and DQN, a learned policy trained via deep Q-learning. Against each, we apply a progression of falsification methods of increasing sophistication: Monte Carlo (MC), Importance Sampling (IS), and the Cross-Entropy Method (CEM).

The central contribution is a comparative study of how efficiently each method discovers failures as the disturbance space is stressed, and what the discovered failure modes reveal about each agent's behavioral policy.

---

## 2. Related Work

**Falsification and stress testing.** Kochenderfer et al. [CITATION] survey falsification through optimization (Ch. 4), planning (Ch. 5), and failure probability estimation (Ch. 7), providing the algorithmic foundations for CEM, MCTS, and IS respectively used in this work.

**Adaptive Stress Testing.** Lee et al. [CITATION] introduce AST, which frames failure-finding as an RL problem where an adversary applies per-timestep disturbances with a likelihood-weighted reward, finding the most probable failures rather than any failure. AST is the principled extension of our approach to sequential disturbances — our one-shot disturbance parameterization precludes direct application, but the framework motivates future work.

**Highway-env.** The merge-v0 environment [CITATION] provides a lightweight, configurable autonomous driving simulation with discrete meta-actions and kinematic observations, widely used for RL research in autonomous driving.

---

## 3. System and Environment

### 3.1 Merge Scenario

We use the `merge-v0` environment from highway-env [CITATION]. The scenario places an ego vehicle on a two-lane highway approaching a merge point, where vehicles enter from a sinusoidal access ramp. The ego must maintain high speed while making room for merging traffic — competing objectives that produce structured failure modes absent in simpler highway scenarios.

Episodes terminate when the ego vehicle collides with another vehicle or passes position x=370m. The action space consists of five discrete meta-actions: IDLE, LANE\_LEFT, LANE\_RIGHT, FASTER, and SLOWER. Observations are kinematic features of the ego and up to four neighboring vehicles.

The reward combines speed incentive, right-lane preference, lane-change penalty, and an altruistic term penalizing slow merging vehicles:

$$r = -1 \cdot \mathbf{1}[\text{crash}] + 0.2 \cdot \tilde{v} + 0.1 \cdot \mathbf{1}[\text{right lane}] - 0.05 \cdot \mathbf{1}[\text{lane change}] - 0.5 \cdot \sum_{\text{merging}} \frac{v^* - v}{v^*}$$

where $\tilde{v}$ is ego speed linearly scaled to [0,1] over [20, 30] m/s.

### 3.2 Agents

**IDM+MOBIL** is a classical rule-based controller. Longitudinal behavior follows the Intelligent Driver Model (IDM), which produces smooth car-following acceleration as a function of gap and relative velocity. Lateral behavior follows MOBIL, which triggers lane changes when they reduce braking impact across all surrounding vehicles. Key parameters: ACC\_MAX=6 m/s², COMFORT\_ACC\_MAX=3 m/s², TIME\_WANTED=1.5s. IDM+MOBIL requires no training and is fully deterministic.

**DQN** is a learned policy trained via Deep Q-Network with an MLP architecture using Stable Baselines3. We train for 100,000 timesteps and evaluate on 5 independent training runs (200 episodes each, 1000 total) to characterize variance. At 100k timesteps the policy has not fully converged, producing high run-to-run variance (±3.0% crash rate).

### 3.3 Disturbance Space

We parameterize disturbances over two properties of the merging vehicle:
- **merging\_speed** ∈ [10, 25] m/s — initial speed at episode start (default: 20 m/s)
- **merging\_position** ∈ [70, 130] m — position along the access ramp (default: 110 m)

Disturbances are applied after `env.reset()` by patching the merging vehicle's speed, position, and heading directly, without modifying the environment source. The nominal distribution p(d) is uniform over this support.

### 3.4 Failure Conditions

We track two failure types:

1. **Safety failure** — ego vehicle collides with any vehicle (crash)
2. **Liveness failure** — ego vehicle speed drops below V\_TARGET = 17 m/s
   - Opt 1: minimum speed ever below threshold
   - Opt 2a: average episode speed below threshold

V\_TARGET = 17 m/s was selected because IDM's average minimum speed is 17.6 m/s — at 15 m/s the average-speed metric was zero for both agents, providing no liveness signal.

---

## 4. Approach

### 4.1 Monte Carlo Baseline

The MC baseline samples disturbances uniformly from p(d) and rolls out one episode per sample. The failure probability estimate is:

$$\hat{P}_{\text{MC}}(\text{fail}) = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[f(d_i)]$$

MC is unbiased and requires no prior knowledge of the failure space, but converges slowly when failures are rare.

### 4.2 Importance Sampling

IS samples from a proposal distribution q(d) biased toward dangerous configurations and reweights outcomes to maintain an unbiased estimate:

$$\hat{P}_{\text{IS}}(\text{fail}) = \frac{1}{N} \sum_{i=1}^{N} \frac{p(d_i)}{q(d_i)} \cdot \mathbf{1}[f(d_i)], \quad d_i \sim q$$

We use independent truncated normals for each disturbance parameter. Proposals are informed by the grid sweep: for DQN, q is biased toward low position (μ=80m, σ=15m) and low speed (μ=15 m/s, σ=5 m/s). IS is only as good as its proposal — a misspecified proposal can perform worse than MC, as we observe.

### 4.3 Cross-Entropy Method

CEM iteratively fits a distribution over disturbance parameters toward failure-causing configurations. Each iteration:

1. Sample a batch of N disturbances from the current distribution TruncNorm(μ, σ)
2. Roll out N episodes and collect outcomes
3. Select elites: crashed episodes only (distribution held fixed if no crashes found)
4. Update μ, σ from elite means and standard deviations with exponential smoothing (α=0.5)
5. Clamp σ to minimum floors (speed: 2 m/s, position: 10 m) to prevent collapse

We use N=100 samples per iteration, 5 iterations (500 total episodes), initialized at the center of the disturbance space (μ\_spd=17.5, μ\_pos=100). The "hold when no crashes" elite rule prevents the distribution from being misled by proxy metrics when failures are absent.

### 4.4 Design Decisions

**CEM over MCTS.** MCTS is naturally suited to sequential disturbance problems where an adversary acts at each timestep. Our disturbance is one-shot — set once at episode start — making CEM's direct distribution optimization more appropriate. MCTS would require discretizing the continuous disturbance space and a rollout policy that provides almost no signal at 3% failure rates.

**IS as intermediate method.** IS sits naturally between MC and CEM in complexity and serves as a narrative bridge. Its observed failure on this problem — misspecified proposal, fewer crashes than MC — directly motivates CEM's adaptive approach.

---

## 5. Results

### 5.1 Baseline: IDM vs DQN

Figure 1 shows failure rates for both agents under their default environments (200 IDM episodes, 5×200 DQN episodes). IDM never crashes (0%) but frequently brakes below 17 m/s (26.5% opt1, 2% opt2a), sacrificing speed to yield to mergers. DQN crashes in 4.6% of episodes (±3.0% across training runs) but almost never drops below 17 m/s — it learns aggressive behavior that maintains throughput at the cost of safety.

The high DQN variance (±3.0%) reflects underfitting at 100k training timesteps. The safety-performance tradeoff is the central behavioral finding: IDM optimizes for liveness at the expense of safety; DQN optimizes for speed at the expense of crash avoidance.

### 5.2 Disturbance Grid Sweep

Figure 2 shows crash rate heatmaps across the 4×4 disturbance grid (20 trials/cell). IDM fails only at (speed=10, position=130) — a very slow merger appearing late at the merge point, leaving little reaction time. DQN exhibits a multi-modal failure pattern: crashes concentrate at low position (70–90m) across most speeds, with a secondary mode at (speed=25, position=110m). This reveals a qualitative behavioral difference: IDM fails when it cannot react fast enough; DQN fails when it must anticipate a merger that appears early, suggesting the policy learned reactive rather than predictive behavior.

### 5.3 Sample Efficiency: MC vs IS vs CEM

Figure 3 shows cumulative crashes found over 500 episodes for each method applied to DQN. CEM finds crashes fastest in early episodes — 7 crashes by episode 100, compared to ~2 for MC at the same point. Its distribution converges from μ\_pos=100 to μ\_pos=110 over 5 iterations, discovering the high-position failure region without prior knowledge.

MC accumulates crashes steadily and ends highest (20 crashes) due to its uniform coverage of the multi-modal failure space. IS, with a proposal biased toward low position (μ=80m), finds only 13 crashes — fewer than MC — because the proposal undersamples the high-position failure mode. The resulting IS estimate (3.3%) converges near MC's (4.0%) given enough samples, but raw failure discovery is worse.

For IDM, all three methods found zero crashes in 500 episodes. A dedicated 1000-episode MC run found one crash (0.1%) at episode 475 at configuration (speed=11.9, position=78) — a low-position failure mode not identified by the grid. This illustrates that for highly reliable agents, failure discovery is inherently sample-intensive: even adaptive methods cannot overcome the fundamental rarity of the event.

---

## 6. Discussion

**Multi-modal failure spaces limit single-proposal methods.** DQN's failure space has two distinct modes — low position (early merge) and high position (late merge at high speed). A single Gaussian proposal (IS) or single Gaussian CEM distribution cannot efficiently cover both simultaneously. This is where umbrella sampling or mixture IS would provide direct benefit: running separate proposals per mode and combining results. We leave this to future work.

**IS is only as good as its proposal.** The IS proposal was informed by the grid sweep, which suggested low-position failures dominated for DQN. Continuous MC sampling revealed that high-position failures are equally prevalent — a structure the 20-trial-per-cell grid could not reliably surface. This circular dependency (IS needs the failure structure to design a good proposal; discovering the failure structure requires many samples) is precisely the problem CEM solves adaptively.

**CEM characterizes failure regions, not just failure counts.** While MC found the most raw crashes (20), CEM's real value is in its converged distribution: μ\_pos=110, μ\_spd=15.5, converged entirely from the data without any prior. This distribution is a compact description of where DQN is most vulnerable — directly useful for targeted re-testing or policy improvement.

**Adaptive Stress Testing as a natural extension.** AST [CITATION] would address both the one-shot limitation and the most-likely-failure objective simultaneously. Extending the disturbance from a single episode-start perturbation to per-timestep adversarial control of the merging vehicle would enable AST's RL-based solver to find the most probable failure trajectories, rather than just any failure.

---

## 7. Conclusion

We compared Monte Carlo, Importance Sampling, and CEM for failure mode discovery in a highway merge scenario with two qualitatively different agents. CEM provides the best early sample efficiency and automatically characterizes failure regions without manual proposal design. IS with a misspecified proposal underperforms MC, illustrating the proposal sensitivity inherent to fixed-distribution methods. IDM's near-zero crash rate (0.1% in 1000 episodes) demonstrates that validation cost scales inversely with failure probability — a fundamental challenge that motivates per-timestep adversarial approaches like AST for highly reliable agents.

---

## References

- Kochenderfer, M. J., et al. *Algorithms for Validation* (2025). Ch. 4 (CEM), Ch. 5 (MCTS), Ch. 7 (IS).
- Lee, R., et al. (2020). Adaptive Stress Testing: Finding Likely Failure Events with Reinforcement Learning. *JAIR*, 69, 1165–1201.
- Leurent, E. (2018). highway-env: An environment for autonomous driving decision-making. GitHub.
