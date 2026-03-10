"""
Run MC (500 eps) + IS (500 eps) + CEM (500 eps) for DQN.
Overwrites previous results with matched-budget versions.
"""

from evaluate import (
    run_mc_disturbance_episodes, run_is_episodes,
    is_failure_prob, save_results,
)
from cem import run_cem

N = 500

# ── MC ────────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  DQN — MC uniform ({N} episodes)")
print(f"{'='*50}")
mc = run_mc_disturbance_episodes(agent="dqn", n_episodes=N)
save_results(mc, "dqn_mc_disturbance.csv")
print(f"MC crash rate: {sum(r['crashed'] for r in mc)/N:.4f}")

# ── IS ────────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  DQN — Importance Sampling ({N} episodes)")
print(f"{'='*50}")
is_ = run_is_episodes(agent="dqn", n_episodes=N)
save_results(is_, "dqn_is.csv")
print(f"IS  P(crash): {is_failure_prob(is_):.4f}")
print(f"IS  raw crashes: {sum(r['crashed'] for r in is_)}/{N}")

# ── CEM ───────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  DQN — CEM (5 iters × 100 samples = {N} episodes)")
print(f"{'='*50}")
cem = run_cem(agent="dqn")
save_results(cem, "dqn_cem.csv")
print(f"CEM crash rate: {sum(r['crashed'] for r in cem)/N:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("  Summary (DQN, 500 episodes each)")
print(f"{'='*50}")
print(f"  MC  crashes: {sum(r['crashed'] for r in mc)}/{N}")
print(f"  IS  crashes: {sum(r['crashed'] for r in is_)}/{N}  (IS P(crash)={is_failure_prob(is_):.4f})")
print(f"  CEM crashes: {sum(r['crashed'] for r in cem)}/{N}")
print("\nDONE")
