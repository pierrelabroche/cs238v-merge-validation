"""
Cross-Entropy Method (CEM) for failure mode discovery.

Iteratively fits a distribution over disturbance parameters toward
failure-causing configurations. Each iteration:
  1. Sample n_samples disturbances from current TruncNorm distribution
  2. Run episodes, collect outcomes
  3. Select elite set (crashed episodes; fallback: lowest min_speed)
  4. Update distribution mean/std from elites (with smoothing)
  5. Repeat

Reference: Kochenderfer et al., Algorithms for Validation (2025), Ch. 4
"""

from dataclasses import dataclass
import numpy as np

from evaluate import (
    Disturbance, run_episodes, run_episodes_dqn, save_results,
    SPEED_LO, SPEED_HI, POS_LO, POS_HI, _trunc,
)


# ── CEM parameters ────────────────────────────────────────────────────────────

@dataclass
class CEMParams:
    n_samples:   int   = 100   # episodes per iteration — larger batch = better crash signal
    n_iter:      int   = 5     # iterations (total = n_samples * n_iter = 500)
    alpha:       float = 0.5   # smoothing: μ_new = α*elite_mean + (1-α)*μ_old

    # initial distribution — center of disturbance space
    speed_init_mean: float = 17.5
    speed_init_std:  float = 5.0
    pos_init_mean:   float = 100.0
    pos_init_std:    float = 20.0

    # minimum std to prevent distribution collapse
    speed_min_std: float = 2.0   # was 1.0 — keeps speed exploration alive
    pos_min_std:   float = 10.0  # was 5.0 — keeps position exploration alive


# ── Core CEM loop ─────────────────────────────────────────────────────────────

def run_cem(
    agent:  str      = "dqn",
    params: CEMParams | None = None,
) -> list[dict]:
    """
    Run CEM for one agent. Returns flat list of all episode results across
    all iterations, each row tagged with iteration number and current
    distribution parameters for post-hoc analysis.
    """
    if params is None:
        params = CEMParams()

    assert agent in ("idm", "dqn"), "agent must be 'idm' or 'dqn'"
    runner = run_episodes if agent == "idm" else run_episodes_dqn

    mu_spd, sig_spd = params.speed_init_mean, params.speed_init_std
    mu_pos, sig_pos = params.pos_init_mean,   params.pos_init_std

    all_results: list[dict] = []

    for it in range(params.n_iter):
        spd_rv = _trunc(mu_spd, sig_spd, SPEED_LO, SPEED_HI)
        pos_rv = _trunc(mu_pos, sig_pos, POS_LO,   POS_HI)

        # --- sample and evaluate batch ---
        batch: list[dict] = []
        for _ in range(params.n_samples):
            d = Disturbance(
                merging_speed=float(spd_rv.rvs()),
                merging_position=float(pos_rv.rvs()),
            )
            row = runner(n_episodes=1, disturbance=d)[0]
            row["iteration"]    = it
            row["cem_mu_spd"]   = mu_spd
            row["cem_mu_pos"]   = mu_pos
            row["cem_sig_spd"]  = sig_spd
            row["cem_sig_pos"]  = sig_pos
            batch.append(row)

        all_results.extend(batch)

        # --- elite selection ---
        # Only update when crashes are found — avoids being misled by proxy
        # metrics (e.g. min_speed) that anchor the distribution to wrong regions
        crashed = [r for r in batch if r["crashed"]]

        if crashed:
            elite_spds = [e["merging_speed"]    for e in crashed]
            elite_poss = [e["merging_position"] for e in crashed]

            new_mu_spd  = float(np.mean(elite_spds))
            new_mu_pos  = float(np.mean(elite_poss))
            new_sig_spd = max(params.speed_min_std, float(np.std(elite_spds)))
            new_sig_pos = max(params.pos_min_std,   float(np.std(elite_poss)))

            # exponential smoothing — prevents large jumps between iterations
            mu_spd  = params.alpha * new_mu_spd  + (1 - params.alpha) * mu_spd
            mu_pos  = params.alpha * new_mu_pos  + (1 - params.alpha) * mu_pos
            sig_spd = params.alpha * new_sig_spd + (1 - params.alpha) * sig_spd
            sig_pos = params.alpha * new_sig_pos + (1 - params.alpha) * sig_pos

        print(
            f"  iter {it+1:2d}/{params.n_iter}"
            f"  crashes={len(crashed)}/{params.n_samples}"
            f"  μ=({mu_spd:.1f}, {mu_pos:.1f})"
            f"  σ=({sig_spd:.1f}, {sig_pos:.1f})"
            f"  {'(updated)' if crashed else '(held)'}",
            flush=True,
        )

    return all_results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for agent in ("idm", "dqn"):
        print(f"\n{'='*50}")
        print(f"  CEM — {agent.upper()}")
        print(f"{'='*50}")
        results = run_cem(agent=agent)
        save_results(results, f"{agent}_cem.csv")

        total    = len(results)
        crashes  = sum(r["crashed"] for r in results)
        print(f"\n  Total crashes: {crashes}/{total} = {crashes/total:.2%}")
