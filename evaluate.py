import os
from dataclasses import dataclass
from itertools import product
import gymnasium as gym
import highway_env
from highway_env.vehicle.behavior import IDMVehicle
from stable_baselines3 import DQN
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

# ── Constants ────────────────────────────────────────────────────────────────
V_TARGET = 17   # liveness threshold (m/s) — set at 17 to capture meaningful IDM braking degradation
K        = 5    # consecutive steps below V_TARGET to count as speed failure

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Disturbance space ─────────────────────────────────────────────────────────
# When adding CEM/IS, consider moving this block to disturbances.py

GRID_SPEEDS    = [10.0, 15.0, 20.0, 25.0]   # merging vehicle initial speed (m/s)
GRID_POSITIONS = [70.0, 90.0, 110.0, 130.0]  # position along j-k ramp lane (m)
GRID_TRIALS    = 20                           # episodes per (speed, position) cell

@dataclass
class Disturbance:
    merging_speed: float    = 20.0   # initial speed of merging vehicle (m/s)
    merging_position: float = 110.0  # position along j-k ramp lane (m)
    n_merging: int          = 1      # number of merging vehicles on ramp

def apply_disturbance(env, d: Disturbance) -> None:
    """
    Patch merging vehicle state after env.reset().

    The merge-v0 environment hard-codes one merging vehicle at position=110,
    speed=20. This function repositions and re-speeds it (and optionally spawns
    additional ones) without touching the environment's internals permanently.
    """
    road    = env.unwrapped.road
    ego     = env.unwrapped.vehicle
    jk_lane = road.network.get_lane(("j", "k", 0))

    ramp_ids = {("j", "k", 0), ("k", "b", 0), ("b", "c", 2)}
    merging_vs = [
        v for v in road.vehicles
        if tuple(v.lane_index) in ramp_ids and v is not ego
    ]

    if not merging_vs:
        return

    # --- primary merging vehicle ---
    mv         = merging_vs[0]
    mv.speed   = d.merging_speed
    mv.position = jk_lane.position(d.merging_position, 0).copy()
    mv.heading  = jk_lane.heading_at(d.merging_position)

    # --- additional merging vehicles (20 m spacing behind) ---
    VehicleClass = type(mv)
    for i in range(1, d.n_merging):
        pos = d.merging_position - i * 20.0
        if pos < 0:
            break
        extra = VehicleClass(
            road,
            jk_lane.position(pos, 0),
            heading=jk_lane.heading_at(pos),
            speed=d.merging_speed,
        )
        extra.target_speed = 30.0
        road.vehicles.append(extra)

# ── Helpers ──────────────────────────────────────────────────────────────────
def save_results(results: list[dict], filename: str) -> None:
    filepath = os.path.join(RESULTS_DIR, filename)
    pd.DataFrame(results).to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

def load_results(filename: str) -> list[dict]:
    filepath = os.path.join(RESULTS_DIR, filename)
    return pd.read_csv(filepath).to_dict("records")

def _speed_stats(speeds: list[float], crashed: bool) -> dict:
    """Return the three speed-failure metrics for one episode."""
    min_spd  = min(speeds)
    avg_spd  = float(np.mean(speeds))

    # consecutive steps below V_TARGET
    max_consec = cur_consec = 0
    for s in speeds:
        if s < V_TARGET:
            cur_consec += 1
            max_consec  = max(max_consec, cur_consec)
        else:
            cur_consec  = 0

    return {
        "crashed"              : crashed,
        "final_speed"          : speeds[-1] if speeds else float("nan"),
        "min_speed"            : min_spd,
        "avg_speed"            : avg_spd,
        "max_consecutive_below": max_consec,
        # option 1 – min ever below threshold
        "speed_fail_opt1"      : min_spd  < V_TARGET,
        # option 2a – average below threshold
        "speed_fail_opt2a"     : avg_spd  < V_TARGET,
        # option 2b – k+ consecutive steps below threshold
        "speed_fail_opt2b"     : max_consec > K,
    }

# ── IDM agent ────────────────────────────────────────────────────────────────
def run_episodes(
    n_episodes: int = 200,
    render: bool = False,
    disturbance: Disturbance | None = None,
) -> list[dict]:
    env = gym.make("merge-v0", render_mode="human" if render else None)

    results = []
    for episode in range(n_episodes):
        obs, info  = env.reset()
        terminated = truncated = False
        speeds: list[float] = []

        # replace controlled ego with a fully autonomous IDM+MOBIL vehicle
        old_ego = env.unwrapped.vehicle  # type: ignore
        ego = IDMVehicle(
            env.unwrapped.road,          # type: ignore
            old_ego.position,
            heading=old_ego.heading,
            speed=old_ego.speed,
        )
        env.unwrapped.road.vehicles[0] = ego  # type: ignore
        env.unwrapped.vehicle          = ego  # type: ignore

        if disturbance is not None:
            apply_disturbance(env, disturbance)

        while not (terminated or truncated):
            action = ego.act()
            obs, reward, terminated, truncated, info = env.step(action)
            speeds.append(float(ego.speed))

        row = {"episode": episode, **_speed_stats(speeds, bool(ego.crashed))}
        if disturbance is not None:
            row["merging_speed"]    = disturbance.merging_speed
            row["merging_position"] = disturbance.merging_position
        results.append(row)

    env.close()
    return results

# ── DQN agent ────────────────────────────────────────────────────────────────
def train_dqn(timesteps: int = 50_000) -> DQN:
    env   = gym.make("merge-v0")
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save("dqn_merge")
    env.close()
    print("DQN training complete — model saved to dqn_merge.zip")
    return model

def evaluate_dqn_runs(n_runs=5, timesteps=100_000, n_episodes=200):
    """Train DQN multiple times and average results to account for variance."""
    all_results = []
    
    for run in range(n_runs):
        print(f"\nDQN Run {run+1}/{n_runs}...")
        train_dqn(timesteps=timesteps)
        results = run_episodes_dqn(n_episodes=n_episodes)
        save_results(results, f"dqn_run_{run+1}.csv")
        
        crashed = sum(r["crashed"] for r in results)
        print(f"Run {run+1} crash rate: {crashed}/{n_episodes} = {crashed/n_episodes:.1%}")
        all_results.append(results)
    
    # aggregate across runs
    crash_rates = [sum(r["crashed"] for r in run)/n_episodes for run in all_results]
    avg_speeds  = [np.mean([r["avg_speed"] for r in run]) for run in all_results]
    
    print(f"\n=== DQN Across {n_runs} Runs ===")
    print(f"Crash rate: {np.mean(crash_rates):.1%} ± {np.std(crash_rates):.1%}")
    print(f"Avg speed:  {np.mean(avg_speeds):.1f} ± {np.std(avg_speeds):.1f} m/s")
    
    return all_results

def run_episodes_dqn(
    n_episodes: int = 200,
    render: bool = False,
    disturbance: Disturbance | None = None,
) -> list[dict]:
    env   = gym.make("merge-v0", render_mode="human" if render else None)
    model = DQN.load("dqn_merge")

    results = []
    for episode in range(n_episodes):
        obs, info  = env.reset()
        terminated = truncated = False
        speeds: list[float] = []

        if disturbance is not None:
            apply_disturbance(env, disturbance)

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            speeds.append(float(env.unwrapped.vehicle.speed))  # type: ignore

        crashed = bool(env.unwrapped.vehicle.crashed)  # type: ignore
        row = {"episode": episode, **_speed_stats(speeds, crashed)}
        if disturbance is not None:
            row["merging_speed"]    = disturbance.merging_speed
            row["merging_position"] = disturbance.merging_position
        results.append(row)

    env.close()
    return results

# ── Importance Sampling ───────────────────────────────────────────────────────
# Nominal distribution p(d): Uniform over [SPEED_LO, SPEED_HI] x [POS_LO, POS_HI]
# Proposal distribution q(d): product of truncated normals biased toward failure regions
# IS weight: w(d) = p(d) / q(d)
# Failure probability estimate: P(fail) ≈ (1/N) Σ w_i * f(d_i),  d_i ~ q

SPEED_LO, SPEED_HI = 10.0, 25.0   # disturbance support: merging speed (m/s)
POS_LO,   POS_HI   = 70.0, 130.0  # disturbance support: merging position (m)

@dataclass
class ISProposal:
    speed_mean: float
    speed_std:  float
    pos_mean:   float
    pos_std:    float

# Proposals informed by grid sweep results:
#   DQN fails at low position (70-90), any speed → bias proposal toward low position
#   IDM fails at high position (130), low speed  → bias proposal toward high pos + low speed
IS_PROPOSALS = {
    "dqn": ISProposal(speed_mean=15.0, speed_std=5.0,  pos_mean=80.0,  pos_std=15.0),
    "idm": ISProposal(speed_mean=12.0, speed_std=4.0,  pos_mean=125.0, pos_std=8.0),
}

def _trunc(mean: float, std: float, lo: float, hi: float) -> truncnorm:
    a, b = (lo - mean) / std, (hi - mean) / std
    return truncnorm(a, b, loc=mean, scale=std)

def sample_proposal(proposal: ISProposal) -> tuple[Disturbance, float]:
    """
    Sample d ~ q(d) and return (disturbance, importance_weight).

    Importance weight w = p(d) / q(d) where p is the uniform nominal distribution.
    Multiply w by a failure indicator to get an unbiased failure probability estimate.
    """
    spd_rv = _trunc(proposal.speed_mean, proposal.speed_std, SPEED_LO, SPEED_HI)
    pos_rv = _trunc(proposal.pos_mean,   proposal.pos_std,   POS_LO,   POS_HI)

    spd = float(spd_rv.rvs())
    pos = float(pos_rv.rvs())

    p = 1.0 / ((SPEED_HI - SPEED_LO) * (POS_HI - POS_LO))  # uniform density
    q = float(spd_rv.pdf(spd)) * float(pos_rv.pdf(pos))      # proposal density

    return Disturbance(merging_speed=spd, merging_position=pos), p / q

def sample_uniform() -> Disturbance:
    """Sample disturbance uniformly from the nominal distribution p(d)."""
    return Disturbance(
        merging_speed=float(np.random.uniform(SPEED_LO, SPEED_HI)),
        merging_position=float(np.random.uniform(POS_LO, POS_HI)),
    )

def is_failure_prob(results: list[dict], failure_key: str = "crashed") -> float:
    """IS estimate of failure probability: (1/N) Σ w_i * f_i."""
    return float(np.mean([r["is_weight"] * r[failure_key] for r in results]))

def run_is_episodes(
    agent: str = "dqn",
    n_episodes: int = 200,
    proposal: ISProposal | None = None,
) -> list[dict]:
    """
    Run IS episodes: each disturbance is sampled from q, episode is rolled out,
    result includes 'is_weight' = p(d)/q(d) for unbiased failure prob estimation.
    """
    if proposal is None:
        proposal = IS_PROPOSALS[agent]
    runner = run_episodes if agent == "idm" else run_episodes_dqn

    all_results = []
    for episode in range(n_episodes):
        d, weight = sample_proposal(proposal)
        row = runner(n_episodes=1, disturbance=d)[0]
        row["episode"]   = episode
        row["is_weight"] = weight
        all_results.append(row)

        if (episode + 1) % 50 == 0:
            print(f"  ep {episode+1}/{n_episodes}  IS P(crash) = {is_failure_prob(all_results):.4f}", flush=True)

    return all_results

def run_mc_disturbance_episodes(
    agent: str = "dqn",
    n_episodes: int = 200,
) -> list[dict]:
    """
    MC baseline with uniform disturbance sampling — apples-to-apples comparison
    against IS. is_weight=1.0 for all episodes (p=q=uniform).
    """
    runner = run_episodes if agent == "idm" else run_episodes_dqn

    all_results = []
    for episode in range(n_episodes):
        d = sample_uniform()
        row = runner(n_episodes=1, disturbance=d)[0]
        row["episode"]   = episode
        row["is_weight"] = 1.0
        all_results.append(row)

        if (episode + 1) % 50 == 0:
            crashes = sum(r["crashed"] for r in all_results)
            print(f"  ep {episode+1}/{n_episodes}  MC crash rate = {crashes/(episode+1):.4f}", flush=True)

    return all_results

# ── Grid sweep ───────────────────────────────────────────────────────────────
def run_grid_sweep(
    agent: str = "idm",
    speeds: list[float] = GRID_SPEEDS,
    positions: list[float] = GRID_POSITIONS,
    n_trials: int = GRID_TRIALS,
) -> list[dict]:
    """
    Run a full disturbance grid sweep for one agent.

    For each (merging_speed, merging_position) cell, runs n_trials episodes and
    records per-episode outcomes.  Results include disturbance params so the
    DataFrame can be pivoted into a heatmap.

    agent: "idm" or "dqn"
    """
    assert agent in ("idm", "dqn"), "agent must be 'idm' or 'dqn'"
    runner = run_episodes if agent == "idm" else run_episodes_dqn

    all_results = []
    cells = list(product(speeds, positions))
    for i, (spd, pos) in enumerate(cells):
        d = Disturbance(merging_speed=spd, merging_position=pos)
        print(f"  [{i+1}/{len(cells)}] speed={spd:.0f} m/s  position={pos:.0f} m ...", flush=True)
        rows = runner(n_episodes=n_trials, disturbance=d)
        all_results.extend(rows)

    return all_results

# ── Reporting ────────────────────────────────────────────────────────────────
def print_summary(results: list[dict], label: str) -> None:
    n = len(results)
    crashes = sum(r["crashed"]           for r in results)
    o1      = sum(r["speed_fail_opt1"]   for r in results)
    o2a     = sum(r["speed_fail_opt2a"]  for r in results)
    o2b     = sum(r["speed_fail_opt2b"]  for r in results)

    print(f"\n{'='*45}")
    print(f"  {label} Results  ({n} episodes)")
    print(f"{'='*45}")
    print(f"  Crash rate          : {crashes}/{n} = {crashes/n:.1%}")
    print(f"  Avg final speed     : {np.mean([r['final_speed'] for r in results]):.1f} m/s")
    print(f"  Avg min speed       : {np.mean([r['min_speed']   for r in results]):.1f} m/s")
    print(f"  Avg avg speed       : {np.mean([r['avg_speed']   for r in results]):.1f} m/s")
    print(f"  --- Speed Failures (V_TARGET={V_TARGET} m/s, K={K} steps) ---")
    print(f"  Opt 1  | min < {V_TARGET}              : {o1}/{n} = {o1/n:.1%}")
    print(f"  Opt 2a | avg < {V_TARGET}              : {o2a}/{n} = {o2a/n:.1%}")
    print(f"  Opt 2b | >{K} consec steps < {V_TARGET} : {o2b}/{n} = {o2b/n:.1%}")

def print_comparison(idm: list[dict], dqn: list[dict]) -> None:
    n_idm = len(idm)
    n_dqn = len(dqn)
    idm_crash = sum(r["crashed"] for r in idm)
    dqn_crash = sum(r["crashed"] for r in dqn)
    idm_avg   = np.mean([r["avg_speed"] for r in idm])
    dqn_avg   = np.mean([r["avg_speed"] for r in dqn])

    print(f"\n{'='*45}")
    print("  Comparison (DQN vs IDM)")
    print(f"{'='*45}")
    print(f"  IDM crash rate      : {idm_crash}/{n_idm} = {idm_crash/n_idm:.1%}")
    print(f"  DQN crash rate      : {dqn_crash}/{n_dqn} = {dqn_crash/n_dqn:.1%}")
    print(f"  Crash rate delta    : {(dqn_crash/n_dqn) - (idm_crash/n_idm):+.1%}")
    print(f"  Avg speed delta     : {dqn_avg - idm_avg:+.1f} m/s")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N = 200

    # IDM baseline
    print("Running IDM episodes...")
    idm_results = run_episodes(n_episodes=N)
    print_summary(idm_results, "IDM+MOBIL")
    save_results(idm_results, "idm_baseline.csv")

    # DQN with multiple runs
    print("\nStarting DQN multi-run evaluation...")
    dqn_all_runs = evaluate_dqn_runs(n_runs=5, timesteps=100_000, n_episodes=N)

    # compare IDM against average DQN run
    dqn_flat = [r for run in dqn_all_runs for r in run]
    print_comparison(idm_results, dqn_flat)

    # Grid sweep (4 speeds × 4 positions × 20 trials = 320 episodes per agent)
    print("\nRunning IDM grid sweep...")
    idm_grid = run_grid_sweep(agent="idm")
    save_results(idm_grid, "idm_grid.csv")

    print("\nRunning DQN grid sweep...")
    dqn_grid = run_grid_sweep(agent="dqn")
    save_results(dqn_grid, "dqn_grid.csv")