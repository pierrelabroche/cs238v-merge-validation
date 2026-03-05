import os
import gymnasium as gym
import highway_env
from highway_env.vehicle.behavior import IDMVehicle
from stable_baselines3 import DQN
import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
V_TARGET = 15   # liveness threshold (m/s)
K        = 5    # consecutive steps below V_TARGET to count as speed failure

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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
def run_episodes(n_episodes: int = 200, render: bool = False) -> list[dict]:
    env = gym.make("merge-v0", render_mode="human" if render else None)

    results = []
    for episode in range(n_episodes):
        obs, info    = env.reset()
        terminated   = truncated = False
        speeds: list[float] = []

        # replace controlled ego with a fully autonomous IDM+MOBIL vehicle
        ego = env.unwrapped.vehicle  # type: ignore
        old_ego = ego
        ego = IDMVehicle(
            env.unwrapped.road,          # type: ignore
            old_ego.position,
            heading=old_ego.heading,
            speed=old_ego.speed,
        )
        env.unwrapped.road.vehicles[0] = ego  # type: ignore
        env.unwrapped.vehicle          = ego  # type: ignore

        while not (terminated or truncated):
            action = ego.act()
            obs, reward, terminated, truncated, info = env.step(action)
            speeds.append(float(ego.speed))

        results.append({"episode": episode, **_speed_stats(speeds, bool(ego.crashed))})

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

def run_episodes_dqn(n_episodes: int = 200, render: bool = False) -> list[dict]:
    env   = gym.make("merge-v0", render_mode="human" if render else None)
    model = DQN.load("dqn_merge")

    results = []
    for episode in range(n_episodes):
        obs, info  = env.reset()
        terminated = truncated = False
        speeds: list[float] = []

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            speeds.append(float(env.unwrapped.vehicle.speed))  # type: ignore

        crashed = bool(env.unwrapped.vehicle.crashed)  # type: ignore
        results.append({"episode": episode, **_speed_stats(speeds, crashed)})

    env.close()
    return results

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