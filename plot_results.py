"""
Plotting script for CS238v final project results.

Figures generated:
  1. Sample efficiency: cumulative crashes vs episodes (MC vs IS vs CEM)
  2. Crash rate heatmaps: IDM and DQN grid sweep
  3. Bar chart: baseline crash rate + speed failures (IDM vs DQN)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def cumulative_crashes(df: pd.DataFrame) -> np.ndarray:
    return df["crashed"].astype(int).cumsum().values


# ── Figure 1: Sample efficiency ───────────────────────────────────────────────

def plot_sample_efficiency(agent: str = "dqn") -> None:
    mc  = pd.read_csv(os.path.join(RESULTS_DIR, f"{agent}_mc_disturbance.csv"))
    is_ = pd.read_csv(os.path.join(RESULTS_DIR, f"{agent}_is.csv"))
    cem = pd.read_csv(os.path.join(RESULTS_DIR, f"{agent}_cem.csv"))

    mc_crashes  = cumulative_crashes(mc)
    is_crashes  = cumulative_crashes(is_)
    cem_crashes = cumulative_crashes(cem)

    mc_eps  = np.arange(1, len(mc_crashes)  + 1)
    is_eps  = np.arange(1, len(is_crashes)  + 1)
    cem_eps = np.arange(1, len(cem_crashes) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(mc_eps,  mc_crashes,  label="Monte Carlo (uniform)",   color="steelblue",  linewidth=2)
    ax.plot(is_eps,  is_crashes,  label="Importance Sampling",      color="darkorange", linewidth=2, linestyle="--")
    ax.plot(cem_eps, cem_crashes, label="CEM (adaptive)",           color="seagreen",   linewidth=2)

    # mark CEM iteration boundaries
    n_samples = len(cem) // 10  # infer n_samples per iter
    for i in range(1, 10):
        ax.axvline(i * n_samples, color="seagreen", linewidth=0.5, linestyle=":", alpha=0.5)

    ax.set_xlabel("Episodes sampled", fontsize=12)
    ax.set_ylabel("Cumulative crashes found", fontsize=12)
    ax.set_title(f"Sample efficiency: {agent.upper()} agent", fontsize=13)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(FIGURES_DIR, f"sample_efficiency_{agent}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ── Figure 2: Grid sweep heatmaps ─────────────────────────────────────────────

def plot_heatmaps() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, agent in zip(axes, ("idm", "dqn")):
        df = pd.read_csv(os.path.join(RESULTS_DIR, f"{agent}_grid.csv"))
        pivot = (
            df.groupby(["merging_speed", "merging_position"])["crashed"]
            .mean()
            .unstack("merging_position")
        )

        im = ax.imshow(
            pivot.values * 100,
            aspect="auto",
            origin="lower",
            cmap="Reds",
            vmin=0,
            vmax=25,
        )

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{int(p)}" for p in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{int(s)}" for s in pivot.index])
        ax.set_xlabel("Merging position (m)", fontsize=11)
        ax.set_ylabel("Merging speed (m/s)", fontsize=11)
        ax.set_title(f"{agent.upper()} crash rate (%)", fontsize=12)

        # annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j] * 100
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=9, color="black" if val < 15 else "white")

        fig.colorbar(im, ax=ax, label="Crash rate (%)")

    fig.suptitle("Crash rate across disturbance grid (4×4, 20 trials/cell)", fontsize=13)
    path = os.path.join(FIGURES_DIR, "heatmaps.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ── Figure 3: Baseline bar chart ──────────────────────────────────────────────

V_TARGET = 17  # must match evaluate.py

def plot_baseline_bars() -> None:
    idm = pd.read_csv(os.path.join(RESULTS_DIR, "idm_baseline.csv"))
    dqn_runs = [
        pd.read_csv(os.path.join(RESULTS_DIR, f"dqn_run_{i}.csv"))
        for i in range(1, 6)
    ]

    # recompute opt1/opt2a from raw speed columns at current V_TARGET
    # (avoids dependence on pre-baked boolean columns computed at old threshold)
    def opt1(df):  return (df["min_speed"] < V_TARGET).mean() * 100
    def opt2a(df): return (df["avg_speed"] < V_TARGET).mean() * 100
    def crash(df): return df["crashed"].mean() * 100

    labels = ["Crash rate", f"Speed fail\n(min<{V_TARGET})", f"Speed fail\n(avg<{V_TARGET})"]

    idm_vals = [crash(idm), opt1(idm), opt2a(idm)]

    dqn_run_vals = [[crash(r), opt1(r), opt2a(r)] for r in dqn_runs]
    dqn_means = np.mean(dqn_run_vals, axis=0)
    dqn_stds  = np.std(dqn_run_vals,  axis=0)

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_idm = ax.bar(x - w/2, idm_vals,  w, label="IDM+MOBIL", color="steelblue")
    ax.bar(x + w/2, dqn_means, w, label="DQN", color="darkorange",
           yerr=dqn_stds, capsize=4)

    # annotate zero IDM crash bar explicitly
    ax.text(x[0] - w/2, 0.3, "0%", ha="center", va="bottom", fontsize=9, color="steelblue")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Rate (%)", fontsize=11)
    ax.set_title(f"Failure rates: IDM+MOBIL vs DQN (V_TARGET={V_TARGET} m/s)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(FIGURES_DIR, "baseline_bars.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...")
    plot_sample_efficiency(agent="dqn")
    plot_heatmaps()
    plot_baseline_bars()
    print("Done. Figures saved to figures/")
