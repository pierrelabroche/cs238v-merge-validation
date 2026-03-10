from evaluate import (
    run_is_episodes, run_mc_disturbance_episodes,
    is_failure_prob, save_results,
)

N = 200  # episodes per method per agent

for agent in ("idm", "dqn"):
    print(f"\n{'='*45}")
    print(f"  {agent.upper()} — MC (uniform disturbance)")
    print(f"{'='*45}")
    mc = run_mc_disturbance_episodes(agent=agent, n_episodes=N)
    save_results(mc, f"{agent}_mc_disturbance.csv")
    print(f"MC  P(crash): {sum(r['crashed'] for r in mc)/N:.4f}")

    print(f"\n{'='*45}")
    print(f"  {agent.upper()} — Importance Sampling")
    print(f"{'='*45}")
    is_ = run_is_episodes(agent=agent, n_episodes=N)
    save_results(is_, f"{agent}_is.csv")
    print(f"IS  P(crash): {is_failure_prob(is_):.4f}")

print("\nDONE")
