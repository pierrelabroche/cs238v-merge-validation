from evaluate import run_grid_sweep, save_results

print("=== IDM Grid Sweep ===", flush=True)
idm_grid = run_grid_sweep(agent="idm")
save_results(idm_grid, "idm_grid.csv")

print("\n=== DQN Grid Sweep ===", flush=True)
dqn_grid = run_grid_sweep(agent="dqn")
save_results(dqn_grid, "dqn_grid.csv")

print("\nDONE", flush=True)
