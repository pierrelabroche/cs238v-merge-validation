from evaluate import run_mc_disturbance_episodes, save_results

results = run_mc_disturbance_episodes(agent="idm", n_episodes=1000)
save_results(results, "idm_mc_disturbance_1000.csv")
print(f"Crashes: {sum(r['crashed'] for r in results)}/1000")
