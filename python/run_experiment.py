"""Runs the adaptive (property-estimate-conditioned) policy and the
property-blind baseline over the identical set of out-of-distribution
(mass, mu) trials and noise draws, and reports success rate plus the
sliding-window estimator's own per-tick solve-time distribution (the basis
for the "120 Hz in the control loop" claim).
"""

import argparse
import json
import time

import numpy as np

import config
from simulate import make_model_data, run_trial
from policy import AdaptivePolicy, BlindPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=config.N_OOD_TRIALS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    control_dt = 1.0 / config.CONTROL_HZ
    model, data = make_model_data()

    trial_rng = np.random.default_rng(args.seed)
    masses = trial_rng.uniform(*config.OOD_MASS_RANGE, size=args.n_trials)
    mus = trial_rng.uniform(*config.OOD_MU_RANGE, size=args.n_trials)
    noise_seeds = trial_rng.integers(0, 2**31 - 1, size=args.n_trials)

    adaptive_successes = 0
    blind_successes = 0
    solve_times = []
    final_mass_err = []
    final_mu_err = []

    t_start = time.perf_counter()
    for i in range(args.n_trials):
        mass, mu, seed = masses[i], mus[i], int(noise_seeds[i])

        adaptive_policy = AdaptivePolicy(control_dt)
        rng_a = np.random.default_rng(seed)
        result_a = run_trial(model, data, mass, mu, adaptive_policy, rng_a, solve_time_log=solve_times)
        adaptive_successes += result_a["success"]
        if adaptive_policy.last_estimate is not None:
            final_mass_err.append(abs(adaptive_policy.last_estimate[0] - mass))
            final_mu_err.append(abs(adaptive_policy.last_estimate[1] - mu))

        blind_policy = BlindPolicy()
        rng_b = np.random.default_rng(seed)
        result_b = run_trial(model, data, mass, mu, blind_policy, rng_b)
        blind_successes += result_b["success"]

    wall_s = time.perf_counter() - t_start
    solve_times = np.array(solve_times)

    report = {
        "n_trials": args.n_trials,
        "adaptive_success_rate": adaptive_successes / args.n_trials,
        "blind_success_rate": blind_successes / args.n_trials,
        "adaptive_successes": adaptive_successes,
        "blind_successes": blind_successes,
        "estimator_solve_time_p50_ms": float(np.percentile(solve_times, 50) * 1000),
        "estimator_solve_time_p95_ms": float(np.percentile(solve_times, 95) * 1000),
        "estimator_solve_time_p99_ms": float(np.percentile(solve_times, 99) * 1000),
        "control_budget_ms": 1000.0 / config.CONTROL_HZ,
        "mean_abs_mass_error_kg": float(np.mean(final_mass_err)),
        "mean_abs_mu_error": float(np.mean(final_mu_err)),
        "wall_clock_s": wall_s,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
