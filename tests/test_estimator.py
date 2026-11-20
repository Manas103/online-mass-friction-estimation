import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import config  # noqa: E402
from property_estimator import (  # noqa: E402
    SlidingWindowPropertyEstimator,
    _softplus,
    predicted_displacement_magnitude,
)
from policy import AdaptivePolicy, BlindPolicy, grip_force_from_properties  # noqa: E402
from simulate import make_model_data, run_trial  # noqa: E402


def test_softplus_is_smooth_and_nonnegative():
    x = np.linspace(-50, 50, 2001)
    y = _softplus(x)
    assert np.all(np.isfinite(y))
    assert np.all(y >= -1e-9)
    # within the range the estimator actually operates in (n_star and
    # applied forces are both bounded to the actuator's range), the
    # derivative never rounds to exactly zero; a hard max(0, x) clip's
    # derivative is exactly zero on the whole "held" side, which is what
    # left the solver stuck at its initial guess (see README "Findings")
    x_operating = np.linspace(-15, 15, 2001)
    y_operating = _softplus(x_operating)
    dy = np.diff(y_operating) / np.diff(x_operating)
    assert np.all(dy > 0)


def test_predicted_displacement_monotonic_in_undershoot():
    n_meas = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    pred = predicted_displacement_magnitude(3.0, n_meas)
    assert np.all(np.diff(pred) > 0), "predicted displacement must grow as applied force drops further below n_star"


def synthetic_window(n_star_true, scale=0.003, slope=5.0, window=config.WINDOW_TICKS, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n_meas = np.concatenate([
        np.linspace(n_star_true + 2.0, n_star_true - 2.0, window // 2),
        np.full(window - window // 2, n_star_true - 2.0),
    ])
    disp = scale * np.log1p(np.exp(np.clip(slope * (n_star_true - n_meas), -30, 30)))
    rel_z = -disp + rng.normal(0.0, noise, size=window)
    baseline = 0.0
    return rel_z, n_meas, baseline


def brute_force_oracle(rel_z, n_meas, baseline, grid=None):
    """Deliberately slow, obviously-correct grid-search reference, diffed
    against the sliding-window least-squares solver below."""
    if grid is None:
        grid = np.linspace(config.GRIP_CTRL_MIN, config.GRIP_CTRL_MAX, 4000)
    disp_mag = np.abs(np.asarray(rel_z) - baseline)
    best_cost = np.inf
    best_n_star = grid[0]
    for n_star in grid:
        pred = predicted_displacement_magnitude(n_star, n_meas)
        cost = np.sum((disp_mag - pred) ** 2)
        if cost < best_cost:
            best_cost = cost
            best_n_star = n_star
    return best_n_star


def test_estimator_matches_reference_oracle_on_synthetic_window():
    rel_z, n_meas, baseline = synthetic_window(3.4, seed=1)
    est = SlidingWindowPropertyEstimator(control_dt=1.0 / config.CONTROL_HZ)
    for z, n in zip(rel_z, n_meas):
        est.update(z + baseline, 0.0, n, n)
    oracle_n_star = brute_force_oracle(rel_z, n_meas, baseline)
    assert abs(est.n_star - oracle_n_star) < 0.05


def test_estimator_recovers_known_critical_force_noiseless():
    true_n_star = 4.2
    rel_z, n_meas, baseline = synthetic_window(true_n_star, seed=2)
    est = SlidingWindowPropertyEstimator(control_dt=1.0 / config.CONTROL_HZ)
    for z, n in zip(rel_z, n_meas):
        est.update(z + baseline, 0.0, n, n)
    assert abs(est.n_star - true_n_star) < 0.1


def test_multistart_escapes_a_bad_warm_start():
    """Regression test: a single warm start from a value far outside the
    window's own force range has ~zero gradient and gets permanently stuck
    (see README "Findings"). Seed the estimator at that corner and confirm
    it still finds the right answer."""
    true_n_star = 3.4
    rel_z, n_meas, baseline = synthetic_window(true_n_star, seed=3)
    est = SlidingWindowPropertyEstimator(control_dt=1.0 / config.CONTROL_HZ)
    est.n_star = 0.1  # the exact corner the single-start version got stuck at
    for z, n in zip(rel_z, n_meas):
        est.update(z + baseline, 0.0, n, n)
    assert abs(est.n_star - true_n_star) < 0.3


def test_held_window_does_not_diverge():
    """A window where every applied force already exceeds the (unknown)
    critical force should settle near a safe, bounded estimate, not run to
    the parameter's lower bound."""
    rel_z, n_meas, baseline = synthetic_window(1.0, seed=4)  # n_star well below all applied forces
    est = SlidingWindowPropertyEstimator(control_dt=1.0 / config.CONTROL_HZ)
    for z, n in zip(rel_z, n_meas):
        est.update(z + baseline, 0.0, n, n)
    assert config.GRIP_CTRL_MIN <= est.n_star <= config.GRIP_CTRL_MAX


def test_grip_force_law_scales_with_safety_factor():
    base = grip_force_from_properties(0.2, 0.5)
    import importlib
    import policy as policy_module
    old = config.SAFETY_FACTOR
    try:
        config.SAFETY_FACTOR = old * 2
        importlib.reload(policy_module)
        doubled = policy_module.grip_force_from_properties(0.2, 0.5)
    finally:
        config.SAFETY_FACTOR = old
        importlib.reload(policy_module)
    assert doubled == pytest.approx(base * 2)


def test_blind_policy_never_changes_its_command():
    pol = BlindPolicy()
    cmds = [pol(t, 0.0, 0.0, 0.0) for t in range(50)]
    assert len(set(cmds)) == 1


@pytest.mark.slow
def test_lower_friction_slips_more_under_identical_fixed_force():
    """Invariant check against the simulator itself: holding grip force and
    mass fixed, a lower friction coefficient must not slip less."""
    model, data = make_model_data()

    def fixed(force):
        def pol(tick, box_z_vision, n_meas, palm_z_true):
            return force

        pol.last_estimate = None
        return pol

    force = 2.0
    mass = 0.3
    r_low_mu = run_trial(model, data, mass, 0.3, fixed(force), np.random.default_rng(0))
    r_high_mu = run_trial(model, data, mass, 0.9, fixed(force), np.random.default_rng(0))
    sag_low = r_low_mu["final_palm_z"] - r_low_mu["final_box_z"]
    sag_high = r_high_mu["final_palm_z"] - r_high_mu["final_box_z"]
    assert sag_low >= sag_high - 1e-6


@pytest.mark.slow
def test_adaptive_bootstrap_secures_object_before_any_fit():
    """During the bootstrap ticks, the adaptive policy must command the
    actuator's safe maximum regardless of the (untrained) estimate."""
    pol = AdaptivePolicy(control_dt=1.0 / config.CONTROL_HZ)
    cmd = pol(0, 0.0, 0.0, 0.0)
    assert cmd == config.GRIP_CTRL_MAX
