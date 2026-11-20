"""Sliding-window factor graph that fuses a noisy vision measurement of box
height and a noisy fingertip force reading to estimate the critical grip
force below which an object slips, online, re-solved every control tick.

First attempt (kept here only in "Findings," not in the code path): fit the
Coulomb slip acceleration a_pred = g - (N_FINGERS*mu/mass)*N directly, by
forward-integrating it from an unknown window-start velocity and matching
the noisy vision trace. Measured sliding-window costs showed the data
preferred a constant-velocity (no-slip) explanation over the true physics
at the true parameters, because MuJoCo's contact solver does not decelerate
a slipping grasp the way the idealized point-mass Coulomb model predicts
over a 125ms window; the acceleration-matching fit was not reliable at this
window length and noise level on this machine (see README "Findings").

What is actually fit instead is model-free and robust to that mismatch: a
single critical-force parameter N*, fit against the *magnitude* of
vision-observed displacement away from the grip's position when the probe
ramp began. Displacement magnitude is monotonic once slip starts (gravity
pulls one way), so the fit only needs a monotonic transition through N*,
not the exact post-slip trajectory shape the acceleration model got wrong. The fingertip force sensor's reading, not just the
commanded force, is what is actually fit against (the two usually agree
closely; where they do not, the measured force is the one trusted).
"""

import numpy as np
from scipy.optimize import least_squares

import config


def _softplus(x, beta=1.0):
    bx = beta * x
    return np.where(bx > 30, x, np.log1p(np.exp(np.clip(bx, -30, 30))) / beta)


# Empirically calibrated (see docs/calibration_output.txt): the displacement
# magnitude a fully-slipping grasp accumulates over one SETTLE_TICKS window
# at this control rate, on this machine. Not a physical constant; a fit
# nuisance scale shared across trials.
DISPLACEMENT_SCALE = 0.003
SLOPE = 5.0


def predicted_displacement_magnitude(n_star, n_meas_history):
    return DISPLACEMENT_SCALE * _softplus(SLOPE * (n_star - np.asarray(n_meas_history)))


def residual_vector(params, rel_z_history, n_meas_history, baseline):
    n_star = params[0]
    disp_mag = np.abs(np.asarray(rel_z_history) - baseline)
    pred = predicted_displacement_magnitude(n_star, n_meas_history)
    return disp_mag - pred


N_STAR_BOUNDS = (0.1, config.GRIP_CTRL_MAX)


class SlidingWindowPropertyEstimator:
    """Re-solved every control tick over the most recent WINDOW_TICKS
    samples, against displacement relative to a fixed baseline (the
    relative height recorded the first time `update` is called, i.e. when
    the probe ramp begins). Exposes mass/mu for reporting: mass is pinned
    at its training-distribution prior (mass and mu are not independently
    identifiable from this sensor set, only their ratio is; see README
    "Findings"), and mu is backed out from N* and that fixed mass."""

    def __init__(self, control_dt, window=config.WINDOW_TICKS, n_star_prior=None):
        self.control_dt = control_dt
        self.window = window
        self.rel_z = []
        self.n_meas = []
        self.baseline = None
        self.n_star = n_star_prior if n_star_prior is not None else (
            config.MASS_PRIOR * config.GRAVITY / (config.N_FINGERS * config.MU_PRIOR)
        )

    @property
    def estimate(self):
        mass_hat = config.MASS_PRIOR
        mu_hat = mass_hat * config.GRAVITY / (config.N_FINGERS * self.n_star)
        return np.array([mass_hat, mu_hat])

    def update(self, box_z_vision, palm_z_true, n_applied, n_meas):
        rel_z = box_z_vision - palm_z_true
        if self.baseline is None:
            self.baseline = rel_z
        self.rel_z.append(rel_z)
        self.n_meas.append(n_meas)
        if len(self.rel_z) > self.window:
            self.rel_z.pop(0)
            self.n_meas.pop(0)

        if len(self.rel_z) < 5:
            return self.estimate

        # Multi-start: the softplus model has ~zero gradient far from the
        # window's own force range, so warm-starting from a single, possibly
        # already-cornered previous estimate could get permanently stuck at
        # a bound (see README "Findings"). Restarting from a few fixed
        # points spread across the actuator's range every tick is cheap
        # (each solve is sub-millisecond) and never carries that bad state
        # forward.
        best_result = None
        for x0 in (self.n_star, 1.5, 6.0):
            result = least_squares(
                residual_vector,
                [x0],
                args=(self.rel_z, self.n_meas, self.baseline),
                bounds=([N_STAR_BOUNDS[0]], [N_STAR_BOUNDS[1]]),
                method="trf",
                max_nfev=50,
            )
            if best_result is None or result.cost < best_result.cost:
                best_result = result
        self.n_star = best_result.x[0]
        return self.estimate
