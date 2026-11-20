"""Two grip-force controllers sharing the identical physics-based control
law; the only difference is whether the (mass, mu) it reads is the live
online estimate or a fixed, property-blind prior.
"""

import config
from property_estimator import SlidingWindowPropertyEstimator


def grip_force_from_properties(m_hat, mu_hat):
    n_cmd = config.SAFETY_FACTOR * m_hat * config.GRAVITY / (config.N_FINGERS * mu_hat)
    return n_cmd


class AdaptivePolicy:
    """Conditions the grip force on the sliding-window factor graph's live
    estimate, in three phases:

    1. Bootstrap (BOOTSTRAP_TICKS): grip at the actuator's safe maximum.
       An object that slips past the fingers' contact height before the
       estimator has any data is unrecoverable for the rest of the trial
       (see README "Findings"), so this phase exists purely to secure the
       object, not to calibrate anything.
    2. Probe (PROBE_TICKS): ramp the grip force down from that maximum
       toward PROBE_FLOOR. A grasp that is successfully held under excess
       force is consistent with every higher mu/m ratio and gives the
       slip-acceleration factor a flat (uninformative) residual; ramping
       through the true slip threshold is what actually excites it.
    3. Steady state: the physics-based control law, fed the estimate the
       probe produced. The estimate is committed once, at the tick the
       probe ends, rather than re-applied every tick: feeding a changing
       grip force back into the very window used to estimate the next
       grip force is a closed loop on top of the control loop, and it
       measurably oscillates between the correct answer and the prior
       rather than settling (see README "Findings"). The sliding-window
       solve itself keeps running every tick for the rest of the trial
       purely so its latency can still be benchmarked under the policy
       it is not, post-commit, steering.
    """

    BOOTSTRAP_TICKS = config.BOOTSTRAP_TICKS
    PROBE_TICKS = config.PROBE_TICKS
    SETTLE_TICKS = config.SETTLE_TICKS

    def __init__(self, control_dt):
        self.estimator = SlidingWindowPropertyEstimator(control_dt)
        self.last_estimate = (config.MASS_PRIOR, config.MU_PRIOR)
        self._committed = None

    def __call__(self, tick, box_z_vision, n_meas, palm_z_true):
        n_applied = getattr(self, "_last_n_applied", 0.0)
        m_hat, mu_hat = self.estimator.update(box_z_vision, palm_z_true, n_applied, n_meas)
        self.last_estimate = (m_hat, mu_hat)

        ramp_end_tick = self.BOOTSTRAP_TICKS + self.PROBE_TICKS
        commit_tick = ramp_end_tick + self.SETTLE_TICKS
        if tick < self.BOOTSTRAP_TICKS:
            cmd = config.GRIP_CTRL_MAX
        elif tick < ramp_end_tick:
            frac = (tick - self.BOOTSTRAP_TICKS) / self.PROBE_TICKS
            cmd = config.GRIP_CTRL_MAX + frac * (config.PROBE_FLOOR - config.GRIP_CTRL_MAX)
        elif tick < commit_tick:
            cmd = config.PROBE_FLOOR
        else:
            if self._committed is None:
                self._committed = (m_hat, mu_hat)
                self.last_estimate = self._committed
            cmd = grip_force_from_properties(*self._committed)
        self._last_n_applied = cmd
        return cmd


class BlindPolicy:
    """Property-blind baseline: one fixed grip force, set from the training
    distribution's mean mass and friction, never updated from sensor data."""

    def __init__(self):
        self.last_estimate = (config.MASS_PRIOR, config.MU_PRIOR)
        self._cmd = grip_force_from_properties(config.MASS_PRIOR, config.MU_PRIOR)

    def __call__(self, tick, box_z_vision, n_meas, palm_z_true):
        return self._cmd
