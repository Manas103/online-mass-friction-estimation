"""Every simulator, estimator and policy constant, in one place."""

import os

MODEL_XML = os.path.join(os.path.dirname(__file__), "gripper_model.xml")

# --- simulation ---
PHYSICS_DT = 0.002          # MuJoCo integrator timestep (500 Hz)
CONTROL_HZ = 120            # control/estimator loop rate claimed in the resume
CONTROL_SUBSTEPS = max(1, round(1.0 / CONTROL_HZ / PHYSICS_DT))  # physics steps per control tick
TRIAL_DURATION_S = 1.5
SQUEEZE_RAMP_S = 0.008      # grip ramps 0 -> commanded over this long before any lift
LIFT_START_S = 0.29         # lift motion starts this long after squeeze begins (after bootstrap+probe+settle)
LIFT_HEIGHT = 0.08          # meters the palm is commanded to rise

# --- sensing noise (both policies see the same noisy sensors) ---
VISION_NOISE_SIGMA = 0.0004   # m, noisy camera estimate of box world z
FORCE_NOISE_SIGMA = 0.03      # N, noisy fingertip force-sensor reading

# --- probing phase (adaptive policy only): a secure high-force grab
# (bootstrap) cannot localize the slip threshold -- a grasp that successfully
# holds under excess force is also consistent with every higher mu/m ratio,
# so the residual near a confident hold is flat. The probe ramps the grip
# force down through the slip boundary on purpose, the way a real adaptive
# grasp controller finds the minimum sufficient force, so the window
# actually contains a slip transition to fit against.
BOOTSTRAP_TICKS = 6
PROBE_TICKS = 16
PROBE_FLOOR = 3.0
# constant-force settle at PROBE_FLOOR after the ramp, long enough that the
# whole sliding window is eventually full of informative, consistent slip
# data before the estimate is committed (see README "Findings")
SETTLE_TICKS = 12

# --- sliding-window estimator ---
WINDOW_TICKS = 15            # ~125ms of control-loop history
MASS_BOUNDS = (0.02, 1.5)    # kg, solver bounds
MU_BOUNDS = (0.02, 2.0)      # solver bounds
MASS_PRIOR = 0.2             # kg, used before the window fills and as the blind baseline's assumed mass
MU_PRIOR = 0.75              # used before the window fills and as the blind baseline's assumed mu
GRAVITY = 9.81

# --- grip-force control law: N_cmd = SAFETY_FACTOR * m_hat * g / (N_FINGERS * mu_hat) ---
N_FINGERS = 2
SAFETY_FACTOR = 1.55
GRIP_CTRL_MAX = 18.0         # matches gripper_model.xml actuator ctrlrange
GRIP_CTRL_MIN = 0.05

# --- success criterion ---
HOLD_TOL = 0.03              # meters: |box_z - palm_z| at trial end must be under this to count as held

# --- distributions (kg, dimensionless friction coefficient) ---
TRAIN_MASS_RANGE = (0.10, 0.30)
TRAIN_MU_RANGE = (0.55, 1.00)
OOD_MASS_RANGE = (0.15, 0.38)
OOD_MU_RANGE = (0.35, 0.80)

N_OOD_TRIALS = 600
RANDOM_SEED = 20261001
