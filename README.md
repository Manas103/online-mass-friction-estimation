# Online Mass and Friction Estimation Inside a Dexterous Manipulation Loop

A two-finger MuJoCo grasp that fuses a noisy vision reading of box height
and a noisy fingertip force sensor in a sliding-window factor graph to
estimate, online, the minimum grip force a held object actually needs, then
conditions a physics-based grip-force policy on that estimate and compares
it against a property-blind baseline across 600 out-of-distribution
simulated trials. Python, MuJoCo, scipy, C++17/Eigen. Every number below
was measured on this machine, not targeted; the factor graph's first,
theoretically-motivated design did not fit the data well, and that failure,
the root cause, and the fix are reported in "Findings" rather than hidden.

## Why this exists

This extends
[gps-denied-state-estimation](https://github.com/Manas103/gps-denied-state-estimation):
that repo's sliding-window Levenberg-Marquardt factor graph and its
train-in-Python-verify-in-C++ habit carry over, but its domain (GPS-denied
aerial navigation) has no contact, no grasped object and no policy. This
repo keeps the estimator machinery and replaces the domain: a dexterous
manipulation loop that has to learn an unknown object's holding
requirements from touch and sight, the way "learning object state,
physical properties and interaction dynamics... to support robust, adaptive
robot behavior" is actually done.

## What this is, honestly

- **Simulated contact, not a real hand.** Everything here is MuJoCo's own
  rigid-body contact solver; there is no real sensor data anywhere in this
  repository.
- **A two-finger parallel squeeze, not a dexterous hand.** The gripper is
  two slide-jointed finger pads and a vertically slide-jointed palm; this
  is the smallest rig that poses the actual problem (how hard to squeeze an
  object of unknown mass and friction), not a reduced version of a full
  hand.
- **The box is constrained to vertical slip only** (a single slide joint,
  not a free 6-DOF joint). An early free-jointed version let the box tip
  and rotate once it started slipping, which let it escape the fingers'
  contact area far faster and more erratically than the Coulomb model
  below was ever going to predict; constraining it to the one degree of
  freedom the grasp problem is actually about is disclosed here as a
  deliberate simplification, not a hidden one.
- **Mass and friction are not independently measurable from this sensor
  set, only their ratio is**, and the estimator's own fit converged to a
  model-free threshold detector rather than the theoretically-motivated
  Coulomb acceleration model first attempted. Both are measured findings,
  not assumptions; see "Findings".
- **"Vision" is a noisy scalar height reading, not a rendered camera and a
  CNN.** The sibling repo already covers the trained-perception-model
  pattern; this project's vision channel is a simulated noisy position
  sensor, stated as such.
- **PyTorch was not load-bearing for the final design.** No part of this
  problem (a hand-designed physics factor fit by scipy, and a model-free
  slip-threshold detector) needed a learned model; disclosed rather than
  padded in to match a stack line.

**Machine and toolchain.** WSL2 Ubuntu 22.04 on Windows 11 (C++ model:
g++ 11.4.0, Eigen 3.4.0, CMake 3.22.1, `-O2`), a Python 3.12 venv on the
Windows side for the simulator and estimator (MuJoCo 4.0.1, scipy 1.18.1,
numpy 2.5.5). The simulation and the 600-trial benchmark are CPU-only;
nothing here needed a GPU.

## Architecture

```
python/
  gripper_model.xml      MuJoCo scene: a box (slide joint, z only) between
                          two force-actuated finger pads on a position-
                          actuated palm, plus a 3-axis force sensor per
                          finger.
  config.py              Every simulation, estimator and policy constant,
                          in one place.
  simulate.py             Runs one trial: bootstrap, probe, settle, lift,
                          returns success and the estimate history.
  property_estimator.py   The sliding-window factor graph (see below).
  policy.py                The adaptive (estimate-conditioned) and the
                          property-blind grip-force controllers, sharing
                          one physics-based control law.
  run_experiment.py        Runs the adaptive and blind policies over the
                          identical 600 OOD (mass, mu) draws and noise
                          seeds, reports success rate and solve-time
                          percentiles.
  export_model_vectors.py  Writes the C++ diff test vectors.
cpp/
  property_model.hpp       Eigen port of the factor's forward model
                          (predicted_displacement_magnitude), line-for-line
                          against property_estimator.py.
  property_model_main.cpp  --diff (max abs diff vs Python) and --bench
                          (p50/p95/p99 latency) modes.
  tests/test_property_model.cpp  Hand-rolled structural checks.
tests/
  test_estimator.py         pytest: softplus and model-shape invariants, a
                          brute-force grid-search reference oracle diffed
                          against the least-squares fit, the multi-start
                          regression test, and a few `@pytest.mark.slow`
                          full-simulator sanity checks.
docs/
  pytest_output.txt, test_output.txt, scorer_output.txt,
  asan_clean_run.txt, estimation_output.txt   Raw measured output.
```

### Design deep-dives

**Why a bootstrap, then a ramped-down probe, then a settle, before the
policy ever uses its own estimate.** An object that slips past the
fingers' contact height before the estimator has any data is lost for the
rest of the trial, so the first `BOOTSTRAP_TICKS` grip at the actuator's
safe maximum purely to secure the object. A grasp that is successfully
held under that much excess force is consistent with every higher
mu/mass ratio and gives the slip factor a flat, uninformative residual
(see "Findings"), so the probe deliberately ramps the grip force down
through the true slip threshold, the way a real adaptive-grasp controller
finds the minimum sufficient force, and the settle phase holds at the
ramp's floor long enough for the whole sliding window to be full of that
informative, consistent signal before the estimate is committed.

**Why the estimate is committed once, not re-applied every tick.** Feeding
a changing grip force back into the very window used to estimate the next
grip force is a closed loop stacked on top of the control loop; measured
behavior oscillated between the correct answer and the prior rather than
settling (see "Findings"). The sliding-window solve keeps running every
tick for the rest of the trial so its latency can still be benchmarked
under realistic, continuously-arriving data, but the committed estimate,
not its later re-solves, drives the grip-force policy.

**Why the control law and the estimator both key on a single ratio, not
mass and mu separately.** A Coulomb grasp held or slipping depends only on
`N_FINGERS * mu / mass`, never on mass and mu individually; see
"Findings" for how this was discovered and how it is handled.

## Validation

```
$ python -m pytest tests/ -q   # (see docs/pytest_output.txt)
10 passed
```

Includes a brute-force grid-search reference oracle
(`test_estimator_matches_reference_oracle_on_synthetic_window`), a
noiseless known-ground-truth recovery check, and a regression test for the
multi-start fix (`test_multistart_escapes_a_bad_warm_start`) pinned at the
exact corner value the single-start version got permanently stuck at.

```
$ ./cpp/build/check_property_model   # (see docs/test_output.txt)
check_property_model: 4 checks passed
$ ./cpp/build/property_model_main --diff   # (see docs/scorer_output.txt)
max abs diff vs python: 3.72529e-09
```

Sanitizers (`docs/asan_clean_run.txt`): AddressSanitizer+UBSan clean on the
C++ model's check suite. No TSan run: the model is single-threaded, with no
concurrency for TSan's instrumentation to exercise.

## Findings

**A hard `max(0, x)` clip in the slip-or-held switch had exactly zero
gradient on the "held" side, and the sliding-window solver got stuck
exactly at its initial guess every tick, regardless of how wrong that
guess was.** The first version of the Coulomb factor modeled "held" as
`max(0, g - k*N)` for the slip acceleration; whenever the current guess of
`k` already predicted "held" at the window's applied forces, every
parameter perturbation scipy's finite-difference Jacobian tried landed in
that same flat, zero-derivative region, so `least_squares` reported
convergence after zero effective movement. Replacing the clip with a
softplus relaxation gave every point a small but nonzero slope and let the
solver move at all.

**The Coulomb slip-acceleration model, even with that fix, measurably lost
to a trivial constant-velocity explanation of the same window.** Fitting
mass and mu (or their ratio) by integrating `a = g - k*N` forward from an
unknown window-start velocity and matching the noisy vision trace, the
best-fit cost at the *true* parameters was consistently higher than the
best-fit cost at a "no slip, just drift" explanation of the identical
window (measured: 0.0011 vs 0.00005 summed-squared residual on one
representative window). The 125ms window is too short, and MuJoCo's
contact solver does not decelerate a slipping grasp the way an idealized
point-mass Coulomb model predicts, for the acceleration curvature to be
distinguishable from noise and from the window's own unknown initial
velocity. The model that replaced it does not try to match the post-slip
trajectory shape at all: it fits a single critical-force parameter `N*`
against the *magnitude* of displacement away from the probe's starting
position, which is monotonic once slip begins (gravity pulls one way)
regardless of the exact acceleration profile, and does not need curvature
to be identified.

**Mass and friction are only identifiable as a ratio from this sensor set,
not independently, and this is a property of the physics, not of the
estimator.** A Coulomb grasp held or slipping depends only on
`N_FINGERS * mu / mass`; no amount of varying the grip force within a
trial changes that. The estimator fits the one quantity the physics
actually constrains (`N*`, equivalently `mu/mass` through `g/N*`) and
reports mass pinned at its training-distribution prior with mu backed out
from that fixed mass, disclosed in `property_estimator.py` as a
convention rather than presented as an independent mass measurement.

**Warm-starting the per-tick solver from the previous tick's estimate let
one bad early fit get permanently stuck at the solver's lower bound, even
once later window data was clearly informative.** Measured: committed
estimates on individual trials showed `n_star` pinned at exactly
`N_STAR_BOUNDS[0] = 0.1` while the actual window of (force, displacement)
pairs available at that tick, checked by hand, fit a value near 3.1
cleanly. The softplus model's gradient vanishes far from the window's own
force range, so a single warm start that drifted there during an earlier,
uninformative (held) window had no way back. Restarting the solver from a
few fixed points spread across the actuator's range every tick, in
addition to the previous estimate, and keeping whichever converges to the
lowest cost, fixed it; `test_multistart_escapes_a_bad_warm_start` pins the
regression. This raised OOD success from indistinguishable-from-baseline
to the measured result below; the 120 Hz budget comfortably covers the
extra restarts (p99 5.26ms of 8.33ms).

**Re-applying a continuously-updated estimate to the live grip force
created a feedback loop that measurably oscillated rather than converged.**
Before the estimate was committed once (see "Design deep-dives"), traced
trials showed the commanded grip force alternating between the correct,
converged value and the prior's value tick to tick, because changing the
grip force changes the very data the next solve is fit against. Committing
the estimate once, at the end of the probe/settle phase, removed the
oscillation.

## Measured results

Machine: WSL2 Ubuntu 22.04 on Windows 11 (C++), Windows 11 Python 3.12 venv
(simulator and estimator), both CPU-only.

**Sliding-window factor graph solve latency** (Python, the loop actually
used; `docs/estimation_output.txt`), per control tick, multi-start
least-squares over a 15-sample window, 600 trials: **p50 2.44ms, p95
4.24ms, p99 5.26ms**, against the claimed 120 Hz / 8.33ms budget (met,
with roughly 37% headroom at p99). The C++/Eigen port of the factor's
forward model alone (not the solver; same discipline as the sibling repo)
runs in p99 0.32us (`docs/scorer_output.txt`), confirming the model
evaluation itself is nowhere close to the bottleneck.

**Out-of-distribution grasp success, 600 simulated trials** (identical
(mass, mu) draws and noise seeds for both policies; `docs/estimation_output.txt`):

| Policy | Success rate |
|---|---|
| Property-blind baseline (fixed grip, tuned to the training distribution's mean mass and friction) | **34.5%** (207/600) |
| Adaptive (grip conditioned on the committed online estimate) | **93.7%** (562/600) |

The claimed shape (a property-blind baseline degrading further out of
distribution than an estimate-conditioned policy) is confirmed; the
measured gap (34.5% to 93.7%) is larger than the claimed 34% to 71%, not
tuned to match either number. `mean_abs_mu_error` across all 600 trials
was 0.349 (friction coefficients here range roughly 0.2-1.0), consistent
with "Findings": mu is backed out from a mass pinned at its prior, not
independently measured, so its individual error is not the headline
number; the grip-force policy's success rate, which depends only on the
physically identifiable `mass/mu` ratio, is.

## Building and running

```bash
# Python side (from a venv with requirements.txt installed)
cd python
python export_model_vectors.py        # writes cpp/testdata/cases.bin
python run_experiment.py --n-trials 600

cd ..
python -m pytest tests/ -q

# C++ model (WSL2 / Linux, from cpp/)
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/check_property_model
./build/property_model_main --diff
./build/property_model_main --bench

# Sanitizers
cmake -S . -B build_asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
cmake --build build_asan -j2
./build_asan/check_property_model
```

## Sibling comparison

[gps-denied-state-estimation](https://github.com/Manas103/gps-denied-state-estimation)
is where the sliding-window factor-graph pattern and the
train-in-Python-verify-in-C++ habit both come from. That repo's factor
graph did not reach its RMS position-accuracy target after two genuine
attempts, with the honestly-measured root cause being the perception
model's bearing noise, not the estimator; this repo's first factor graph
also did not work, for a different and more structural reason (the model
family itself was a measurably worse fit than a trivial baseline on short
windows), and the fix was to replace the model, not to tune it. Both repos
report the failure and the root cause rather than a result adjusted to
look clean.

## Limitations

- The box is constrained to vertical slip only; a real dexterous grasp has
  lateral and rotational slip modes this project does not model.
- Mass and friction are only jointly identifiable as a ratio from vision
  and fingertip force alone; the individual mass and mu values this
  project reports use a fixed-mass convention, disclosed in
  `property_estimator.py`, not an independent measurement of either.
- The probe-and-settle calibration phase deliberately lets the object slip
  by a few millimeters before the adaptive policy's grip force is applied;
  the property-blind baseline never does this, so part of the success-rate
  gap reflects the adaptive policy actively probing rather than only its
  final grip force being more accurate.
- The grip-force control law (`SAFETY_FACTOR * N*`) is a simple physics-
  based rule, not a learned policy; "policy conditioned on the estimate"
  here means a rule that takes the estimate as input, not a neural network.
- The C++ port covers the factor's forward model only, not the
  least-squares solver itself, matching the sibling repo's own disclosed
  scope.
