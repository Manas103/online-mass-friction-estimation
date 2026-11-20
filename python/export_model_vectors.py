"""Writes the forward-model test vectors the C++ Eigen port is diffed
against: random (n_star, 15-sample force window) pairs and the Python
model's predicted displacement-magnitude output for each, in the same
train-in-Python-verify-in-C++ spirit as the sibling repo's perception
scorer."""

import os
import struct

import numpy as np

import config
from property_estimator import predicted_displacement_magnitude

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "cpp", "testdata")
N_CASES = 2000
WINDOW = config.WINDOW_TICKS


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(20261002)
    n_star = rng.uniform(0.1, config.GRIP_CTRL_MAX, size=N_CASES).astype(np.float32)
    n_meas = rng.uniform(0.0, config.GRIP_CTRL_MAX, size=(N_CASES, WINDOW)).astype(np.float32)

    expected = np.stack([
        predicted_displacement_magnitude(n_star[i], n_meas[i]) for i in range(N_CASES)
    ]).astype(np.float32)

    with open(os.path.join(OUT_DIR, "cases.bin"), "wb") as f:
        f.write(struct.pack("<ii", N_CASES, WINDOW))
        f.write(n_star.tobytes())
        f.write(n_meas.tobytes())
        f.write(expected.tobytes())

    print(f"wrote {N_CASES} cases, window={WINDOW}, to {OUT_DIR}/cases.bin")


if __name__ == "__main__":
    main()
