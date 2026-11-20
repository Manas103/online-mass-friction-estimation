#pragma once

#include <Eigen/Dense>
#include <cmath>

// Line-for-line port of python/property_estimator.py's
// predicted_displacement_magnitude(), the forward model the sliding-window
// least-squares solver evaluates every control tick. Not the solver itself
// (that stays scipy's, same discipline as the sibling repo's perception
// scorer): this is the piece a real-time C++ control loop would actually
// need to call at 120 Hz, and the piece this repo diffs and benchmarks.

namespace property_model {

constexpr float kDisplacementScale = 0.003f;
constexpr float kSlope = 5.0f;

inline float Softplus(float x, float beta = 1.0f) {
  float bx = beta * x;
  if (bx > 30.0f) return x;
  float clipped = std::max(-30.0f, std::min(30.0f, bx));
  return std::log1p(std::exp(clipped)) / beta;
}

// n_meas: one force-sensor reading per window sample.
// Returns: predicted displacement magnitude per sample.
inline Eigen::VectorXf PredictedDisplacementMagnitude(float n_star, const Eigen::VectorXf& n_meas) {
  Eigen::VectorXf out(n_meas.size());
  for (int i = 0; i < n_meas.size(); ++i) {
    out(i) = kDisplacementScale * Softplus(kSlope * (n_star - n_meas(i)));
  }
  return out;
}

}  // namespace property_model
