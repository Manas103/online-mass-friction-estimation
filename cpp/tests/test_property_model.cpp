#include <cassert>
#include <cmath>
#include <cstdio>

#include "../property_model.hpp"

namespace {

void TestHeldRegimeIsNearZero() {
  // n_star well below every applied force: should predict ~0 displacement.
  Eigen::VectorXf n_meas(5);
  n_meas << 10.0f, 11.0f, 12.0f, 10.5f, 11.5f;
  Eigen::VectorXf pred = property_model::PredictedDisplacementMagnitude(1.0f, n_meas);
  for (int i = 0; i < pred.size(); ++i) {
    assert(pred(i) >= 0.0f);
    assert(pred(i) < 1e-4f);
  }
  std::printf("TestHeldRegimeIsNearZero passed\n");
}

void TestSlipRegimeGrowsWithUndershoot() {
  // As n_meas drops further below n_star, predicted displacement should
  // monotonically increase (the model's whole point).
  float n_star = 5.0f;
  Eigen::VectorXf n_meas(4);
  n_meas << 4.9f, 3.0f, 1.0f, 0.0f;
  Eigen::VectorXf pred = property_model::PredictedDisplacementMagnitude(n_star, n_meas);
  for (int i = 1; i < pred.size(); ++i) {
    assert(pred(i) > pred(i - 1));
  }
  std::printf("TestSlipRegimeGrowsWithUndershoot passed\n");
}

void TestDeterministic() {
  Eigen::VectorXf n_meas(3);
  n_meas << 2.0f, 2.0f, 2.0f;
  Eigen::VectorXf a = property_model::PredictedDisplacementMagnitude(3.0f, n_meas);
  Eigen::VectorXf b = property_model::PredictedDisplacementMagnitude(3.0f, n_meas);
  for (int i = 0; i < a.size(); ++i) {
    assert(a(i) == b(i));
  }
  std::printf("TestDeterministic passed\n");
}

void TestFinite() {
  Eigen::VectorXf n_meas(3);
  n_meas << 1e6f, -1e6f, 0.0f;
  Eigen::VectorXf pred = property_model::PredictedDisplacementMagnitude(3.0f, n_meas);
  for (int i = 0; i < pred.size(); ++i) {
    assert(std::isfinite(pred(i)));
  }
  std::printf("TestFinite passed\n");
}

}  // namespace

int main() {
  TestHeldRegimeIsNearZero();
  TestSlipRegimeGrowsWithUndershoot();
  TestDeterministic();
  TestFinite();
  std::printf("check_property_model: 4 checks passed\n");
  return 0;
}
