#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "property_model.hpp"

namespace {

struct Cases {
  int n_cases;
  int window;
  std::vector<float> n_star;
  std::vector<float> n_meas;    // n_cases * window
  std::vector<float> expected;  // n_cases * window
};

Cases LoadCases(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "failed to open " << path << "\n";
    std::exit(1);
  }
  Cases c;
  int32_t n_cases, window;
  in.read(reinterpret_cast<char*>(&n_cases), sizeof(n_cases));
  in.read(reinterpret_cast<char*>(&window), sizeof(window));
  c.n_cases = n_cases;
  c.window = window;
  c.n_star.resize(n_cases);
  c.n_meas.resize(static_cast<size_t>(n_cases) * window);
  c.expected.resize(static_cast<size_t>(n_cases) * window);
  in.read(reinterpret_cast<char*>(c.n_star.data()), c.n_star.size() * sizeof(float));
  in.read(reinterpret_cast<char*>(c.n_meas.data()), c.n_meas.size() * sizeof(float));
  in.read(reinterpret_cast<char*>(c.expected.data()), c.expected.size() * sizeof(float));
  return c;
}

int RunDiff(const std::string& testdata_path) {
  Cases c = LoadCases(testdata_path);
  float max_abs_diff = 0.0f;
  for (int i = 0; i < c.n_cases; ++i) {
    Eigen::VectorXf n_meas = Eigen::Map<const Eigen::VectorXf>(&c.n_meas[static_cast<size_t>(i) * c.window], c.window);
    Eigen::VectorXf pred = property_model::PredictedDisplacementMagnitude(c.n_star[i], n_meas);
    for (int j = 0; j < c.window; ++j) {
      float diff = std::abs(pred(j) - c.expected[static_cast<size_t>(i) * c.window + j]);
      max_abs_diff = std::max(max_abs_diff, diff);
    }
  }
  std::cout << "cases: " << c.n_cases << " window: " << c.window << "\n";
  std::cout << "max abs diff vs python: " << max_abs_diff << "\n";
  return 0;
}

int RunBench(const std::string& testdata_path) {
  Cases c = LoadCases(testdata_path);
  std::vector<double> latencies_us;
  latencies_us.reserve(c.n_cases);
  for (int i = 0; i < c.n_cases; ++i) {
    Eigen::VectorXf n_meas = Eigen::Map<const Eigen::VectorXf>(&c.n_meas[static_cast<size_t>(i) * c.window], c.window);
    auto t0 = std::chrono::high_resolution_clock::now();
    Eigen::VectorXf pred = property_model::PredictedDisplacementMagnitude(c.n_star[i], n_meas);
    volatile float sink = pred(0);
    (void)sink;
    auto t1 = std::chrono::high_resolution_clock::now();
    latencies_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  std::sort(latencies_us.begin(), latencies_us.end());
  auto pct = [&](double p) { return latencies_us[static_cast<size_t>(p * (latencies_us.size() - 1))]; };
  std::cout << "n: " << latencies_us.size() << "\n";
  std::cout << "p50 us: " << pct(0.50) << "\n";
  std::cout << "p95 us: " << pct(0.95) << "\n";
  std::cout << "p99 us: " << pct(0.99) << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "--diff";
  std::string testdata_path = argc > 2 ? argv[2] : "testdata/cases.bin";
  if (mode == "--diff") return RunDiff(testdata_path);
  if (mode == "--bench") return RunBench(testdata_path);
  std::cerr << "usage: property_model_main [--diff|--bench] [testdata/cases.bin]\n";
  return 1;
}
