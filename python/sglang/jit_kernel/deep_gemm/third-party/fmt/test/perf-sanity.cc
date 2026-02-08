// A quick and dirty performance test.
// For actual benchmarks see https://github.com/fmtlib/format-benchmark.

#include <atomic>
#include <chrono>
#include <iterator>

#include "fmt/format.h"

int main() {
  const int n = 10000000;

  auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < n; ++iteration) {
    auto buf = fmt::memory_buffer();
    fmt::format_to(std::back_inserter(buf),
                   "Hello, {}. The answer is {} and {}.", 1, 2345, 6789);
  }
  std::atomic_signal_fence(std::memory_order_acq_rel);  // Clobber memory.
  auto end = std::chrono::steady_clock::now();

  // Print time in milliseconds.
  std::chrono::duration<double> duration = end - start;
  fmt::print("{:.1f}\n", duration.count() * 1000);
}
