// Build: mkdir build && cd build && cmake .. && make -j
// Usage: ./bench_cpu_kernels [--iters 200]

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

at::Tensor silu_and_mul_cpu(at::Tensor& input);
at::Tensor gelu_tanh_and_mul_cpu(const at::Tensor& input);
at::Tensor gelu_and_mul_cpu(const at::Tensor& input);

double bench(std::function<void()> fn, int warmup = 20, int iters = 200) {
  for (int i = 0; i < warmup; ++i)
    fn();
  std::vector<double> times;
  times.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0);
  }
  std::sort(times.begin(), times.end());
  return times[iters / 2];
}

int main(int argc, char** argv) {
  int iters = 200;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);

  std::cout << "CPU Activation Kernel Benchmark (iters=" << iters << ")\n";
  std::cout << "Threads: " << at::get_num_threads() << "\n\n";
  std::cout << "  kernel            batch    dim    time(us)" << std::endl;

  std::vector<std::pair<int, int>> configs = {{1, 7168}, {32, 14336}, {128, 14336}, {512, 14336}};

  for (auto [bs, d] : configs) {
    auto input = torch::randn({bs, d * 2}, torch::kBFloat16);

    double t1 = bench([&]() { silu_and_mul_cpu(input); }, 20, iters);
    printf("  silu_and_mul      %5d  %5d  %10.1f\n", bs, d, t1);

    double t2 = bench([&]() { gelu_tanh_and_mul_cpu(input); }, 20, iters);
    printf("  gelu_tanh_and_mul %5d  %5d  %10.1f\n", bs, d, t2);

    double t3 = bench([&]() { gelu_and_mul_cpu(input); }, 20, iters);
    printf("  gelu_and_mul      %5d  %5d  %10.1f\n", bs, d, t3);
  }
  return 0;
}
