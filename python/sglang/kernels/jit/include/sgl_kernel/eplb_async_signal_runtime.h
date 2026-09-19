#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace sglang::eplb {

struct EplbAsyncSignal {
  uint64_t step_and_owner = 0;
};

constexpr uint64_t kSignalOwnerGpu = 0ULL;
constexpr uint64_t kSignalOwnerCpu = 1ULL;
constexpr uint64_t kSignalOwnerMask = 1ULL;
constexpr uint64_t kSignalSkipStep = 1ULL << 1U;
constexpr uint64_t kSignalStepShift = 2ULL;
constexpr uint64_t kSignalDisabled = 1ULL << 63U;
constexpr uint64_t kSignalStepMask = ~(kSignalOwnerMask | kSignalSkipStep | kSignalDisabled);

__host__ __device__ inline uint64_t encode_signal(int64_t step, uint64_t owner, bool skip_step, bool disabled) {
  uint64_t encoded = (static_cast<uint64_t>(step) << kSignalStepShift) | (owner & kSignalOwnerMask);
  if (skip_step) {
    encoded |= kSignalSkipStep;
  }
  if (disabled) {
    encoded |= kSignalDisabled;
  }
  return encoded;
}

__host__ __device__ inline bool is_signal_disabled(uint64_t signal_value) {
  return (signal_value & kSignalDisabled) != 0;
}

__host__ __device__ inline bool should_skip_signal_step(uint64_t signal_value) {
  return (signal_value & kSignalSkipStep) != 0;
}

__host__ __device__ inline uint64_t decode_signal_owner(uint64_t signal_value) {
  return signal_value & kSignalOwnerMask;
}

__host__ __device__ inline int64_t decode_signal_step(uint64_t signal_value) {
  return static_cast<int64_t>((signal_value & kSignalStepMask) >> kSignalStepShift);
}

void wait_signal_for_gpu_stage_device(
    EplbAsyncSignal* signal,
    int* enabled,
    int64_t* h2d_done_flag,
    int64_t expected_h2d_generation,
    cudaStream_t stream);

void set_signal_for_cpu_stage_device(EplbAsyncSignal* signal, cudaStream_t stream);

}  // namespace sglang::eplb
