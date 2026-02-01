#pragma once

#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>

namespace c10::hip {
using OptionalHIPGuardMasqueradingAsCUDA = c10::cuda::OptionalCUDAGuard;
}  // namespace c10::hip

namespace at::hip {
using OptionalHIPGuardMasqueradingAsCUDA = c10::cuda::OptionalCUDAGuard;
inline hipStream_t getCurrentHIPStream() {
  return c10::cuda::getCurrentCUDAStream().stream();
}
}  // namespace at::hip
