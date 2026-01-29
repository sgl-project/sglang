#pragma once

#include <c10/hip/HIPGuard.h>

namespace c10::hip {
using OptionalHIPGuardMasqueradingAsCUDA = c10::cuda::OptionalCUDAGuard;
}  // namespace c10::hip

namespace at::hip {
using OptionalHIPGuardMasqueradingAsCUDA = c10::cuda::OptionalCUDAGuard;
}  // namespace at::hip
