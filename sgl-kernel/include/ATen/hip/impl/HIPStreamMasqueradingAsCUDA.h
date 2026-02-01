#pragma once

#include <c10/hip/HIPFunctions.h>
#include <c10/hip/HIPStream.h>

namespace c10::hip {
inline c10::cuda::CUDAStream getCurrentHIPStreamMasqueradingAsCUDA() {
  return c10::cuda::getCurrentCUDAStream();
}

inline c10::DeviceIndex current_device() {
  return c10::cuda::current_device();
}
}  // namespace c10::hip

namespace at::cuda {
inline c10::cuda::CUDAStream getCurrentHIPStreamMasqueradingAsCUDA() {
  return c10::cuda::getCurrentCUDAStream();
}
}  // namespace at::cuda
