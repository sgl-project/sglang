#pragma once

#ifdef __CUDACC__
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#endif

namespace device {

inline constexpr float FP8_E4M3_MAX = 448.0f;

}  // namespace device
