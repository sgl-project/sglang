#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Half.h>

#include <dlpack/dlpack.h>

#include <limits>

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#if HIP_FP8_TYPE_FNUZ
#include <c10/util/Float8_e4m3fnuz.h>
#else
#if HIP_FP8_TYPE_E4M3
#include <c10/util/Float8_e4m3fn.h>
#endif
#endif
#endif

#ifdef __CUDACC__
#include <cuda_fp8.h>
#endif

namespace device {

#ifndef USE_ROCM
using FP8_TYPE = c10::Float8_e4m3fn;
constexpr float FP8_E4M3_MAX = 448.0f;
#else
#if HIP_FP8_TYPE_FNUZ
using FP8_TYPE = c10::Float8_e4m3fnuz;
constexpr float FP8_E4M3_MAX = 224.0f;
#else
#if HIP_FP8_TYPE_E4M3
using FP8_TYPE = c10::Float8_e4m3fn;
constexpr float FP8_E4M3_MAX = 448.0f;
#endif
#endif
#endif

}  // namespace device
