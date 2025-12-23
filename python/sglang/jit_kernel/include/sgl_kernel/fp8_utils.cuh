#pragma once

#include <c10/util/Float8_e4m3fn.h>

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

namespace device {

#ifndef USE_ROCM
using FP8_TYPE = c10::Float8_e4m3fn;
inline constexpr auto FP8_E4M3_MAX = std::numeric_limits<FP8_TYPE>::max();
#else
#if HIP_FP8_TYPE_FNUZ
using FP8_TYPE = c10::Float8_e4m3fnuz;
inline constexpr auto FP8_E4M3_MAX = 224.0f;
#else
#if HIP_FP8_TYPE_E4M3
using FP8_TYPE = c10::Float8_e4m3fn;
inline constexpr auto FP8_E4M3_MAX = std::numeric_limits<FP8_TYPE>::max();
#endif
#endif
#endif

}  // namespace device
