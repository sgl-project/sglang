#pragma once

#include <sgl_kernel/tensor.h>

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

namespace host {
namespace details {

// dtype_trait specializations for c10 types
template <>
struct dtype_trait<c10::Half> {
  inline static constexpr DLDataType value = {.code = DLDataTypeCode::kDLFloat, .bits = 16, .lanes = 1};
};

template <>
struct dtype_trait<c10::BFloat16> {
  inline static constexpr DLDataType value = {.code = DLDataTypeCode::kDLBfloat, .bits = 16, .lanes = 1};
};

template <>
struct dtype_trait<c10::Float8_e4m3fn> {
  inline static constexpr DLDataType value = {.code = DLDataTypeCode::kDLFloat, .bits = 8, .lanes = 1};
};

#ifdef __CUDACC__
// Alias for __nv_fp8_e4m3 which is the same as c10::Float8_e4m3fn
template <>
struct dtype_trait<__nv_fp8_e4m3> {
  inline static constexpr DLDataType value = {.code = DLDataTypeCode::kDLFloat, .bits = 8, .lanes = 1};
};
#endif

}  // namespace details
}  // namespace host

namespace device {

#ifndef USE_ROCM
using FP8_TYPE = c10::Float8_e4m3fn;
C10_HOST_DEVICE constexpr auto FP8_E4M3_MAX = std::numeric_limits<FP8_TYPE>::max();
#else
#if HIP_FP8_TYPE_FNUZ
using FP8_TYPE = c10::Float8_e4m3fnuz;
constexpr auto FP8_E4M3_MAX = 224.0f;
#else
#if HIP_FP8_TYPE_E4M3
using FP8_TYPE = c10::Float8_e4m3fn;
C10_HOST_DEVICE constexpr auto FP8_E4M3_MAX = std::numeric_limits<FP8_TYPE>::max();
#endif
#endif
#endif

}  // namespace device
