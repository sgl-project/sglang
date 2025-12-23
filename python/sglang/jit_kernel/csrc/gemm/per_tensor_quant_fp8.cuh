// Include c10 headers BEFORE sgl_kernel/tensor.h to avoid std::source_location issues
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Half.h>

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

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <cub/block/block_reduce.cuh>
#include <flashinfer/vec_dtypes.cuh>

#include <cstddef>
#include <cstdint>

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
template <>
struct dtype_trait<__nv_fp8_e4m3> {
  inline static constexpr DLDataType value = {.code = DLDataTypeCode::kDLFloat, .bits = 8, .lanes = 1};
};
#endif

}  // namespace details
}  // namespace host

// FP8 constants and device functions
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

namespace {

using device::FP8_E4M3_MAX;

constexpr unsigned int FULL_MASK = 0xffffffff;
constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
#ifndef USE_ROCM
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
                     : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
#else
  int* addr_as_i = (int*)addr;
  int old = *addr_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_i, assumed, __float_as_int(fmaxf(value, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
#endif
}

__device__ __forceinline__ float warpReduceMax(float value) {
  value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 16));
  value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 8));
  value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 4));
  value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 2));
  value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 1));
  return value;
}

__device__ __forceinline__ float blockReduceMax(float value) {
  static __shared__ float warpLevelMaxs[WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;

  value = warpReduceMax(value);

  if (laneId == 0) warpLevelMaxs[warpId] = value;
  __syncthreads();

  value = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelMaxs[laneId] : 0;
  if (warpId == 0) value = warpReduceMax(value);

  return value;
}

template <typename T>
__global__ void per_tensor_absmax_kernel(const T* __restrict__ input,
                                         float* __restrict__ output_s,
                                         const int64_t num_elements) {
  float max_value = 0.0f;
  unsigned int tid = threadIdx.x;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_size = blockDim.x * gridDim.x;

  constexpr uint32_t vec_size = 16 / sizeof(T);
  using vec_t = flashinfer::vec_t<T, vec_size>;

  const int32_t num_vec_elems = num_elements / vec_size;

  for (int32_t i = gid; i < num_vec_elems; i += grid_size) {
    vec_t input_vec;
    input_vec.cast_load(input + i * vec_size);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = static_cast<float>(input_vec[j]);
      max_value = fmaxf(max_value, fabsf(val));
    }
  }

  const int32_t remaining_start = num_vec_elems * vec_size;
  for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size) {
    float val = static_cast<float>(input[idx]);
    max_value = fmaxf(max_value, fabsf(val));
  }

  max_value = blockReduceMax(max_value);

  if (tid == 0) {
    atomicMaxFloat(output_s, max_value / FP8_E4M3_MAX);
  }
}

template <typename T, typename DST_DTYPE>
__global__ void per_tensor_quant_fp8_kernel(const T* __restrict__ input,
                                            DST_DTYPE* __restrict__ output,
                                            const float* __restrict__ scale,
                                            const int64_t num_elements) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_size = blockDim.x * gridDim.x;
  const float scale_val = 1.0f / (*scale);

  const uint32_t VEC_SIZE = 16;
  using vec_t = flashinfer::vec_t<T, VEC_SIZE>;

  const int32_t num_vec_elems = num_elements / VEC_SIZE;

  for (int32_t i = gid; i < num_vec_elems; i += grid_size) {
    vec_t input_vec;
    input_vec.cast_load(input + i * VEC_SIZE);

    DST_DTYPE output_arr[VEC_SIZE];
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      float val = fmax(fmin(static_cast<float>(input_vec[j]) * scale_val, FP8_E4M3_MAX),
                       -FP8_E4M3_MAX);
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
      output_arr[j] = static_cast<DST_DTYPE>(val);
#else
      output_arr[j] = c10::Float8_e4m3fnuz(
          __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation,
                                 fp8::fp8_type::__default_interpret),
          c10::Float8_e4m3fnuz::from_bits());
#endif
    }
    *(uint4*)(output + i * VEC_SIZE) = *(uint4*)output_arr;
  }

  const int32_t remaining_start = num_vec_elems * VEC_SIZE;
  for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size) {
    float val = fmax(-FP8_E4M3_MAX,
                     fmin(static_cast<float>(input[idx]) * scale_val, FP8_E4M3_MAX));
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
    output[idx] = static_cast<DST_DTYPE>(val);
#else
    output[idx] = c10::Float8_e4m3fnuz(
        __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation,
                               fp8::fp8_type::__default_interpret),
        c10::Float8_e4m3fnuz::from_bits());
#endif
  }
}

constexpr size_t kBlockSize = 256;

void per_tensor_quant_fp8_dynamic(tvm::ffi::TensorView input,
                                   tvm::ffi::TensorView output_q,
                                   tvm::ffi::TensorView output_s) {
  using namespace host;

  SymbolicSize num_elements = {"num_elements"};
  SymbolicDevice device_;
  SymbolicDType input_dtype;

  TensorMatcher({num_elements})
      .with_dtype<float, c10::Half, c10::BFloat16>(input_dtype)
      .with_device<kDLCUDA>(device_)
      .verify(input);

  TensorMatcher({num_elements})
      .with_dtype<__nv_fp8_e4m3>()
      .with_device<kDLCUDA>(device_)
      .verify(output_q);

  TensorMatcher({})
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(output_s);

  const size_t total_elements = num_elements.unwrap();
  const size_t num_blocks = std::min((total_elements + kBlockSize - 1) / kBlockSize, size_t(1024));
  const DLDevice device = device_.unwrap();

  auto launch_kernels = [&]<typename T>() {
    LaunchKernel(num_blocks, kBlockSize, device)(
        per_tensor_absmax_kernel<T>,
        static_cast<const T*>(input.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        static_cast<int64_t>(total_elements));

    LaunchKernel(num_blocks, kBlockSize, device)(
        per_tensor_quant_fp8_kernel<T, __nv_fp8_e4m3>,
        static_cast<const T*>(input.data_ptr()),
        static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
        static_cast<const float*>(output_s.data_ptr()),
        static_cast<int64_t>(total_elements));
  };

  const DLDataType dtype = input_dtype.unwrap();
  if (dtype.code == kDLFloat && dtype.bits == 32) {
    launch_kernels.template operator()<float>();
  } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
    launch_kernels.template operator()<c10::BFloat16>();
  } else if (dtype.code == kDLFloat && dtype.bits == 16) {
    launch_kernels.template operator()<c10::Half>();
  }
}

void per_tensor_quant_fp8_static(tvm::ffi::TensorView input,
                                  tvm::ffi::TensorView output_q,
                                  tvm::ffi::TensorView output_s) {
  using namespace host;

  SymbolicSize num_elements = {"num_elements"};
  SymbolicDevice device_;
  SymbolicDType input_dtype;

  TensorMatcher({num_elements})
      .with_dtype<float, c10::Half, c10::BFloat16>(input_dtype)
      .with_device<kDLCUDA>(device_)
      .verify(input);

  TensorMatcher({num_elements})
      .with_dtype<__nv_fp8_e4m3>()
      .with_device<kDLCUDA>(device_)
      .verify(output_q);

  TensorMatcher({})
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(output_s);

  const size_t total_elements = num_elements.unwrap();
  const size_t num_blocks = std::min((total_elements + kBlockSize - 1) / kBlockSize, size_t(1024));
  const DLDevice device = device_.unwrap();

  auto launch_kernels = [&]<typename T>() {
    LaunchKernel(num_blocks, kBlockSize, device)(
        per_tensor_quant_fp8_kernel<T, __nv_fp8_e4m3>,
        static_cast<const T*>(input.data_ptr()),
        static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
        static_cast<const float*>(output_s.data_ptr()),
        static_cast<int64_t>(total_elements));
  };

  const DLDataType dtype = input_dtype.unwrap();
  if (dtype.code == kDLFloat && dtype.bits == 32) {
    launch_kernels.template operator()<float>();
  } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
    launch_kernels.template operator()<c10::BFloat16>();
  } else if (dtype.code == kDLFloat && dtype.bits == 16) {
    launch_kernels.template operator()<c10::Half>();
  }
}

}  // namespace
