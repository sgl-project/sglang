/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_UTILS_CUH_
#define FLASHINFER_UTILS_CUH_
#include <cuda_bf16.h>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

#include "exception.h"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// macro to turn off fp16 qk reduction to reduce binary
#ifndef FLASHINFER_ALWAYS_DISUSE_FP16_QK_REDUCTION
#define FLASHINFER_ALWAYS_DISUSE_FP16_QK_REDUCTION 0
#endif

#ifndef NDEBUG
#define FLASHINFER_CUDA_CALL(func, ...)                                                     \
  {                                                                                         \
    cudaError_t e = (func);                                                                 \
    if (e != cudaSuccess) {                                                                 \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e << ") " << __FILE__ \
                << ": line " << __LINE__ << " at function " << STR(func) << std::endl;      \
      return e;                                                                             \
    }                                                                                       \
  }
#else
#define FLASHINFER_CUDA_CALL(func, ...) \
  {                                     \
    cudaError_t e = (func);             \
    if (e != cudaSuccess) {             \
      return e;                         \
    }                                   \
  }
#endif

#define DISPATCH_USE_FP16_QK_REDUCTION(use_fp16_qk_reduction, USE_FP16_QK_REDUCTION, ...) \
  if (use_fp16_qk_reduction) {                                                            \
    FLASHINFER_ERROR("FP16_QK_REDUCTION disabled at compile time");                       \
  } else {                                                                                \
    constexpr bool USE_FP16_QK_REDUCTION = false;                                         \
    __VA_ARGS__                                                                           \
  }

#define DISPATCH_NUM_MMA_Q(num_mma_q, NUM_MMA_Q, ...)  \
  if (num_mma_q == 1) {                                \
    constexpr size_t NUM_MMA_Q = 1;                    \
    __VA_ARGS__                                        \
  } else if (num_mma_q == 2) {                         \
    constexpr size_t NUM_MMA_Q = 2;                    \
    __VA_ARGS__                                        \
  } else {                                             \
    std::ostringstream err_msg;                        \
    err_msg << "Unsupported num_mma_q: " << num_mma_q; \
    FLASHINFER_ERROR(err_msg.str());                   \
  }

#define DISPATCH_NUM_MMA_KV(max_mma_kv, NUM_MMA_KV, ...) \
  if (max_mma_kv >= 8) {                                 \
    constexpr size_t NUM_MMA_KV = 8;                     \
    __VA_ARGS__                                          \
  } else if (max_mma_kv >= 4) {                          \
    constexpr size_t NUM_MMA_KV = 4;                     \
    __VA_ARGS__                                          \
  } else if (max_mma_kv >= 2) {                          \
    constexpr size_t NUM_MMA_KV = 2;                     \
    __VA_ARGS__                                          \
  } else if (max_mma_kv >= 1) {                          \
    constexpr size_t NUM_MMA_KV = 1;                     \
    __VA_ARGS__                                          \
  } else {                                               \
    std::ostringstream err_msg;                          \
    err_msg << "Unsupported max_mma_kv: " << max_mma_kv; \
    FLASHINFER_ERROR(err_msg.str());                     \
  }

#define DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, ...)   \
  switch (cta_tile_q) {                                    \
    case 128: {                                            \
      constexpr uint32_t CTA_TILE_Q = 128;                 \
      __VA_ARGS__                                          \
      break;                                               \
    }                                                      \
    case 64: {                                             \
      constexpr uint32_t CTA_TILE_Q = 64;                  \
      __VA_ARGS__                                          \
      break;                                               \
    }                                                      \
    case 16: {                                             \
      constexpr uint32_t CTA_TILE_Q = 16;                  \
      __VA_ARGS__                                          \
      break;                                               \
    }                                                      \
    default: {                                             \
      std::ostringstream err_msg;                          \
      err_msg << "Unsupported cta_tile_q: " << cta_tile_q; \
      FLASHINFER_ERROR(err_msg.str());                     \
    }                                                      \
  }

#define DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...) \
  if (group_size == 1) {                                     \
    constexpr size_t GROUP_SIZE = 1;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 2) {                              \
    constexpr size_t GROUP_SIZE = 2;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 3) {                              \
    constexpr size_t GROUP_SIZE = 3;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 4) {                              \
    constexpr size_t GROUP_SIZE = 4;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 8) {                              \
    constexpr size_t GROUP_SIZE = 8;                         \
    __VA_ARGS__                                              \
  } else {                                                   \
    std::ostringstream err_msg;                              \
    err_msg << "Unsupported group_size: " << group_size;     \
    FLASHINFER_ERROR(err_msg.str());                         \
  }

#define DISPATCH_MASK_MODE(mask_mode, MASK_MODE, ...)             \
  switch (mask_mode) {                                            \
    case MaskMode::kNone: {                                       \
      constexpr MaskMode MASK_MODE = MaskMode::kNone;             \
      __VA_ARGS__                                                 \
      break;                                                      \
    }                                                             \
    case MaskMode::kCausal: {                                     \
      constexpr MaskMode MASK_MODE = MaskMode::kCausal;           \
      __VA_ARGS__                                                 \
      break;                                                      \
    }                                                             \
    case MaskMode::kCustom: {                                     \
      constexpr MaskMode MASK_MODE = MaskMode::kCustom;           \
      __VA_ARGS__                                                 \
      break;                                                      \
    }                                                             \
    case MaskMode::kMultiItemScoring: {                           \
      constexpr MaskMode MASK_MODE = MaskMode::kMultiItemScoring; \
      __VA_ARGS__                                                 \
      break;                                                      \
    }                                                             \
    default: {                                                    \
      std::ostringstream err_msg;                                 \
      err_msg << "Unsupported mask_mode: " << int(mask_mode);     \
      FLASHINFER_ERROR(err_msg.str());                            \
    }                                                             \
  }

// convert head_dim to compile-time constant
#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)     \
  switch (head_dim) {                                  \
    case 64: {                                         \
      constexpr size_t HEAD_DIM = 64;                  \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    case 128: {                                        \
      constexpr size_t HEAD_DIM = 128;                 \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    case 256: {                                        \
      constexpr size_t HEAD_DIM = 256;                 \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    case 512: {                                        \
      constexpr size_t HEAD_DIM = 512;                 \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    default: {                                         \
      std::ostringstream err_msg;                      \
      err_msg << "Unsupported head_dim: " << head_dim; \
      FLASHINFER_ERROR(err_msg.str());                 \
    }                                                  \
  }

#define DISPATCH_POS_ENCODING_MODE(pos_encoding_mode, POS_ENCODING_MODE, ...)    \
  switch (pos_encoding_mode) {                                                   \
    case PosEncodingMode::kNone: {                                               \
      constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kNone;      \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    case PosEncodingMode::kRoPELlama: {                                          \
      constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kRoPELlama; \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    case PosEncodingMode::kALiBi: {                                              \
      constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kALiBi;     \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    default: {                                                                   \
      std::ostringstream err_msg;                                                \
      err_msg << "Unsupported pos_encoding_mode: " << int(pos_encoding_mode);    \
      FLASHINFER_ERROR(err_msg.str());                                           \
    }                                                                            \
  }

#define DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...) \
  switch (aligned_vec_size) {                                              \
    case 16: {                                                             \
      constexpr size_t ALIGNED_VEC_SIZE = 16;                              \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 8: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 8;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 4: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 4;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 2: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 2;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 1: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 1;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    default: {                                                             \
      std::ostringstream err_msg;                                          \
      err_msg << "Unsupported aligned_vec_size: " << aligned_vec_size;     \
      FLASHINFER_ERROR(err_msg.str());                                     \
    }                                                                      \
  }

#define DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, ...) \
  if (compute_capacity.first >= 8) {                                                        \
    constexpr uint32_t NUM_STAGES_SMEM = 2;                                                 \
    __VA_ARGS__                                                                             \
  } else {                                                                                  \
    constexpr uint32_t NUM_STAGES_SMEM = 1;                                                 \
    __VA_ARGS__                                                                             \
  }

namespace flashinfer {

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y) {
  return (x + y - 1) / y;
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 round_up(const T1 x, const T2 y) {
  return ceil_div(x, y) * y;
}

inline std::pair<int, int> GetCudaComputeCapability() {
  int device_id = 0;
  cudaGetDevice(&device_id);
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
  return std::make_pair(major, minor);
}

template <typename T>
inline void DebugPrintCUDAArray(T* device_ptr, size_t size, std::string prefix = "") {
  std::vector<T> host_array(size);
  std::cout << prefix;
  cudaMemcpy(host_array.data(), device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < size; ++i) {
    std::cout << host_array[i] << " ";
  }
  std::cout << std::endl;
}

inline uint32_t FA2DetermineCtaTileQ(int64_t avg_packed_qo_len, uint32_t head_dim) {
  if (avg_packed_qo_len > 64 && head_dim < 256) {
    return 128;
  } else {
    auto compute_capacity = GetCudaComputeCapability();
    if (compute_capacity.first >= 8) {
      // Ampere or newer
      if (avg_packed_qo_len > 16) {
        // avg_packed_qo_len <= 64
        return 64;
      } else {
        // avg_packed_qo_len <= 16
        return 16;
      }
    } else {
      // NOTE(Zihao): not enough shared memory on Turing for 1x4 warp layout
      return 64;
    }
  }
}

inline int UpPowerOfTwo(int x) {
  // Returns the smallest power of two greater than or equal to x
  if (x <= 0) return 1;
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

#define LOOP_SPLIT_MASK(iter, COND1, COND2, ...)       \
  {                                                    \
    _Pragma("unroll 1") for (; (COND1); (iter) -= 1) { \
      constexpr bool WITH_MASK = true;                 \
      __VA_ARGS__                                      \
    }                                                  \
    _Pragma("unroll 1") for (; (COND2); (iter) -= 1) { \
      constexpr bool WITH_MASK = false;                \
      __VA_ARGS__                                      \
    }                                                  \
  }

/*!
 * \brief Return x - y if x > y, otherwise return 0.
 */
__device__ __forceinline__ uint32_t sub_if_greater_or_zero(uint32_t x, uint32_t y) {
  return (x > y) ? x - y : 0U;
}

__device__ __forceinline__ void swap(uint32_t& a, uint32_t& b) {
  uint32_t tmp = a;
  a = b;
  b = tmp;
}

__device__ __forceinline__ uint32_t dim2_offset(const uint32_t& dim_a, const uint32_t& idx_b,
                                                const uint32_t& idx_a) {
  return idx_b * dim_a + idx_a;
}

__device__ __forceinline__ uint32_t dim3_offset(const uint32_t& dim_b, const uint32_t& dim_a,
                                                const uint32_t& idx_c, const uint32_t& idx_b,
                                                const uint32_t& idx_a) {
  return (idx_c * dim_b + idx_b) * dim_a + idx_a;
}

__device__ __forceinline__ uint32_t dim4_offset(const uint32_t& dim_c, const uint32_t& dim_b,
                                                const uint32_t& dim_a, const uint32_t& idx_d,
                                                const uint32_t& idx_c, const uint32_t& idx_b,
                                                const uint32_t& idx_a) {
  return ((idx_d * dim_c + idx_c) * dim_b + idx_b) * dim_a + idx_a;
}

#define DEFINE_HAS_MEMBER(member)                                                              \
  template <typename T, typename = void>                                                       \
  struct has_##member : std::false_type {};                                                    \
  template <typename T>                                                                        \
  struct has_##member<T, std::void_t<decltype(std::declval<T>().member)>> : std::true_type {}; \
  template <typename T>                                                                        \
  inline constexpr bool has_##member##_v = has_##member<T>::value;

}  // namespace flashinfer

#endif  // FLASHINFER_UTILS_CUH_
