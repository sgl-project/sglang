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
#ifndef SGL_FLASH_HELPER_CUH_
#define SGL_FLASH_HELPER_CUH_

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>  // upstream

#include "hip/hip_vec_dtypes.h"  // upstream
#include "utils.h"
#else
#include <flashinfer/pos_enc.cuh>  // upstream
#endif

#ifdef USE_ROCM
#define EnablePDL(val, enable_pdl) enable_pdl;
#else
#define EnablePDL(val, enable_pdl) val.programmaticStreamSerializationAllowed = enable_pdl;
#endif
#ifdef USE_ROCM

#define FLASHINFER_ERROR(message) throw sgl_hip::Error(__FUNCTION__, __FILE__, __LINE__, message)

// FLASHINFER_CUDA_CALL definition
#ifndef NDEBUG
#define FLASHINFER_CUDA_CALL(func, ...)                                                                              \
  {                                                                                                                  \
    cudaError_t e = (func);                                                                                          \
    if (e != cudaSuccess) {                                                                                          \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e << ") " << __FILE__ << ": line " << __LINE__ \
                << " at function " << STR(func) << std::endl;                                                        \
      return e;                                                                                                      \
    }                                                                                                                \
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

#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    const bool INTERLEAVE = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool INTERLEAVE = false;                       \
    __VA_ARGS__                                          \
  }

namespace sgl_hip {

// adapted from flashinfer, once it's supported amd officially, we can remove this part
class Error : public std::exception {
 private:
  std::string message_;

 public:
  Error(const std::string& func, const std::string& file, int line, const std::string& message) {
    std::ostringstream oss;
    oss << "Error in function '" << func << "' "
        << "at " << file << ":" << line << ": " << message;
    message_ = oss.str();
  }

  virtual const char* what() const noexcept override {
    return message_.c_str();
  }
};

__host__ __device__ __forceinline__ size_t
get_elem_offset_impl(size_t elem_idx, size_t head_idx, size_t feat_idx, size_t stride_n, size_t stride_h) {
  return elem_idx * stride_n + head_idx * stride_h + feat_idx;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin(
    const T* x,
    const vec_t<float, vec_size>& cos,
    const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    permuted_vec.cast_load(
        x + ((threadIdx.x * vec_size < rotary_dim / 2) ? threadIdx.x * vec_size + rotary_dim / 2
                                                       : threadIdx.x * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] =
          vec[i] * cos[i] + ((threadIdx.x * vec_size < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin[i];
    }
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin_interleave_reuse_half(
    const T* x,
    const vec_t<float, vec_size>& cos,
    const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      // i / 2 is to get the index of the first half of cos and sin
      vec[i] = vec[i] * cos[i / 2] + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin[i / 2];
    }
  }
  return vec;
}

using cudaLaunchConfig_t = hipLaunchConfig_t;
using cudaLaunchAttribute = hipLaunchAttribute;

#define cudaLaunchAttributeProgrammaticStreamSerialization hipLaunchAttributeAccessPolicyWindow

template <typename Func_t, typename... ArgTypes>
__host__ __forceinline__ cudaError_t
cudaLaunchKernelEx(const cudaLaunchConfig_t* config, const Func_t kernel, ArgTypes&&... args) {
  kernel<<<config->gridDim, config->blockDim, config->dynamicSmemBytes, config->stream>>>(
      std::forward<ArgTypes>(args)...);
  return cudaGetLastError();
}

}  // namespace sgl_hip
#endif  // USE_ROCM
#endif  // SGL_POS_ENC_CUH_
