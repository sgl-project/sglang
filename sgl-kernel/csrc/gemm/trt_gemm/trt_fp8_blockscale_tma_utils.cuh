// Adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_tma_utils.cuh
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cuda/barrier>
#include <cute/arch/util.hpp>

namespace tensorrt_llm::kernels::fp8_blockscale_gemm {

template <class T>
inline CUtensorMapDataType get_CUtensorMapDataType() {
  if constexpr (std::is_same<T, int8_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if constexpr (std::is_same<T, uint8_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if constexpr (std::is_same<T, __nv_fp8_e4m3>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if constexpr (std::is_same<T, __nv_fp8_e5m2>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if constexpr (std::is_same<T, uint16_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT16;
  } else if constexpr (std::is_same<T, uint32_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT32;
  } else if constexpr (std::is_same<T, uint64_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT64;
  } else if constexpr (std::is_same<T, int32_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_INT32;
  } else if constexpr (std::is_same<T, int64_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_INT64;
  } else if constexpr (std::is_same<T, __half>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if constexpr (std::is_same<T, float>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else if constexpr (std::is_same<T, double>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
  } else {
    static_assert(sizeof(T) < 0, "Unknown TMA Format!");
  }
}

PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
  // Get pointer to cuTensorMapEncodeTiled
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
#if (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 5)
  cudaGetDriverEntryPointByVersion(
      "cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status);
#else
  cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, cudaEnableDefault, &driver_status);
#endif

  if (driver_status != cudaDriverEntryPointSuccess) {
    throw std::runtime_error("driver_status != cudaDriverEntryPointSuccess");
  }

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(cuTensorMapEncodeTiled_ptr);
}

template <typename data_type>
CUtensorMap make_2d_tma_copy_desc(
    data_type* global_address,
    uint64_t gmem_dim[2],
    uint64_t stride_in_bytes,
    uint32_t smem_dim[2],
    CUtensorMapSwizzle swizzle_type,
    PFN_cuTensorMapEncodeTiled encode_func = nullptr) {
  CUtensorMap tensor_map{};
  constexpr uint32_t rank = 2;
  uint64_t global_stride[rank - 1] = {stride_in_bytes};
  uint32_t elem_strides[rank] = {1, 1};

  if (encode_func == nullptr) {
    encode_func = get_cuTensorMapEncodeTiled();
  }

  CUresult res = encode_func(
      &tensor_map,
      get_CUtensorMapDataType<typename std::remove_cv<data_type>::type>(),
      rank,
      global_address,
      gmem_dim,
      global_stride,
      smem_dim,
      elem_strides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle_type,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (int(res) == 1) {
    std::cout << "check 0: " << int(res) << std::endl;
    std::cout << gmem_dim[0] << "\t" << gmem_dim[1] << std::endl;
  }
  return tensor_map;
}

__device__ uint64_t mbarrier_arrive_1_expect_tx_cta(void* smem_ptr, uint32_t tx_count) {
  uint64_t state;
  asm("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2; // 8. "
      : "=l"(state)
      : "r"(static_cast<uint32_t>(cute::cast_smem_ptr_to_uint(smem_ptr))), "r"(tx_count)
      : "memory");
  return state;
}

}  // namespace tensorrt_llm::kernels::fp8_blockscale_gemm
