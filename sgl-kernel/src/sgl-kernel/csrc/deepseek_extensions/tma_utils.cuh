#pragma once

#include <cassert>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda/barrier>

#include "utils.cuh"

namespace deep_gemm {

template <class T>
constexpr CUtensorMapDataType get_CUtensorMapDataType() {
    if constexpr (std::is_same<T, uint8_t>::value) {
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
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    }  else if constexpr (std::is_same<T, double>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
    }
}

PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    // Get pointer to `cuTensorMapEncodeTiled`
    cudaDriverEntryPointQueryResult driver_status;
    void* cuTensorMapEncodeTiled_ptr = nullptr;

#if CUDA_VERSION >= 12050
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000,
                                     cudaEnableDefault, &driver_status);
#else
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr,
                            cudaEnableDefault, &driver_status);
#endif

    if (driver_status != cudaDriverEntryPointSuccess)
        throw std::runtime_error("driver_status != cudaDriverEntryPointSuccess");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(cuTensorMapEncodeTiled_ptr);
}

template <typename T>
CUtensorMap make_2d_tma_copy_desc(T* global_address, uint64_t gmem_dim[2],
                                  uint64_t stride_in_bytes, uint32_t smem_dim[2],
                                  CUtensorMapSwizzle swizzle_type,
                                  PFN_cuTensorMapEncodeTiled encode_func = nullptr) {
    CUtensorMap tensor_map{};
    constexpr uint32_t rank = 2;
    uint64_t global_stride[rank - 1] = {stride_in_bytes};
    uint32_t elem_strides[rank] = {1, 1};

    if (encode_func == nullptr)
        encode_func = get_cuTensorMapEncodeTiled();

    auto result = encode_func(
            &tensor_map, get_CUtensorMapDataType<typename std::remove_cv<T>::type>(), rank,
            global_address, gmem_dim, global_stride, smem_dim, elem_strides,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_type,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    DG_HOST_ASSERT(result == CUDA_SUCCESS);
    return tensor_map;
}

template <uint32_t kNumTMAMulticast = 1>
__device__ __forceinline__ void
tma_copy(void const* desc_ptr, uint64_t* barrier_ptr, void* smem_ptr,
         int32_t const& crd_0, int32_t const& crd_1) {
    constexpr auto cache_hint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
    if constexpr (kNumTMAMulticast == 1) {
        cute::SM90_TMA_LOAD_2D::copy(desc_ptr, barrier_ptr, cache_hint, smem_ptr, crd_0, crd_1);
    } else if (cute::block_rank_in_cluster() == 0) {
        cute::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, barrier_ptr, (1 << kNumTMAMulticast) - 1, cache_hint, smem_ptr, crd_0, crd_1);
    }
}

}  // namespace deep_gemm
