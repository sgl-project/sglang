#pragma once

#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/packing.cuh>

namespace deep_gemm::tma {

template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, uint32_t kPackFactor, typename dtype_t>
constexpr uint32_t get_inner_block_atom_size() {
    return kSwizzleMode == 0 ? BLOCK_INNER / kPackFactor : kSwizzleMode / sizeof(dtype_t);
}

template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER,
          uint32_t kSwizzleMode,
          typename dtype_t, bool kIs3DTMA = false,
          uint64_t kCacheHint = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL)>
CUTLASS_DEVICE void
copy(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
     dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx,
     const uint32_t& num_tma_multicast = 1, const uint32_t& batch_idx = 0,
     const uint64_t& cache_hint = kCacheHint) {
    DG_STATIC_ASSERT(static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL) ==
                     static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL), "Invalid cache hint");
    constexpr uint32_t kPackFactor = get_smem_pack_factor<dtype_t>();
    // NOTES: for sub-byte packed types, the SMEM atom stride stays in
    //        bytes (`sizeof(dtype_t) == 1`), while the GMEM index and loop count run over logical elements
    DG_STATIC_ASSERT(kPackFactor == 1 or sizeof(dtype_t) == 1, "Packing expects a 1-byte storage type");
    constexpr uint32_t BLOCK_INNER_ATOM_STORAGE = get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, kPackFactor, dtype_t>();
    constexpr uint32_t BLOCK_INNER_ATOM_LOGICAL = BLOCK_INNER_ATOM_STORAGE * kPackFactor;
    DG_STATIC_ASSERT(BLOCK_INNER % BLOCK_INNER_ATOM_LOGICAL == 0, "TMA inner block must contain whole atoms");

    if constexpr (not kIs3DTMA) {
        if (num_tma_multicast == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i) {
                cute::SM90_TMA_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                             cache_hint,
                                             smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
                                             inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_idx);
            }
        } else {
            #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
                // 2-CTA function will send signals to the leader CTA only
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i) {
                    cute::SM100_TMA_2SM_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                      cache_hint,
                                                      smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
                                                      inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_idx);
                }
            #elif (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
                if (cute::block_rank_in_cluster() == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i) {
                        cute::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                               (1 << num_tma_multicast) - 1, cache_hint,
                                                               smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
                                                               inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_idx);
                    }
                }
            #endif
        }
    } else {
        if (num_tma_multicast == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i) {
                cute::SM90_TMA_LOAD_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                            static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                            smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
                                            inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_idx, batch_idx);
            }
        } else {
            #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
                // 2-CTA function will send signals to the leader CTA only
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i) {
                    cute::SM100_TMA_2SM_LOAD_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                      static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                                      smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
                                                      inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_idx, batch_idx);
                }
            #elif (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
                if (cute::block_rank_in_cluster() == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i) {
                        cute::SM90_TMA_LOAD_MULTICAST_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                               (1 << num_tma_multicast) - 1, static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                                               smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
                                                               inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_idx, batch_idx);
                    }
                }
            #endif
        }
    }
}

} // namespace deep_gemm::tma
