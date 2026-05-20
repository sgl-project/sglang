/**
 * SYCL Utility Headers for SGLang JIT Kernels
 * 
 * This file provides SYCL equivalents for common CUDA primitives used in SGLang kernels.
 * It bridges the gap between CUDA-style kernel code and SYCL programming model.
 */

#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <type_traits>

namespace sgl {
namespace sycl {

// ============================================================================
// Type Aliases matching SGLang conventions
// ============================================================================

using fp32_t = float;
using fp16_t = ::sycl::half;
using bf16_t = ::sycl::ext::oneapi::bfloat16;
using int8_t = std::int8_t;
using int64_t = std::int64_t;

// ============================================================================
// Thread/Block Index Helpers (CUDA-style API)
// ============================================================================

template <int Dim = 0>
inline size_t threadIdx(const ::sycl::nd_item<3>& item) {
    static_assert(Dim >= 0 && Dim < 3, "threadIdx Dim must be 0, 1, or 2");
    if constexpr (Dim == 0) return item.get_local_id(2);  // x
    else if constexpr (Dim == 1) return item.get_local_id(1);  // y
    else if constexpr (Dim == 2) return item.get_local_id(0);  // z
}

template <int Dim = 0>
inline size_t blockIdx(const ::sycl::nd_item<3>& item) {
    if constexpr (Dim == 0) return item.get_group(2);  // x
    else if constexpr (Dim == 1) return item.get_group(1);  // y
    else if constexpr (Dim == 2) return item.get_group(0);  // z
}

template <int Dim = 0>
inline size_t blockDim(const ::sycl::nd_item<3>& item) {
    if constexpr (Dim == 0) return item.get_local_range(2);  // x
    else if constexpr (Dim == 1) return item.get_local_range(1);  // y
    else if constexpr (Dim == 2) return item.get_local_range(0);  // z
}

template <int Dim = 0>
inline size_t gridDim(const ::sycl::nd_item<3>& item) {
    if constexpr (Dim == 0) return item.get_group_range(2);  // x
    else if constexpr (Dim == 1) return item.get_group_range(1);  // y
    else if constexpr (Dim == 2) return item.get_group_range(0);  // z
}

// ============================================================================
// Synchronization Primitives
// ============================================================================

inline void syncthreads(const ::sycl::nd_item<3>& item) {
    item.barrier(::sycl::access::fence_space::local_space);
}

inline void syncwarp(const ::sycl::nd_item<3>& item) {
    // SYCL sub-group barrier
    auto sg = item.get_sub_group();
    ::sycl::group_barrier(sg);
}

// ============================================================================
// Warp/Sub-group Shuffle Operations
// ============================================================================

template <typename T>
inline T shfl_xor(const ::sycl::nd_item<3>& item, T value, int mask) {
    auto sg = item.get_sub_group();
    return ::sycl::permute_group_by_xor(sg, value, mask);
}

template <typename T>
inline T shfl_down(const ::sycl::nd_item<3>& item, T value, int delta) {
    auto sg = item.get_sub_group();
    return ::sycl::shift_group_left(sg, value, delta);
}

template <typename T>
inline T shfl_up(const ::sycl::nd_item<3>& item, T value, int delta) {
    auto sg = item.get_sub_group();
    return ::sycl::shift_group_right(sg, value, delta);
}

// ============================================================================
// Warp-level Reductions
// ============================================================================

template <typename T>
inline T warp_reduce_sum(const ::sycl::nd_item<3>& item, T value) {
    auto sg = item.get_sub_group();
    return ::sycl::reduce_over_group(sg, value, ::sycl::plus<T>());
}

template <typename T>
inline T warp_reduce_max(const ::sycl::nd_item<3>& item, T value) {
    auto sg = item.get_sub_group();
    return ::sycl::reduce_over_group(sg, value, ::sycl::maximum<T>());
}

template <typename T>
inline T warp_reduce_min(const ::sycl::nd_item<3>& item, T value) {
    auto sg = item.get_sub_group();
    return ::sycl::reduce_over_group(sg, value, ::sycl::minimum<T>());
}

// ============================================================================
// Block-level Reductions
// ============================================================================

template <typename T>
inline T block_reduce_sum(const ::sycl::nd_item<3>& item, T value, T* shared_mem) {
    auto sg = item.get_sub_group();
    size_t sg_size = sg.get_local_linear_range();

    // First, reduce within warp
    value = warp_reduce_sum(item, value);
    
    // Write warp results to shared memory
    size_t lid = threadIdx<0>(item);
    size_t warp_id = lid / sg_size;
    size_t lane_id = lid % sg_size;
    
    if (lane_id == 0) {
        shared_mem[warp_id] = value;
    }
    
    syncthreads(item);
    
    // Final reduction across warps
    size_t num_warps = (blockDim<0>(item) + sg_size - 1) / sg_size;
    
    if (lid < num_warps) {
        value = shared_mem[lid];
    } else {
        value = T(0);
    }
    
    value = warp_reduce_sum(item, value);
    
    return value;
}

// ============================================================================
// Math Functions
// ============================================================================

template <typename T>
inline T rsqrt(T x) {
    return ::sycl::rsqrt(x);
}

template <typename T>
inline T exp(T x) {
    return ::sycl::exp(x);
}

template <typename T>
inline T log(T x) {
    return ::sycl::log(x);
}

template <typename T>
inline T sqrt(T x) {
    return ::sycl::sqrt(x);
}

template <typename T>
inline T tanh(T x) {
    return ::sycl::tanh(x);
}

// ============================================================================
// Memory Access Helpers
// ============================================================================

template <typename T>
inline T ldg(const T* ptr) {
    // Load from global memory (similar to __ldg in CUDA)
    return *ptr;
}

// ============================================================================
// Constants
// ============================================================================

constexpr int kWarpSize = 32;  // Standard sub-group size

} // namespace sycl
} // namespace sgl
