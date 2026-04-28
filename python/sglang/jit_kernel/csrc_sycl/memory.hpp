/**
 * SYCL Memory Utilities for SGLang JIT Kernels
 * 
 * This file provides memory access helpers including shared memory (local_accessor)
 * and global memory operations.
 */

#pragma once

#include <sycl/sycl.hpp>
#include "utils.hpp"

namespace sgl {
namespace sycl {

// ============================================================================
// Shared Memory / Local Accessor Helpers
// ============================================================================

/**
 * SharedMemory: SYCL equivalent of CUDA __shared__ memory
 * 
 * Usage in kernel:
 *   auto shared = SharedMemory<float, 256>(item);
 *   shared[threadIdx<0>(item)] = value;
 */
template <typename T, size_t Size>
class SharedMemory {
public:
    using accessor_t = ::sycl::local_accessor<T, 1>;
    
    SharedMemory(const ::sycl::nd_item<3>& item, accessor_t acc)
        : item_(item), acc_(acc) {}
    
    T& operator[](size_t idx) {
        return acc_[idx];
    }
    
    const T& operator[](size_t idx) const {
        return acc_[idx];
    }
    
    T* data() {
        return acc_.get_pointer();
    }
    
    const T* data() const {
        return acc_.get_pointer();
    }
    
    constexpr size_t size() const {
        return Size;
    }

private:
    const ::sycl::nd_item<3>& item_;
    accessor_t acc_;
};

// ============================================================================
// Global Memory Helpers
// ============================================================================

/**
 * Vectorized load from global memory
 */
template <typename T, int VecSize>
struct VectorizedLoad {
    using vec_t = ::sycl::vec<T, VecSize>;
    
    static vec_t load(const T* ptr) {
        return *reinterpret_cast<const vec_t*>(ptr);
    }
};

/**
 * Vectorized store to global memory
 */
template <typename T, int VecSize>
struct VectorizedStore {
    using vec_t = ::sycl::vec<T, VecSize>;
    
    static void store(T* ptr, const vec_t& value) {
        *reinterpret_cast<vec_t*>(ptr) = value;
    }
};

// ============================================================================
// Coalesced Memory Access Patterns
// ============================================================================

/**
 * Coalesced load: threads in a warp load consecutive elements
 */
template <typename T>
inline T coalesced_load(const T* base_ptr, const ::sycl::nd_item<3>& item) {
    size_t idx = blockIdx<0>(item) * blockDim<0>(item) + threadIdx<0>(item);
    return base_ptr[idx];
}

/**
 * Coalesced store: threads in a warp store to consecutive elements
 */
template <typename T>
inline void coalesced_store(T* base_ptr, T value, const ::sycl::nd_item<3>& item) {
    size_t idx = blockIdx<0>(item) * blockDim<0>(item) + threadIdx<0>(item);
    base_ptr[idx] = value;
}

// ============================================================================
// Strided Memory Access
// ============================================================================

/**
 * Strided load with configurable stride
 */
template <typename T>
inline T strided_load(const T* base_ptr, size_t base_idx, size_t stride, size_t offset) {
    return base_ptr[base_idx * stride + offset];
}

/**
 * Strided store with configurable stride
 */
template <typename T>
inline void strided_store(T* base_ptr, T value, size_t base_idx, size_t stride, size_t offset) {
    base_ptr[base_idx * stride + offset] = value;
}

// ============================================================================
// Pointer Arithmetic Helpers
// ============================================================================

namespace pointer {

template <typename T>
inline T* offset(void* ptr, size_t byte_offset) {
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + byte_offset);
}

template <typename T>
inline const T* offset(const void* ptr, size_t byte_offset) {
    return reinterpret_cast<const T*>(reinterpret_cast<const char*>(ptr) + byte_offset);
}

} // namespace pointer

// ============================================================================
// Memory Tile Abstraction (similar to sgl_kernel/tile.cuh)
// ============================================================================

template <typename T>
struct TileMemory {
    using value_type = T;
    
    /**
     * Load a tile of data from global memory
     * VecSize: number of elements to load per thread
     */
    template <int VecSize>
    static ::sycl::vec<T, VecSize> load_tile(
        const T* ptr,
        const ::sycl::nd_item<3>& item
    ) {
        size_t tid = threadIdx<0>(item);
        return VectorizedLoad<T, VecSize>::load(ptr + tid * VecSize);
    }
    
    /**
     * Store a tile of data to global memory
     */
    template <int VecSize>
    static void store_tile(
        T* ptr,
        const ::sycl::vec<T, VecSize>& data,
        const ::sycl::nd_item<3>& item
    ) {
        size_t tid = threadIdx<0>(item);
        VectorizedStore<T, VecSize>::store(ptr + tid * VecSize, data);
    }
};

} // namespace sycl
} // namespace sgl
