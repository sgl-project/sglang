/**
 * SYCL Atomic Operations for SGLang JIT Kernels
 * 
 * This file provides atomic operation wrappers using sycl::atomic_ref.
 */

#pragma once

#include <sycl/sycl.hpp>
#include "utils.hpp"

namespace sgl {
namespace sycl {

// ============================================================================
// Atomic Operations (CUDA atomicAdd/atomicMax/atomicMin equivalents)
// ============================================================================

template <typename T>
inline T atomic_add(T* addr, T value) {
    ::sycl::atomic_ref<T,
                       ::sycl::memory_order::relaxed,
                       ::sycl::memory_scope::device,
                       ::sycl::access::address_space::global_space> atomic_val(*addr);
    return atomic_val.fetch_add(value);
}

template <typename T>
inline T atomic_sub(T* addr, T value) {
    ::sycl::atomic_ref<T,
                       ::sycl::memory_order::relaxed,
                       ::sycl::memory_scope::device,
                       ::sycl::access::address_space::global_space> atomic_val(*addr);
    return atomic_val.fetch_sub(value);
}

template <typename T>
inline T atomic_max(T* addr, T value) {
    ::sycl::atomic_ref<T,
                       ::sycl::memory_order::relaxed,
                       ::sycl::memory_scope::device,
                       ::sycl::access::address_space::global_space> atomic_val(*addr);
    return atomic_val.fetch_max(value);
}

template <typename T>
inline T atomic_min(T* addr, T value) {
    ::sycl::atomic_ref<T,
                       ::sycl::memory_order::relaxed,
                       ::sycl::memory_scope::device,
                       ::sycl::access::address_space::global_space> atomic_val(*addr);
    return atomic_val.fetch_min(value);
}

template <typename T>
inline T atomic_and(T* addr, T value) {
    ::sycl::atomic_ref<T,
                       ::sycl::memory_order::relaxed,
                       ::sycl::memory_scope::device,
                       ::sycl::access::address_space::global_space> atomic_val(*addr);
    return atomic_val.fetch_and(value);
}

template <typename T>
inline T atomic_or(T* addr, T value) {
    ::sycl::atomic_ref<T,
                       ::sycl::memory_order::relaxed,
                       ::sycl::memory_scope::device,
                       ::sycl::access::address_space::global_space> atomic_val(*addr);
    return atomic_val.fetch_or(value);
}

template <typename T>
inline T atomic_xor(T* addr, T value) {
    ::sycl::atomic_ref<T,
                       ::sycl::memory_order::relaxed,
                       ::sycl::memory_scope::device,
                       ::sycl::access::address_space::global_space> atomic_val(*addr);
    return atomic_val.fetch_xor(value);
}

template <typename T>
inline T atomic_exchange(T* addr, T value) {
    ::sycl::atomic_ref<T,
                       ::sycl::memory_order::relaxed,
                       ::sycl::memory_scope::device,
                       ::sycl::access::address_space::global_space> atomic_val(*addr);
    return atomic_val.exchange(value);
}

template <typename T>
inline T atomic_cas(T* addr, T compare, T value) {
    ::sycl::atomic_ref<T,
                       ::sycl::memory_order::relaxed,
                       ::sycl::memory_scope::device,
                       ::sycl::access::address_space::global_space> atomic_val(*addr);
    atomic_val.compare_exchange_strong(compare, value);
    return compare;
}

// ============================================================================
// Shared Memory Atomics
// ============================================================================

template <typename T>
inline T atomic_add_local(T* addr, T value) {
    ::sycl::atomic_ref<T,
                       ::sycl::memory_order::relaxed,
                       ::sycl::memory_scope::work_group,
                       ::sycl::access::address_space::local_space> atomic_val(*addr);
    return atomic_val.fetch_add(value);
}

template <typename T>
inline T atomic_max_local(T* addr, T value) {
    ::sycl::atomic_ref<T,
                       ::sycl::memory_order::relaxed,
                       ::sycl::memory_scope::work_group,
                       ::sycl::access::address_space::local_space> atomic_val(*addr);
    return atomic_val.fetch_max(value);
}

template <typename T>
inline T atomic_min_local(T* addr, T value) {
    ::sycl::atomic_ref<T,
                       ::sycl::memory_order::relaxed,
                       ::sycl::memory_scope::work_group,
                       ::sycl::access::address_space::local_space> atomic_val(*addr);
    return atomic_val.fetch_min(value);
}

} // namespace sycl
} // namespace sgl
