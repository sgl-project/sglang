/// \file ptx/cp_async.cuh
/// \brief Non-bulk cp.async wrappers for global-to-shared copies.

#pragma once

#include <sgl_kernel/ptx/addr.cuh>
#include <sgl_kernel/utils.cuh>

namespace ptx {

// Copy one 16-byte cache-global segment from global memory to shared memory.
// Both pointers must be 16-byte aligned.
static SGL_DEVICE void cp_async_cg_16b(void* smem_dst, const void* gmem_src) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(to_shared(smem_dst)), "l"(gmem_src)
        : "memory");
}

// Close the current per-thread cp.async group.
static SGL_DEVICE void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;");
}

// Wait until at most N committed cp.async groups remain pending.
template <int N>
static SGL_DEVICE void cp_async_wait_group() {
    static_assert(N >= 0 && N <= 7, "cp.async wait-group count must be in [0, 7]");
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}

static SGL_DEVICE void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;" ::: "memory");
}

}  // namespace ptx
