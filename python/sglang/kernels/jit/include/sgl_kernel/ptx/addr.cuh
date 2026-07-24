/// Extracted reusable utilities from kimi_k3/comm/gemm_ar.cuh.

#pragma once

#include <sgl_kernel/utils.cuh>

#include <cuda_runtime.h>
#include <cstdint>

// ================= common/ptx/addr.cuh =================
// Generic→shared address conversion. Inline-PTX `.shared` instructions take a
// 32-bit byte offset in the shared address window, not a generic 64-bit
// pointer. Use these to convert.
//
// PTX (PTX ISA 9.2 §10.4): `cvta.to.shared.u64 dst, src` (or `.u32`).


namespace ptx {

// Generic ptr → 32-bit `.shared` address. Equivalent to
// `__cvta_generic_to_shared(p)` but explicit so callers don't have to remember
// the builtin name.
template <typename T>
static SGL_DEVICE uint32_t to_shared(T* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}


// Local `.shared` byte offset → cluster-mapped `.shared::cluster` byte offset
// targeting CTA `cta_rank`'s smem at the same offset. This is the principled
// way to construct cross-CTA smem addresses for `.shared::cluster`-qualified
// instructions (mbar arrives, peer smem stores, TMA `cta_group::2` bar).
//
// PTX ISA 9.2 §9.7.12.15: `mapa.shared::cluster.u32 dst, src, cta_rank;`
// Replaces the older bit-twiddle idiom `addr & 0xFEFFFFFF` (= clear bit 24,
// the cluster-CTA-rank bit on `__cvta_generic_to_shared` outputs), which
// only happens to work for cta_rank=0 in a 2-CTA cluster.
static SGL_DEVICE uint32_t mapa_shared_cluster(
        uint32_t local_addr, uint32_t cta_rank) {
    uint32_t mapped;
    asm("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(mapped) : "r"(local_addr), "r"(cta_rank));
    return mapped;
}

}  // namespace ptx
