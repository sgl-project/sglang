#pragma once

#include <cutlass/arch/barrier.h>

#include <cute/arch/copy_sm100.hpp>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::ptx {

/// UMMA versions with relaxed assertions
struct SM100_MMA_F16BF16_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; \n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};

struct SM100_MMA_F16BF16_2x1SM_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p; \n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};

struct SM100_MMA_MXF8F6F4_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
            "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

struct SM100_MMA_MXF8F6F4_2x1SM_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
            "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

struct SM100_MMA_F8F6F4_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};

struct SM100_MMA_F8F6F4_2x1SM_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};

struct SM100_MMA_MXF4_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
               "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

// Stream A0.5: cta_group::2 (cluster) variant of kind::mxf4 for the
// mega_moe 2-CTA path. Mirrors `SM100_MMA_MXF8F6F4_2x1SM_SS` shape — the
// only differences vs the 1-CTA `SM100_MMA_MXF4_SS` above are the
// `cta_group::2` qualifier and the (caller-side) requirement that:
//   - operands are K-major (kind::mxf4 hardware restriction)
//   - smem A/B use the dense FP4 layout (`_ALIGN8B`, 2 nibbles/byte)
//   - SF TMEM address top-2 bits encode HALF-WORD offsets {0, 2} for
//     scale_vec::2X (use `(k_block * 2) << 30`, NOT `k_block << 30`)
struct SM100_MMA_MXF4_2x1SM_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
            "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
            "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
               "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

struct SM100_MMA_MXF4NVF4_2x1SM_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
            "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
            "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
               "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

struct SM100_MMA_F16BF16_WS_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p; \n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};

/// Tensor memory operations
template <uint32_t kNumValues>
CUTLASS_DEVICE void tmem_load_32dp32b(const uint32_t tmem_addr, uint32_t* values) {
    DG_STATIC_ASSERT(kNumValues == 4 or kNumValues == 8 or kNumValues == 16 or
                     kNumValues == 32 or kNumValues == 64 or kNumValues == 128,
                     "Invalid TMEM load width");
    using Loader = cute::conditional_t<kNumValues == 4, cute::SM100_TMEM_LOAD_32dp32b4x,
                   cute::conditional_t<kNumValues == 8, cute::SM100_TMEM_LOAD_32dp32b8x,
                   cute::conditional_t<kNumValues == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
                   cute::conditional_t<kNumValues == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                   cute::conditional_t<kNumValues == 64, cute::SM100_TMEM_LOAD_32dp32b64x,
                                                               cute::SM100_TMEM_LOAD_32dp32b128x>>>>>;
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        Loader::copy(tmem_addr, values[Is]...);
    }(cute::make_index_sequence<kNumValues>{});
}

CUTLASS_DEVICE void umma_arrive_no_elect(cutlass::arch::ClusterTransactionBarrier& barrier) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&barrier))));
}

CUTLASS_DEVICE void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

CUTLASS_DEVICE void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

} // namespace deep_gemm::ptx
