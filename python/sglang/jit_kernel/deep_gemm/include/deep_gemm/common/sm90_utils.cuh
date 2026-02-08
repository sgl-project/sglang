#pragma once

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>
#include <cute/arch/mma_sm100_desc.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>
#include <deep_gemm/common/tma_utils.cuh>

namespace deep_gemm::sm90 {

template <int N_, typename MMA>
struct FP8MMA {

    template <size_t ...Idx>
    __forceinline__ __device__ static void call_fma_impl(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d, cute::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(desc_a, desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }

    __forceinline__ __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        call_fma_impl(desc_a, desc_b, d, scale_d, cute::make_index_sequence<N_/2>{});
    }

    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

template <int N>
struct FP8MMASelector {

    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        if constexpr (N == 8) return MMA_64x8x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 16) return MMA_64x16x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 24) return MMA_64x24x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 32) return MMA_64x32x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 40) return MMA_64x40x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 48) return MMA_64x48x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 56) return MMA_64x56x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 64) return MMA_64x64x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 72) return MMA_64x72x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 80) return MMA_64x80x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 88) return MMA_64x88x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 96) return MMA_64x96x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 104) return MMA_64x104x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 112) return MMA_64x112x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 120) return MMA_64x120x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 128) return MMA_64x128x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 136) return MMA_64x136x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 144) return MMA_64x144x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 152) return MMA_64x152x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 160) return MMA_64x160x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 168) return MMA_64x168x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 176) return MMA_64x176x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 184) return MMA_64x184x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 192) return MMA_64x192x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 200) return MMA_64x200x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 208) return MMA_64x208x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 216) return MMA_64x216x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 224) return MMA_64x224x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 232) return MMA_64x232x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 240) return MMA_64x240x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 248) return MMA_64x248x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 256) return MMA_64x256x32_F32E4M3E4M3_SS_TN();
    }

    static constexpr auto select_type() {
        return FP8MMA<N, decltype(select_mma())>();
    }

    using type = decltype(select_type());
};

template <int N_, typename MMA>
struct BF16MMA {

    template <size_t ...Idx>
    __forceinline__ __device__ static void call_fma_impl(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d, cute::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(desc_a, desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }

    __forceinline__ __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        call_fma_impl(desc_a, desc_b, d, scale_d, cute::make_index_sequence<N_/2>{});
    }

    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 16;
    static constexpr int kNumAccum = M * N / 128;
};

template <cute::UMMA::Major kMajor>
constexpr cute::SM90::GMMA::Major to_sm90_major() {
    DG_STATIC_ASSERT(kMajor == cute::UMMA::Major::K or kMajor == cute::UMMA::Major::MN, "Invalid major-ness");
    return kMajor == cute::UMMA::Major::K ? cute::SM90::GMMA::Major::K : cute::SM90::GMMA::Major::MN;
}

template <int N,
          cute::UMMA::Major kMajorA = cute::UMMA::Major::K,
          cute::UMMA::Major kMajorB = cute::UMMA::Major::K>
struct BF16MMASelector {

    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        constexpr auto kGMMAMajorA = to_sm90_major<kMajorA>();
        constexpr auto kGMMAMajorB = to_sm90_major<kMajorB>();
        if constexpr (N == 8) return MMA_64x8x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 16) return MMA_64x16x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 24) return MMA_64x24x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 32) return MMA_64x32x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 40) return MMA_64x40x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 48) return MMA_64x48x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 56) return MMA_64x56x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 64) return MMA_64x64x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 72) return MMA_64x72x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 80) return MMA_64x80x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 88) return MMA_64x88x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 96) return MMA_64x96x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 104) return MMA_64x104x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 112) return MMA_64x112x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 120) return MMA_64x120x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 128) return MMA_64x128x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 136) return MMA_64x136x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 144) return MMA_64x144x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 152) return MMA_64x152x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 160) return MMA_64x160x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 168) return MMA_64x168x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 176) return MMA_64x176x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 184) return MMA_64x184x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 192) return MMA_64x192x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 200) return MMA_64x200x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 208) return MMA_64x208x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 216) return MMA_64x216x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 224) return MMA_64x224x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 232) return MMA_64x232x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 240) return MMA_64x240x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 248) return MMA_64x248x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 256) return MMA_64x256x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
    }

    static constexpr auto select_type() {
        return BF16MMA<N, decltype(select_mma())>();
    }

    using type = decltype(select_type());
};

template <int N_, typename MMA>
struct TF32MMARS {

    template <size_t ...Idx>
    __forceinline__ __device__ static void call_fma_impl(uint32_t* a, uint64_t const& desc_b, float* d, bool scale_d, cute::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(a[0], a[1], a[2], a[3], desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }

    __forceinline__ __device__ static void wgmma(float* a, uint64_t const& desc_b, float* d, bool scale_d) {
        call_fma_impl(reinterpret_cast<uint32_t*>(a), desc_b, d, scale_d, cute::make_index_sequence<N_/2>{});
    }

    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 8;
    static constexpr int kNumAccum = M * N / 128;
};

template <int N, bool kUseRS = true>
struct TF32MMASelector {

    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        if constexpr (kUseRS) {
            if constexpr (N == 8) return MMA_64x8x8_F32TF32TF32_RS_TN();
            if constexpr (N == 16) return MMA_64x16x8_F32TF32TF32_RS_TN();
            if constexpr (N == 32) return MMA_64x32x8_F32TF32TF32_RS_TN();
            if constexpr (N == 64) return MMA_64x64x8_F32TF32TF32_RS_TN();
            if constexpr (N == 128) return MMA_64x128x8_F32TF32TF32_RS_TN();
            if constexpr (N == 256) return MMA_64x256x8_F32TF32TF32_RS_TN();
            DG_STATIC_ASSERT(N == 8 or N == 16 or N == 32 or N == 64 or N == 128 or N == 256, "Invalid N");
        }
    }

    static constexpr auto select_type() {
        if constexpr (kUseRS) {
            return TF32MMARS<N, decltype(select_mma())>();
        } else {
            DG_STATIC_ASSERT(kUseRS, "SS mode is not supported for TF32MMASelector for now");
        }
    }

    using type = decltype(select_type());
};

template <typename dtype_t>
struct SM90_U32x2_STSM_N {
    __device__ __forceinline__ static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src[0]), "r"(src[1]));
    }
};

struct SM90_U32x2_LDSM_N {
    __device__ __forceinline__ static void
    copy(uint32_t& dst_0, uint32_t& dst_1, void* smem_src) {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(dst_0), "=r"(dst_1)
                     : "l"(__cvta_generic_to_shared(smem_src)));
    }
};

struct SM90_U32x4_LDSM_N {
    __device__ __forceinline__ static void
    copy(uint32_t& dst_0, uint32_t& dst_1, uint32_t& dst_2, uint32_t& dst_3, void* smem_src) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(dst_0), "=r"(dst_1), "=r"(dst_2), "=r"(dst_3)
                     : "l"(__cvta_generic_to_shared(smem_src)));
    }
};

__forceinline__ __device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__forceinline__ __device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__forceinline__ __device__ void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

template <int N>
__forceinline__ __device__ void warpgroup_wait() {
    DG_STATIC_ASSERT(N >= 0 and N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

template <class PointerType>
__device__ cute::GmmaDescriptor make_smem_desc(PointerType smem_ptr, const int& layout_type,
                                               const int& leading_byte_offset = 0,
                                               const int& stride_byte_offset = 1024) {
    // NOTES: the default LBO and SBO are for K-major types
    cute::GmmaDescriptor desc;
    const auto& uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = layout_type;
    desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
    desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.bitfield.base_offset_ = 0;
    return desc;
}

template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, typename dtype_t>
constexpr uint32_t get_inner_block_atom_size() {
    return kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t);
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
__device__ __forceinline__
constexpr uint32_t get_gmma_desc_stride_k() {
    return kMajorMode == cute::UMMA::Major::K ? 1 : get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();
}

// ReSharper disable once CppNotAllPathsReturnValue
template <cute::UMMA::Major kMajorMode, uint32_t kSwizzleMode, typename dtype_t>
constexpr static cute::SM90::GMMA::LayoutType to_gmma_layout_type() {
    DG_STATIC_ASSERT(kSwizzleMode == 0 or kSwizzleMode == 16 or
                     kSwizzleMode == 32 or kSwizzleMode == 64 or
                     kSwizzleMode == 128, "Invalid swizzling mode");

    // Normal cases
    if constexpr (kSwizzleMode == 0)   return cute::SM90::GMMA::LayoutType::INTERLEAVE;
    if constexpr (kSwizzleMode == 16)  return cute::SM90::GMMA::LayoutType::INTERLEAVE;
    if constexpr (kSwizzleMode == 32)  return cute::SM90::GMMA::LayoutType::B32;
    if constexpr (kSwizzleMode == 64)  return cute::SM90::GMMA::LayoutType::B64;
    if constexpr (kSwizzleMode == 128) return cute::SM90::GMMA::LayoutType::B128;
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode, typename dtype_t>
__device__ __forceinline__
uint32_t advance_gmma_desc_lo(const uint32_t& base, const uint32_t& mn_idx, const uint32_t& k_idx, const uint32_t& offset = 0) {
    return base + (((offset + mn_idx * BLOCK_K + k_idx * get_gmma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>()) * static_cast<uint32_t>(sizeof(dtype_t))) >> 4u);
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode, typename dtype_t>
__device__ __forceinline__
cute::GmmaDescriptor make_gmma_desc(dtype_t* base_smem_ptr, uint32_t mn_idx, uint32_t k_idx) {
    const uint32_t stride_k = get_gmma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>();
    const auto& layout_type = to_gmma_layout_type<kMajorMode, kSwizzleMode, dtype_t>();
    constexpr uint32_t num_non_contiguous = 128 / 16;
    if constexpr (kMajorMode == cute::UMMA::Major::K) {
        // NOTES: for K-major layout, the swizzle must be 128B (also, atom index must be 0), as `BLOCK_K` is always 128
        DG_STATIC_ASSERT(kSwizzleMode == BLOCK_K * sizeof(dtype_t), "Unexpected value");

        // Atom size: 8 x `kSwizzleMode` (in bytes, on K)
        // {SBO, LBO} means the byte stride between atoms on {MN, K}
        // NOTES: on K, there is only 1 atom as asserted previously, so LBO can be 0
        const uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K * sizeof(dtype_t);
        const uint32_t leading_byte_offset = 0;
        return make_smem_desc(base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k, static_cast<uint32_t>(layout_type),
                              leading_byte_offset, stride_byte_offset);
    } else {
        constexpr uint32_t BLOCK_MN_ATOM = get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();

        // Must have no in-atom MN-idx
        // NOTES: no worries for the runtime assert, the `mn_idx` are constants at compilation time
        DG_DEVICE_ASSERT(mn_idx % BLOCK_MN_ATOM == 0);
        DG_STATIC_ASSERT(kSwizzleMode > 0, "Invalid swizzling");

        // Atom size: `kSwizzleMode` (in bytes, on MN) x 8
        // NOTES: `kSwizzleMode == 16` mean non-swizzling but interleaving
        // {SBO, LBO} means the byte stride between atoms on {K, MN} for swizzling
        // {SBO, LBO} means the byte stride between atoms on {MN, K} for non-swizzling
        uint32_t stride_byte_offset = num_non_contiguous * BLOCK_MN_ATOM * sizeof(dtype_t);
        uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype_t);
        if constexpr (kSwizzleMode == 16)
            swap(stride_byte_offset, leading_byte_offset);
        return make_smem_desc(base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k, static_cast<uint32_t>(layout_type),
                              leading_byte_offset, stride_byte_offset);
    }
}

} // namespace `deep_gemm::sm90`
