#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.hpp>

#include "common.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

struct SM100ArchSpec {
    static constexpr int smem_capacity = 232448;

    static std::vector<int> get_block_m_candidates(const KernelType& kernel_type, const cute::UMMA::Major& major_a, const int& m) {
        std::vector<int> candidates{128, 256};
        if ((kernel_type == KernelType::Kernel1D1D or kernel_type == KernelType::KernelNoSF) and major_a == cute::UMMA::Major::K) {
            // NOTES: `block_m = 32/64` is smaller than `LAYOUT_AD_M`, should be careful in handling this
            if (m <= 32) candidates.push_back(32);
            if (m <= 64) candidates.push_back(64);
        }
        return candidates;
    }

    static std::vector<int> get_block_n_candidates(const KernelType& kernel_type, const at::ScalarType& cd_dtype) {
        // 16 is for better SM usage
        // Stride 32 is due to low-performance swizzle-16/32B
        std::vector<int> candidates = {16};
        for (int i = 32; i <= 256; i += 32)
            candidates.push_back(i);
        return candidates;
    }

    static int get_ab_load_block_m(const MulticastConfig& config, const int& block_m) {
        return block_m / (config.is_multicast_on_a ? config.num_multicast : 1);
    }

    static int get_ab_load_block_n(const MulticastConfig& config, const int& block_n) {
        return block_n / (config.is_multicast_on_a ? 1 : config.num_multicast);
    }

    static int get_cd_store_block_m(const int& block_m) {
        constexpr int layout_ad_m = 128;
        return std::min(block_m, layout_ad_m);
    }

    static int get_cd_store_block_n(const int& block_n) {
        return block_n;
    }

    static bool enable_cd_swizzle(const at::ScalarType& cd_dtype) {
        return true;
    }

    static std::pair<int, int> get_sf_uttcp_aligned_block_sizes(
        const int& block_m, const int& block_n, const MmaKind& mma_kind) {
        constexpr int num_utccp_aligned_elems = 128;
        switch (mma_kind) {
            case MmaKind::BF16:     return {0, 0};
            case MmaKind::MXFP8FP4: return {align(block_m, num_utccp_aligned_elems), align(block_n, num_utccp_aligned_elems)};
            default: DG_HOST_UNREACHABLE("Unknown dtype");
        }
    }

    static bool is_block_size_legal(const KernelType& kernel_type,
                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                    const MmaKind& mma_kind, const at::ScalarType& cd_dtype,
                                    const int& m, const int& n, const int& k,
                                    const int& block_m, const int& block_n, const int& block_k) {
        // Layout A/D does not support `block_n % 16 != 0`
        if (block_n % 16 != 0)
            return false;

        // Performance is lower with 1D1D and `block_m == 256`
        if (kernel_type == KernelType::Kernel1D1D and major_b == cute::UMMA::Major::K and block_m > 128)
            return false;

        // For small K, fewer store blocks improve store/compute overlap and reduce epilogue bottleneck
        if (k <= 256 and (block_n > 128 or block_m > 128))
            return false;

        // Check tensor memory validity
        int sf_block_m = 0, sf_block_n = 0;
        if (kernel_type == KernelType::Kernel1D1D) {
            const auto& [sf_block_m_, sf_block_n_] = get_sf_uttcp_aligned_block_sizes(block_m, block_n, mma_kind);
            sf_block_m = sf_block_m_, sf_block_n = sf_block_n_;
        }
        if (((2 * block_n) + (sf_block_m / 32) + (sf_block_n / 32)) > 512)
            return false;

        // NOTES: when B is MN-major, we restrict `block_n` to multiples of 64,
        // since TMA performance degrades when `swizzle_b <= 32B` (i.e., when `block_ns % 64 != 0`), even with 3D TMA
        return major_b == cute::UMMA::Major::K or (block_n * get_element_size(mma_kind)) % 64 == 0;
    }

    static bool is_num_stages_legal(const MmaKind& mma_kind, const at::ScalarType& cd_dtype,
                                    const int& num_stages,
                                    const int& block_m, const int& block_n, const int& block_k) {
        return true;
    }

    static std::pair<bool, bool> get_multicast_legality(const GemmType& gemm_type, const int& num_groups,
                                                        const int& m, const int& n, const int& block_m, const int& block_n,
                                                        const int& num_sms) {
        // TODO: support other layouts
        return {
            false,
            is_multicast_legal(m, block_m, 2, num_sms, true) and (gemm_type == GemmType::Normal or gemm_type == GemmType::KGroupedContiguous
                                                                  or (gemm_type == GemmType::Batched and num_groups <= 32)),
        };
    }

    static ThreadConfig get_thread_config(const KernelType& kernel_type,
                                          const int& block_m, const int& block_n) {
        return ThreadConfig::sm100(128, 128);
    }

    static int get_smem_cd_size(const KernelType& kernel_type,
                                const int& block_m, const int& block_n,
                                const int& swizzle_cd_mode,
                                const at::ScalarType& cd_dtype) {
        constexpr static int layout_ad_m = 128;
        return std::min(block_m, layout_ad_m) * swizzle_cd_mode * 2;
    }

    static std::pair<int, int> get_sf_smem_size_per_stage(const KernelType& kernel_type,
                                                          const int& block_m, const int& block_n, const int& block_k,
                                                          const MmaKind& mma_kind, const at::ScalarType& cd_dtype) {
        if (mma_kind == MmaKind::BF16)
            return {0, 0};

        int smem_sfa_per_stage = 0;
        int smem_sfb_per_stage = 0;
        if (kernel_type == KernelType::Kernel1D1D) {
            const auto [sf_block_m, sf_block_n] = get_sf_uttcp_aligned_block_sizes(block_m, block_n, mma_kind);
            smem_sfa_per_stage = sf_block_m * 4;
            smem_sfb_per_stage = sf_block_n * 4;
        } else {
            smem_sfa_per_stage = block_m * 4;
            smem_sfb_per_stage = 0;
        }
        return {smem_sfa_per_stage, smem_sfb_per_stage};
    }

    static int get_extra_sfb_smem_size(const int& m, const int& n, const int& k,
                                       const int& block_m, const int& block_n, const int& block_k) {
        return 0;
    }

    static int get_barrier_smem_size(const int& num_stages) {
        // TODO: remove SF barriers for BF16 GEMMs
        // TMA full/empty barriers, with-SF full barriers, tensor memory full/empty barriers
        // NOTES: some shapes may only have 1 epilogue stage, but we still allocate space for 2 stages
        // NOTES: the last barrier is for tensor core utilization control
        return num_stages * 8 * 3 + 2 * 8 * 2 + 8;
    }

    static int get_tmem_ptr_smem_size() {
        return 4;
    }

    static int get_tensormap_smem_size(const GemmType& gemm_type) {
        return 0;
    }
};

} // namespace deep_gemm
