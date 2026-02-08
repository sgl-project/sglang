#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.hpp>

#include "common.hpp"

namespace deep_gemm {

struct SM90ArchSpec {
    static constexpr int smem_capacity = 232448;
    
    static std::vector<int> get_block_m_candidates(const KernelType& kernel_type, const cute::UMMA::Major& major_a, const int& m) {
        std::vector<int> candidates{64, 128, 256};
        if ((kernel_type == KernelType::Kernel1D2D or kernel_type == KernelType::KernelNoSF) and major_a == cute::UMMA::Major::K) {
            // NOTES: `block_m = 16/32` is smaller than MMA M size, should be careful in handling this
            if (m <= 16) candidates.push_back(16);
            if (m <= 32) candidates.push_back(32);
        }
        return candidates;
    }

    static std::vector<int> get_block_n_candidates(const KernelType& kernel_type, const at::ScalarType& cd_dtype) {
        int start = 16;

        // Avoid bank conflicts for 1D1D kernel FP32 output
        std::vector<int> candidates;
        if (kernel_type == KernelType::Kernel1D1D and cd_dtype == torch::kFloat) {
            candidates.push_back(16);
            start = 24;
        }

        // Push the strided options
        for (int i = start; i <= 256; i += 16)
            candidates.push_back(i);
        return candidates;
    }

    static int get_ab_load_block_m(const MulticastConfig& multicast_config, const int& block_m) {
        return block_m;
    }

    static int get_ab_load_block_n(const MulticastConfig& multicast_config, const int& block_n) {
        return block_n;
    }

    static int get_cd_store_block_m(const int& block_m, const bool& single_warpgroup_sync = false) {
        constexpr int wgmma_m = 64;
        return single_warpgroup_sync ? wgmma_m : block_m;
    }

    static int get_cd_store_block_n(const int& block_n) {
        return block_n;
    }

    static bool enable_cd_swizzle(const at::ScalarType& cd_dtype) {
        return cd_dtype != torch::kFloat;
    }

    static bool is_block_size_legal(const KernelType& kernel_type,
                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                    const MmaKind& mma_kind, const at::ScalarType& cd_dtype,
                                    const int& m, const int& n, const int& k,
                                    const int& block_m, const int& block_n, const int& block_k) {
        // SM90 FP32 output does not support `block_m == 256`
        if (cd_dtype == at::kFloat and block_m == 256)
            return false;

        // Avoid large C/D shared memory for FP32 output
        // Ensure `num_stages >= 4` (for 1D1D Kernel), `num_stages >= 3` (for No SF kernel)
        if (block_n > 128 and cd_dtype == torch::kFloat) {
            if (kernel_type == KernelType::Kernel1D1D and block_n > 152)
                return false;
            if (kernel_type == KernelType::KernelNoSF and block_n > 200)
                return false;
        }

        // When B is N Major, use swizzle 128B for better performance; only affects SM90 BF16 GEMM
        if (major_b == cute::UMMA::Major::MN and block_n >= 128 and block_n % 64 != 0)
            return false;

        // Too many scaling factors in a single block: `block_n > block_k and std::gcd(block_n, block_k) != block_n - block_k`
        // Or too many register spills
        if (block_n > 128 and kernel_type == KernelType::Kernel1D2D and (block_n != 144 and block_n != 160 and block_n != 192))
            return false;

        // The block sizes cannot be too large (for enough registers), so at least one dim less than 128
        return block_m <= 128 or block_n <= 128;
    }

    static bool is_num_stages_legal(const MmaKind& mma_kind, const at::ScalarType& cd_dtype,
                                    const int& num_stages,
                                    const int& block_m, const int& block_n, const int& block_k) {
        // Unrolling both stages and `num_former_iters` will cause large code size
        if (mma_kind == MmaKind::MXFP8FP4 and block_k % block_n != 0 and block_k / std::gcd(block_n, block_k) <= 4)
            return num_stages <= 4;
        return true;
    }

    static std::pair<bool, bool> get_multicast_legality(const GemmType& gemm_type, const int& num_groups,
                                                        const int& m, const int& n, const int& block_m, const int& block_n,
                                                        const int& num_sms) {
        // Disable multicast when the number of k-groups is large (a heuristic)
        if (gemm_type == GemmType::KGroupedContiguous and num_groups > 4)
            return {false, false};

        if (gemm_type == GemmType::Batched)
            return {false, false};

        return {
            is_multicast_legal(n, block_n, 2, num_sms, gemm_type == GemmType::MGroupedMasked),
            // For masked GEMM layout, divisibility on N is also required as we must ensure the total number of blocks is even
            is_multicast_legal(m, block_m, 2, num_sms, false)
                and (gemm_type != GemmType::MGroupedMasked or is_multicast_legal(n, block_n, 2, num_sms, true))
        };
    }

    static ThreadConfig get_thread_config(const KernelType& kernel_type,
                                          const int& block_m, const int& block_n) {
        return ThreadConfig::sm90(128, (block_m <= 64 ? 1 : 2) * 128);
    }

    static int get_smem_cd_size(const KernelType& kernel_type,
                                const int& block_m, const int& block_n,
                                const int& swizzle_cd_mode, const at::ScalarType& cd_dtype) {
        // NOTES: 1024 is for TMA swizzling alignment requirement
        return align(block_m * block_n * static_cast<int>(c10::elementSize(cd_dtype)), 1024);
    }

    static std::pair<int, int> get_sf_smem_size_per_stage(const KernelType& kernel_type,
                                                          const int& block_m, const int& block_n, const int& block_k,
                                                          const MmaKind& mma_kind, const at::ScalarType& cd_dtype) {
        if (mma_kind == MmaKind::BF16)
            return {0, 0};

        // NOTES: 128 is for 2D TMA alignment requirement
        int smem_sfa_per_stage = align(block_m * static_cast<int>(sizeof(float)), 128);
        int smem_sfb_per_stage = 0;
        if (kernel_type == KernelType::Kernel1D1D)
            smem_sfb_per_stage = align(block_n * 4, 128);
        return {smem_sfa_per_stage, smem_sfb_per_stage};
    }

    static int get_extra_sfb_smem_size(const int& m, const int& n, const int& k,
                                       const int& block_m, const int& block_n, const int& block_k) {
        const auto& use_uniform_sfb = block_k % block_n == 0 ? 1 : 2;
        return align<int>(ceil_div(k, block_k) * static_cast<int>(sizeof(float)) * use_uniform_sfb, 8);
    }

    static int get_barrier_smem_size(const int& num_stages) {
        return num_stages * 8 * 2;
    }

    static int get_tmem_ptr_smem_size() {
        return 0;
    }

    static int get_tensormap_smem_size(const GemmType& gemm_type) {
        return gemm_type == GemmType::KGroupedContiguous ? 4 * static_cast<int>(sizeof(CUtensorMap)) : 0;
    }
};

} // namespace deep_gemm
