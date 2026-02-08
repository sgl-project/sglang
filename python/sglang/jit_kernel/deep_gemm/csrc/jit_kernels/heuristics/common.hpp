#pragma once

#include <deep_gemm/common/types.hpp>

#include "../../utils/math.hpp"
#include "../../utils/layout.hpp"
#include "../../utils/system.hpp"

namespace deep_gemm {

struct MulticastConfig {
    int num_multicast;
    bool is_multicast_on_a;

    MulticastConfig(const int& num_multicast, const bool& is_multicast_on_a):
        num_multicast(num_multicast), is_multicast_on_a(is_multicast_on_a) {
        DG_HOST_ASSERT(1 <= num_multicast and num_multicast <= 2);
    }
};

struct SharedMemoryConfig {
    int smem_size;
    int swizzle_a_mode;
    int swizzle_b_mode;
    int swizzle_cd_mode;
};

struct ThreadConfig {
    int num_threads;

    // SM90
    int num_tma_threads;
    int num_math_threads;

    // SM100
    int num_non_epilogue_threads;
    int num_epilogue_threads;

    static ThreadConfig sm90(const int& num_tma_threads,
                             const int& num_math_threads) {
        auto config = ThreadConfig();
        config.num_threads = num_tma_threads + num_math_threads;
        config.num_tma_threads = num_tma_threads;
        config.num_math_threads = num_math_threads;
        return config;
    }

    static ThreadConfig sm100(const int& num_non_epilogue_threads,
                              const int& num_epilogue_threads) {
        auto config = ThreadConfig();
        config.num_threads = num_non_epilogue_threads + num_epilogue_threads;
        config.num_non_epilogue_threads = num_non_epilogue_threads;
        config.num_epilogue_threads = num_epilogue_threads;
        return config;
    }
};

struct GemmConfig {
    // Templated configs
    GemmType gemm_type;
    KernelType kernel_type;
    MmaKind mma_kind;
    at::ScalarType a_dtype, b_dtype, cd_dtype;
    cute::UMMA::Major major_a;
    cute::UMMA::Major major_b;
    bool with_accumulation;
    int block_m, block_n, block_k;
    int num_stages, num_last_stages;

    // Templated device configs
    int num_sms;
    int tc_util;

    // Structured configs
    MulticastConfig multicast_config;
    SharedMemoryConfig smem_config;
    ThreadConfig thread_config;
};

static bool is_multicast_legal(const int& shape_dim, const int& block_dim,
                               const int& num_multicast, const int& num_sms,
                               const bool& require_divisible) {
    const bool& divisible = ceil_div(shape_dim, block_dim) % num_multicast == 0 or not require_divisible;
    return divisible and num_sms % num_multicast == 0;
}

template <typename size_type_t>
static int get_swizzle_mode(const int& block_size, const size_type_t& elem_size) {
    // `> 0` means interleaving
    // 16B actually means non-swizzling (but interleaving)
    for (const int& mode: {128, 64, 32, 16}) {
        if ((block_size * static_cast<int>(elem_size)) % mode == 0)
            return mode;
    }
    DG_HOST_UNREACHABLE("Unreachable");
}

template <typename ArchSpec>
static SharedMemoryConfig get_smem_config(const GemmType& gemm_type, const KernelType& kernel_type,
                                          const int& m, const int& n, const int& k,
                                          const int& block_m, const int& block_n, const int& block_k,
                                          const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                          const MmaKind& mma_kind, const at::ScalarType& cd_dtype,
                                          const int& num_stages, const MulticastConfig& multicast_config) {
    const int& ab_elem_size = static_cast<int>(get_element_size(mma_kind));
    const int& cd_elem_size = static_cast<int>(c10::elementSize(cd_dtype));

    const int& load_block_m = ArchSpec::get_ab_load_block_m(multicast_config, block_m);
    const int& load_block_n = ArchSpec::get_ab_load_block_n(multicast_config, block_n);
    const int& swizzle_a_mode = get_swizzle_mode(major_a == cute::UMMA::Major::K ? block_k : load_block_m, ab_elem_size);
    const int& swizzle_b_mode = get_swizzle_mode(major_b == cute::UMMA::Major::K ? block_k : load_block_n, ab_elem_size);
    const int& swizzle_cd_mode = ArchSpec::enable_cd_swizzle(cd_dtype) ? get_swizzle_mode(block_n, cd_elem_size) : 0;

    // Different archs have different epilogue pipelines
    const int& smem_cd = ArchSpec::get_smem_cd_size(kernel_type, block_m, block_n, swizzle_cd_mode, cd_dtype);

    // A/B shared memory
    const int& smem_a_per_stage = load_block_m * block_k * ab_elem_size;
    const int& smem_b_per_stage = load_block_n * block_k * ab_elem_size;

    // SF shared memory
    const auto& [smem_sfa_per_stage, smem_sfb_per_stage] =
        ArchSpec::get_sf_smem_size_per_stage(kernel_type, block_m, block_n, block_k, mma_kind, cd_dtype);
    const int& smem_extra_sfb = ArchSpec::get_extra_sfb_smem_size(m, n, k, block_m, block_n, block_k);

    // M-barriers and tensor memory pointers
    const int& smem_barrier = ArchSpec::get_barrier_smem_size(num_stages);
    const int& smem_tmem_ptr = ArchSpec::get_tmem_ptr_smem_size();
    const int& smem_tensor_map = ArchSpec::get_tensormap_smem_size(gemm_type);

    // Sum them up
    int smem_size = 0;
    smem_size += smem_tensor_map;
    smem_size += smem_cd;
    smem_size += num_stages * smem_a_per_stage;
    smem_size += num_stages * smem_b_per_stage;
    smem_size += num_stages * smem_sfa_per_stage;
    smem_size += num_stages * smem_sfb_per_stage;
    smem_size += smem_extra_sfb;
    smem_size += smem_barrier;
    smem_size += smem_tmem_ptr;

    return SharedMemoryConfig {
        .smem_size = smem_size,
        .swizzle_a_mode = swizzle_a_mode,
        .swizzle_b_mode = swizzle_b_mode,
        .swizzle_cd_mode = swizzle_cd_mode,
    };
}

template <typename ArchSpec>
static GemmConfig get_best_config(const GemmType& gemm_type, const KernelType& kernel_type,
                                  const int& m, const int& n, const int& k, const int& num_groups,
                                  const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                  const at::ScalarType& a_dtype, const at::ScalarType& b_dtype,
                                  const at::ScalarType& cd_dtype,
                                  const bool& with_accumulation, const int& num_sms) {
    const auto mma_kind = (a_dtype == torch::kBFloat16 ? MmaKind::BF16 : MmaKind::MXFP8FP4);
    if (mma_kind == MmaKind::BF16) {
        DG_HOST_ASSERT(a_dtype == torch::kBFloat16 and b_dtype == torch::kBFloat16);
    } else {
        DG_HOST_ASSERT(a_dtype == torch::kFloat8_e4m3fn or a_dtype == kPackedFP4);
        DG_HOST_ASSERT(b_dtype == torch::kFloat8_e4m3fn or b_dtype == kPackedFP4);
    }
    DG_HOST_ASSERT(cd_dtype == torch::kBFloat16 or cd_dtype == torch::kFloat);

    // Select M/N block sizes
    auto block_ms = ArchSpec::get_block_m_candidates(kernel_type, major_a, m);
    if (gemm_type == GemmType::MGroupedContiguous)
        block_ms = std::vector{get_mk_alignment_for_contiguous_layout()};
    if (gemm_type == GemmType::MGroupedMasked or gemm_type == GemmType::MGroupedContiguousWithPsumLayout) 
        block_ms = std::vector{64, 128};    // Exclude 256 for performance
    auto block_ns = ArchSpec::get_block_n_candidates(kernel_type, cd_dtype);

    // NOTES: TMA copy .b4x16_p64 only supports Swizzle 128B
    // TODO: Optimize it
    if (a_dtype == kPackedFP4 and major_a == cute::UMMA::Major::MN)
        block_ms = std::vector{128};
    if (b_dtype == kPackedFP4 and major_b == cute::UMMA::Major::MN)
        block_ns = std::vector{128};

    // K block size is selected in a fixed manner
    const auto& block_k = (mma_kind == MmaKind::BF16 ? 64 : 128);

    // Some util functions
    const auto& get_num_blocks = [=](const int& block_m, const int& block_n) {
        return ceil_div(m, block_m) * ceil_div(n, block_n) * num_groups;
    };
    const auto& get_num_waves = [=](const int& block_m, const int& block_n) {
        return ceil_div(get_num_blocks(block_m, block_n), num_sms);
    };
    const auto& get_last_wave_util = [=](const int& block_m, const int& block_n) {
        const auto& num_last_blocks = get_num_blocks(block_m, block_n) % num_sms;
        return num_last_blocks == 0 ? num_sms : num_last_blocks;
    };

    // Decide block sizes by waves
    int best_block_m = 0, best_block_n = 0;
    int best_num_waves = 0, best_last_util = 0;
    for (const auto& block_m: block_ms) {
        for (const auto& block_n: block_ns) {
            const int& num_waves = get_num_waves(block_m, block_n);
            const auto& last_util = get_last_wave_util(block_m, block_n);
            if (not ArchSpec::is_block_size_legal(kernel_type, major_a, major_b, mma_kind, cd_dtype, m, n, k, block_m, block_n, block_k))
                continue;

            bool success = false;
            if (best_block_m == 0 or best_block_n == 0 or num_waves < best_num_waves) {
                success = true;
            } else if (num_waves == best_num_waves) {
                // Check last wave utilization
                success = last_util > best_last_util;
                if (last_util == best_last_util) {
                    // Case 1: same `block_m`, smaller `block_n` (wasted)
                    success |= block_m == best_block_m and block_n < best_block_n;
                    // Case 2: same `block_n`, smaller `block_m` (wasted)
                    success |= block_n == best_block_n and block_m < best_block_m;
                    // Case 3: different for both `block_m` and `block_n`, larger `block_n` is better
                    // NOTES: don't pick `block_m/block_n` larger than shape `m/n` in this case
                    success |= block_m != best_block_m and block_n > best_block_n 
                               and block_n <= n and block_m <= m;
                }
            }

            // Replace with the new config if successful
            if (success) {
                best_block_m = block_m, best_block_n = block_n;
                best_num_waves = num_waves, best_last_util = last_util;
            }
        }
    }
    DG_HOST_ASSERT(best_block_m > 0 and best_block_n > 0);

    // Decide the number of TMA multicasts and whether broadcast on A
    MulticastConfig best_multicast_config = {1, false};
    auto [is_legal_on_a, is_legal_on_b] = ArchSpec::get_multicast_legality(
        gemm_type, num_groups, m, n, best_block_m, best_block_n, num_sms);

    // NOTES: TMA copy .b4x16_p64 only supports Swizzle 128B
    // TODO: Optimize it
    if (a_dtype == kPackedFP4 and major_a == cute::UMMA::Major::MN)
        is_legal_on_a = false;
    if (b_dtype == kPackedFP4 and major_b == cute::UMMA::Major::MN)
        is_legal_on_b = false;

    const bool is_legal[2] = {is_legal_on_b, is_legal_on_a};
    bool order[2] = {false, true};
    if (best_block_m > best_block_n)
        std::swap(order[0], order[1]);
    for (const bool& is_multicast_on_a: order) {
        if (m >= 512 and is_legal[static_cast<int>(is_multicast_on_a)]) {
            best_multicast_config = {2, is_multicast_on_a};
            break;
        }
    }

    // Always pick the largest number of stage
    constexpr int smem_capacity = ArchSpec::smem_capacity;
    int best_num_stages = 0;
    SharedMemoryConfig best_smem_config;
    for (int num_stages = 32; num_stages > 0; -- num_stages) {
        if (not ArchSpec::is_num_stages_legal(mma_kind, cd_dtype, num_stages, best_block_m, best_block_n, block_k))
            continue;

        best_smem_config = get_smem_config<ArchSpec>(gemm_type, kernel_type,
                                                     m, n, k,
                                                     best_block_m, best_block_n, block_k,
                                                     major_a, major_b,
                                                     mma_kind, cd_dtype,
                                                     num_stages, best_multicast_config);
        if (best_smem_config.smem_size <= smem_capacity) {
            best_num_stages = num_stages;
            break;
        }
    }
    DG_HOST_ASSERT(best_num_stages != 0);

    // Recompute the minimal number of SMs required
    // NOTES: less L2 cache usage and less GPU frequency drop
    int num_min_sms = num_sms;
    if (get_env<int>("DG_JIT_MINIMIZE_NUM_SMS", 0)) {
        num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, best_num_waves);
        num_min_sms = align(num_min_sms, best_multicast_config.num_multicast);
        DG_HOST_ASSERT(num_min_sms <= num_sms);
    }

    const auto& config = GemmConfig {
        .gemm_type = gemm_type,
        .kernel_type = kernel_type,
        .mma_kind = mma_kind,
        .a_dtype = a_dtype,
        .b_dtype = b_dtype,
        .cd_dtype = cd_dtype,
        .major_a = major_a,
        .major_b = major_b,
        .with_accumulation = with_accumulation,
        .block_m = best_block_m,
        .block_n = best_block_n,
        .block_k = block_k,
        .num_stages = best_num_stages,
        .num_last_stages = ceil_div(k, block_k) % best_num_stages,
        .num_sms = num_min_sms,
        .tc_util = device_runtime->get_tc_util(),
        .multicast_config = best_multicast_config,
        // ReSharper disable once CppLocalVariableMightNotBeInitialized
        .smem_config = best_smem_config,
        .thread_config = ArchSpec::get_thread_config(kernel_type, best_block_m, best_block_n)
    };

    // Only SM100 BF16 kernels support tensor core control
    if (config.tc_util < 100)
        DG_HOST_ASSERT(device_runtime->get_arch_major() == 10 and mma_kind == MmaKind::BF16);

    // Print configs for the first time
    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        auto key = std::make_tuple(gemm_type, kernel_type, m, n, k, num_groups, major_a, major_b,
                                   mma_kind, a_dtype, b_dtype, cd_dtype, with_accumulation, num_sms);
        static std::set<decltype(key)> printed;
        if (printed.count(key) == 0) {
            printf("GEMM type: %d, kernel type: %d, M: %d, N: %d, K: %d, groups: %d, "
                   "A major: %d, B major: %d, MMA kind: %d, A dtype: %s, B dtype: %s, CD dtype: %s, accumulation: %d, "
                   "SM limit: %d -> block M: %d, block N: %d, block K: %d, stages: %d, last stages: %d, "
                   "SMs: %d, multicast: %d, multicast on A: %d, shared memory: %d bytes, swizzle A: %d, "
                   "swizzle B: %d, swizzle CD: %d, SMs: %d, threads: %d, TC util: %d%%\n",
                   static_cast<int>(gemm_type), static_cast<int>(kernel_type), m, n, k, num_groups,
                   static_cast<int>(major_a), static_cast<int>(major_b), static_cast<int>(mma_kind),
                   c10::toString(a_dtype), c10::toString(b_dtype), c10::toString(cd_dtype),
                   static_cast<int>(with_accumulation), num_sms, best_block_m, best_block_n, block_k,
                   best_num_stages, config.num_last_stages, num_min_sms, best_multicast_config.num_multicast,
                   static_cast<int>(best_multicast_config.is_multicast_on_a),
                   best_smem_config.smem_size, best_smem_config.swizzle_a_mode, best_smem_config.swizzle_b_mode,
                   best_smem_config.swizzle_cd_mode, config.num_sms, config.thread_config.num_threads, config.tc_util);
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
