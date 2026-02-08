#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM100BF16HCPrenormGemmRuntime final: public LaunchRuntime<SM100BF16HCPrenormGemmRuntime> {
public:
    struct Args {
        int m, n, k;
        int block_m, block_n, block_k;
        int num_splits;
        int swizzle_cd_mode;
        int num_stages;
        int num_mma_threads, num_cast_and_reduce_threads;

        LaunchArgs launch_args;

        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
        float* sqr_sum;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_tf32_hc_prenorm_gemm_impl<
        {}, {},
        {}, {}, {},
        {},
        {},
        {},
        {}, {}
    >);
}};
)",
        args.n, args.k,
        args.block_m, args.block_n, args.block_k,
        args.num_splits,
        args.swizzle_cd_mode,
        args.num_stages,
        args.num_mma_threads, args.num_cast_and_reduce_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.m, args.tensor_map_a, args.tensor_map_b, args.tensor_map_d, args.sqr_sum));
    }
};

static void sm100_tf32_hc_prenorm_gemm(const torch::Tensor& a,
                                       const torch::Tensor& b,
                                       const torch::Tensor& d,
                                       const torch::Tensor& sqr_sum,
                                       const int& m, const int& n, const int& k,
                                       const int& num_splits) {
    constexpr int block_m = 64;
    constexpr int block_k = 64;
    constexpr int num_mma_threads = 128;
    constexpr int num_cast_and_reduce_threads = 128;

    const int block_n = align(n, 16);
    DG_HOST_ASSERT(n <= block_n);
    DG_HOST_ASSERT(n <= 128 and n % 8 == 0);
    DG_HOST_ASSERT(k % block_k == 0);

    const auto& swizzle_cd_mode = get_swizzle_mode(block_n, sizeof(float));
    const auto& tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a, m, k,
                                               block_m, block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(cute::UMMA::Major::K))), 1,
                                               get_swizzle_mode(block_k, a.element_size()), 0,
                                               true);
    const auto& tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b, n, k,
                                               block_n, block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(cute::UMMA::Major::K))), 1,
                                               get_swizzle_mode(block_k, b.element_size()), 0,
                                               true);
    const auto& tensor_map_d = num_splits == 1 ? make_tma_cd_desc(d, m, n,
                                                                  block_m, block_n,
                                                                  static_cast<int>(d.stride(-2)), 1,
                                                                  swizzle_cd_mode)
                                               : make_tma_3d_desc(d, n, m, num_splits,
                                                                  block_n, block_m, 1,
                                                                  static_cast<int>(d.stride(-2)),
                                                                  static_cast<int>(d.stride(-3)),
                                                                  swizzle_cd_mode);

    // Calculate stages
    int num_stages = 12, smem_size = 0;
    while (num_stages > 0) {
        const int smem_a_per_stage = block_m * block_k * static_cast<int>(sizeof(nv_bfloat16));
        const int smem_b_per_stage = block_n * block_k * static_cast<int>(sizeof(float));
        const int smem_cd = block_m * swizzle_cd_mode;
        const int smem_barriers = (num_stages * 4 + 1) * 8;
        const int smem_tmem_ptr = 4;
        smem_size = (smem_a_per_stage + smem_b_per_stage) * num_stages +
                    smem_cd + smem_barriers + smem_tmem_ptr;

        if (smem_size <= SM100ArchSpec::smem_capacity)
            break;
        -- num_stages;
    }
    DG_HOST_ASSERT(num_stages > 0);

    // Print configs
    if (get_env("DG_JIT_DEBUG", 0)) {
        printf("M: %d, N: %d, K: %d -> "
               "block M: %d, block N: %d, block K: %d, split K: %d"
               "stages: %d, shared memory: %d, swizzle CD: %d\n",
               m, n, k, block_m, block_n, block_k, num_splits,
               num_stages, smem_size, swizzle_cd_mode);
    }

    // Launch
    const SM100BF16HCPrenormGemmRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .block_m = block_m, .block_n = block_n, .block_k = block_k,
        .num_splits = num_splits,
        .swizzle_cd_mode = swizzle_cd_mode,
        .num_stages = num_stages,
        .num_mma_threads = num_mma_threads,
        .num_cast_and_reduce_threads = num_cast_and_reduce_threads,
        .launch_args = LaunchArgs(num_splits * ceil_div(m, block_m), num_mma_threads + num_cast_and_reduce_threads, smem_size, 1),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .sqr_sum = sqr_sum.data_ptr<float>()
    };
    const auto& code = SM100BF16HCPrenormGemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_tf32_hc_prenorm_gemm", code);
    SM100BF16HCPrenormGemmRuntime::launch(runtime, args);
}

} // namespace deep_gemm
