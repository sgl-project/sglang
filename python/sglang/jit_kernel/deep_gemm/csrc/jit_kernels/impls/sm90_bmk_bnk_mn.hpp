#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM90BmkBnkMnRuntime final: public LaunchRuntime<SM90BmkBnkMnRuntime> {
public:
    struct Args {
        int s, m, n, k;
        int block_m, block_n, block_k;
        int split_factor;
        int num_stages;
        int num_tma_threads, num_math_threads;

        LaunchArgs launch_args;

        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        float* d;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_bmk_bnk_mn.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_bmn_bnk_mn_gemm_impl<
        {}, {}, {},
        {}, {}, {},
        {},
        {},
        {}, {}
    >);
}};
)",
        args.m, args.n, args.k,
        args.block_m, args.block_n, args.block_k,
        args.split_factor,
        args.num_stages,
        args.num_tma_threads, args.num_math_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.s, args.tensor_map_a, args.tensor_map_b, args.d));
    }
};


static void sm90_bmn_bnk_mn_gemm(const torch::Tensor &a,
                                 const torch::Tensor &b,
                                 const torch::Tensor &d,
                                 const int &s, const int &m, const int &n, const int &k) {
    constexpr int block_m = 128;
    constexpr int block_n = 128;
    constexpr int block_k = 64;
    constexpr int num_tma_threads = 128;
    constexpr int num_math_threads = 256;
    DG_HOST_ASSERT(k % block_k == 0);
    DG_HOST_ASSERT(m % 64 == 0 and n % 64 == 0);
    DG_HOST_ASSERT(static_cast<int64_t>(s) * static_cast<int64_t>(std::max(m, n)) <= std::numeric_limits<int>::max());

    const int swizzle_ab_mode = get_swizzle_mode(block_k, static_cast<int>(a.element_size()));
    DG_HOST_ASSERT(swizzle_ab_mode == 128);

    // Get best config
    const int num_sms = device_runtime->get_num_sms();
    const int num_mn_blocks = ceil_div(m, block_m) * ceil_div(n, block_n);
    const int num_sk_blocks = s * (k / block_k);
    const int split_factor = ceil_div(num_sk_blocks, std::max(num_sms / num_mn_blocks, 1));

    // Select best number of stages
    int num_stages = 4, smem_size = 0;
    while (true) {
        const int& smem_a_per_stage = block_m * block_k * sizeof(cutlass::bfloat16_t);
        const int& smem_b_per_stage = block_n * block_k * sizeof(cutlass::bfloat16_t);
        const int& smem_barrier = SM90ArchSpec::get_barrier_smem_size(num_stages);

        smem_size = 0;
        smem_size += (smem_a_per_stage + smem_b_per_stage) * num_stages;
        smem_size += smem_barrier;

        if (smem_size <= SM90ArchSpec::smem_capacity)
            break;

        -- num_stages;
    }
    DG_HOST_ASSERT(num_stages > 0);

    // Print configs
    if (get_env("DG_JIT_DEBUG", 0)) {
        printf("S: %d, M: %d, N: %d, K: %d -> "
               "block M: %d, block N: %d, block K: %d, split-K factor: %d"
               "stages: %d, shared memory: %d, swizzle AB: %d\n",
               s, m, n, k, block_m, block_n, block_k, split_factor,
               num_stages, smem_size, swizzle_ab_mode);
    }

    const auto& tensor_map_a = make_tma_2d_desc(a, k, s * m, block_k, block_m, k, swizzle_ab_mode);
    const auto& tensor_map_b = make_tma_2d_desc(b, k, s * n, block_k, block_n, k, swizzle_ab_mode);

    const SM90BmkBnkMnRuntime::Args& args = {
        .s = s, .m = m, .n = n, .k = k,
        .block_m = block_m, .block_n = block_n, .block_k = block_k,
        .split_factor = split_factor,
        .num_stages = num_stages,
        .num_tma_threads = num_tma_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(num_mn_blocks * ceil_div(num_sk_blocks, split_factor), num_tma_threads + num_math_threads, smem_size),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .d = d.data_ptr<float>()
    };
    const auto& code = SM90BmkBnkMnRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_bmn_bnk_mn_gemm", code);
    SM90BmkBnkMnRuntime::launch(runtime, args);
}

} // namespace deep_gemm
