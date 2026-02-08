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

class SM100BmkBnkMnRuntime final: public LaunchRuntime<SM100BmkBnkMnRuntime> {
public:
    struct Args {
        int s, m, n, k;
        int block_m, block_n, block_k;
        int split_factor;
        int swizzle_ab_mode, swizzle_cd_mode;
        int num_stages;
        int num_threads;

        LaunchArgs launch_args;

        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bmk_bnk_mn.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bmn_bnk_mn_gemm_impl<
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {}, {}
    >);
}};
)",
        args.m, args.n, args.k,
        args.block_m, args.block_n, args.block_k,
        args.split_factor,
        args.swizzle_ab_mode, args.swizzle_cd_mode,
        args.num_stages, args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.s, args.tensor_map_a, args.tensor_map_b, args.tensor_map_d));
    }
};


static void sm100_bmn_bnk_mn_gemm(const torch::Tensor &a,
                                  const torch::Tensor &b,
                                  const torch::Tensor &d,
                                  const int &s, const int &m, const int &n, const int &k) {
    constexpr int block_m = 128;
    constexpr int block_n = 128;
    constexpr int block_k = 64;
    constexpr int num_threads = 128;
    DG_HOST_ASSERT(k % block_k == 0);
    DG_HOST_ASSERT(m % 64 == 0 and n % 64 == 0);
    DG_HOST_ASSERT(static_cast<int64_t>(s) * static_cast<int64_t>(std::max(m, n)) <= std::numeric_limits<int>::max());

    const int swizzle_ab_mode = get_swizzle_mode(block_k, static_cast<int>(a.element_size()));
    const int swizzle_cd_mode = get_swizzle_mode(block_n, static_cast<int>(d.element_size()));

    // Get best config
    const int num_sms = device_runtime->get_num_sms();
    const int num_mn_blocks = ceil_div(m, block_m) * ceil_div(n, block_n);
    const int num_sk_blocks = s * (k / block_k);
    const int split_factor = ceil_div(num_sk_blocks, std::max(num_sms / num_mn_blocks, 1));

    // Select best number of stages
    // NOTES: we select 4 as start, as it is tested to be faster than values > 4
    int num_stages = 4, smem_size = 0;
    while (true) {
        const int& smem_cd = block_m * swizzle_cd_mode * 2;
        const int& smem_a_per_stage = block_m * block_k * sizeof(cutlass::bfloat16_t);
        const int& smem_b_per_stage = block_n * block_k * sizeof(cutlass::bfloat16_t);
        const int& smem_barrier = SM100ArchSpec::get_barrier_smem_size(num_stages);
        const int& smem_tmem_ptr = SM100ArchSpec::get_tmem_ptr_smem_size();

        smem_size = 0;
        smem_size += smem_cd;
        smem_size += (smem_a_per_stage + smem_b_per_stage) * num_stages;
        smem_size += smem_barrier;
        smem_size += smem_tmem_ptr;
        if (smem_size <= SM100ArchSpec::smem_capacity)
            break;

        -- num_stages;
    }
    DG_HOST_ASSERT(num_stages > 0);

    // Print configs
    if (get_env("DG_JIT_DEBUG", 0)) {
        printf("S: %d, M: %d, N: %d, K: %d -> "
               "block M: %d, block N: %d, block K: %d, split-K factor: %d"
               "stages: %d, shared memory: %d, swizzle AB: %d, swizzle CD: %d\n",
               s, m, n, k, block_m, block_n, block_k, split_factor,
               num_stages, smem_size, swizzle_ab_mode, swizzle_cd_mode);
    }

    const auto& tensor_map_a = make_tma_2d_desc(a, k, s * m, block_k, block_m, k, swizzle_ab_mode);
    const auto& tensor_map_b = make_tma_2d_desc(b, k, s * n, block_k, block_n, k, swizzle_ab_mode);
    const auto& tensor_map_d = make_tma_2d_desc(d, n, m, block_n, block_m, n, swizzle_cd_mode);

    const SM100BmkBnkMnRuntime::Args& args = {
        .s = s, .m = m, .n = n, .k = k,
        .block_m = block_m, .block_n = block_n, .block_k = block_k,
        .split_factor = split_factor,
        .swizzle_ab_mode = swizzle_ab_mode,
        .swizzle_cd_mode = swizzle_cd_mode,
        .num_stages = num_stages,
        .num_threads = num_threads,
        .launch_args = LaunchArgs(num_mn_blocks * ceil_div(num_sk_blocks, split_factor), num_threads, smem_size),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d
    };
    const auto& code = SM100BmkBnkMnRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_bmn_bnk_mn_gemm", code);
    SM100BmkBnkMnRuntime::launch(runtime, args);
}

} // namespace deep_gemm
