#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM90FP8Gemm1D1DRuntime final: public LaunchRuntime<SM90FP8Gemm1D1DRuntime> {
public:
    struct Args {
        int m, n, k, num_groups;
        const std::string& compiled_dims;

        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void *gmem_a_ptr;
        void *gmem_b_ptr;
        void *grouped_layout;
        void *tensor_map_buffer;
        CUtensorMap tensor_map_a_base;
        CUtensorMap tensor_map_b_base;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d1d_impl<
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {}
    >);
}};
)",
        get_compiled_dim(args.m, 'm', args.compiled_dims), get_compiled_dim(args.n, 'n', args.compiled_dims), get_compiled_dim(args.k, 'k', args.compiled_dims),
        args.num_groups,
        args.gemm_config.block_m, args.gemm_config.block_n, args.gemm_config.block_k,
        args.gemm_config.smem_config.swizzle_a_mode, args.gemm_config.smem_config.swizzle_b_mode,
        args.gemm_config.num_stages,
        args.gemm_config.thread_config.num_tma_threads, args.gemm_config.thread_config.num_math_threads,
        args.gemm_config.multicast_config.num_multicast, args.gemm_config.multicast_config.is_multicast_on_a,
        args.gemm_config.num_sms, to_string(args.gemm_config.gemm_type),
        to_string(args.gemm_config.cd_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.gmem_a_ptr, args.gmem_b_ptr,
            args.grouped_layout,
            args.tensor_map_buffer,
            args.m, args.n, args.k,
            args.tensor_map_a_base, args.tensor_map_b_base,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd));
    }
};

static void sm90_fp8_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                               const torch::Tensor& b, const torch::Tensor& sfb,
                               const std::optional<torch::Tensor>& c,
                               const torch::Tensor& d,
                               const int& m, const int& n, const int& k,
                               const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                               const std::string& compiled_dims) {
    DG_HOST_ASSERT(c.has_value() and d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const auto& config = get_best_config<SM90ArchSpec>(
        GemmType::Normal, KernelType::Kernel1D1D,
        m, n, k, 1, major_a, major_b,
        a.scalar_type(), b.scalar_type(),
        d.scalar_type(), c.has_value(),
        device_runtime->get_num_sms());

    // Requires no TMA splits
    DG_HOST_ASSERT(config.smem_config.swizzle_a_mode == config.block_k);
    DG_HOST_ASSERT(config.smem_config.swizzle_b_mode == config.block_k);

    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                                SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                                config.block_k, k, 1,
                                                config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                                SM90ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                                config.block_k, k, 1,
                                                config.smem_config.swizzle_b_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, config.block_k, 1, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                  config.block_n, config.block_k, 1, 0);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM90ArchSpec::get_cd_store_block_m(config.block_m, true),
                                                 SM90ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), 1,
                                                 0);

    // Launch
    const SM90FP8Gemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .num_groups = 1,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .gmem_a_ptr = nullptr,
        .gmem_b_ptr = nullptr,
        .grouped_layout = nullptr,
        .tensor_map_buffer = nullptr,
        .tensor_map_a_base = tensor_map_a,
        .tensor_map_b_base = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto& code = SM90FP8Gemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_fp8_gemm_1d1d", code);

    SM90FP8Gemm1D1DRuntime::launch(runtime, args);
}

static void sm90_k_grouped_fp8_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                         const torch::Tensor& b, const torch::Tensor& sfb,
                                         const std::optional<torch::Tensor>& c,
                                         const torch::Tensor& d,
                                         const int& m, const int& n,
                                         const std::vector<int>& ks, const torch::Tensor& ks_tensor,
                                         const torch::Tensor& tensor_map_buffer,
                                         const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                         const std::string& compiled_dims) {
    DG_HOST_ASSERT(c.has_value() and d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    // Get config using max K for better performance
    const auto& num_groups = static_cast<int>(ks.size());
    const auto& max_k = *std::max_element(ks.begin(), ks.end());
    const auto& config = get_best_config<SM90ArchSpec>(
        GemmType::KGroupedContiguous, KernelType::Kernel1D1D,
        m, n, max_k, num_groups, major_a, major_b,
        a.scalar_type(), b.scalar_type(),
        d.scalar_type(), c.has_value(),
        device_runtime->get_num_sms());

    // Requires no TMA splits
    DG_HOST_ASSERT(config.smem_config.swizzle_a_mode == config.block_k);
    DG_HOST_ASSERT(config.smem_config.swizzle_b_mode == config.block_k);

    int first_k = 0, sum_k = 0, sum_sf_k = 0;
    for (int i = 0; i < num_groups; ++ i) {
        if (first_k == 0 and ks[i] != 0)
            first_k = ks[i];
        sum_k += ks[i], sum_sf_k += ceil_div(ks[i], 128);
        DG_HOST_ASSERT(ks[i] % 128 == 0);
    }
    const auto& tensor_map_a_base = make_tma_a_desc(major_a, a, m, first_k,
                                                    SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                                    config.block_k, first_k, 1,
                                                    config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b_base = make_tma_b_desc(major_b, b, n, first_k,
                                                    SM90ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                                    config.block_k, first_k, 1,
                                                    config.smem_config.swizzle_b_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, sum_sf_k * 128,
                                                  config.block_m, config.block_k, 1, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, sum_sf_k * 128,
                                                  config.block_n, config.block_k, 1, 0);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM90ArchSpec::get_cd_store_block_m(config.block_m, true),
                                                 SM90ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), num_groups,
                                                 config.smem_config.swizzle_cd_mode);

    // Launch
    const SM90FP8Gemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = sum_k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .gmem_a_ptr = a.data_ptr(),
        .gmem_b_ptr = b.data_ptr(),
        .grouped_layout = ks_tensor.data_ptr(),
        .tensor_map_buffer = tensor_map_buffer.data_ptr(),
        .tensor_map_a_base = tensor_map_a_base,
        .tensor_map_b_base = tensor_map_b_base,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto& code = SM90FP8Gemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_fp8_gemm_1d1d", code);

    SM90FP8Gemm1D1DRuntime::launch(runtime, args);
}

} // namespace deep_gemm
