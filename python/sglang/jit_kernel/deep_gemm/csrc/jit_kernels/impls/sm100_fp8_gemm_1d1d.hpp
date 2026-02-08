#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"

#include "epilogue.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM100FP8FP4Gemm1D1DRuntime final: public LaunchRuntime<SM100FP8FP4Gemm1D1DRuntime> {
public:
    struct Args {
        int m, n, k, num_groups;
        int gran_k_a, gran_k_b;
        const std::string& compiled_dims;
        const std::optional<std::string>& epilogue_type;

        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void* grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_gemm_1d1d_impl<
        {}, {},
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}, {}, {},
        {}
    >);
}};
)",
        to_string(args.gemm_config.major_a), to_string(args.gemm_config.major_b),
        args.gran_k_a, args.gran_k_b,
        get_compiled_dim(args.m, 'm', args.compiled_dims), get_compiled_dim(args.n, 'n', args.compiled_dims), get_compiled_dim(args.k, 'k', args.compiled_dims),
        args.gemm_config.block_m, args.gemm_config.block_n, args.gemm_config.block_k,
        args.num_groups,
        args.gemm_config.smem_config.swizzle_a_mode, args.gemm_config.smem_config.swizzle_b_mode, args.gemm_config.smem_config.swizzle_cd_mode,
        args.gemm_config.num_stages,
        args.gemm_config.thread_config.num_non_epilogue_threads, args.gemm_config.thread_config.num_epilogue_threads,
        args.gemm_config.multicast_config.num_multicast, args.gemm_config.multicast_config.is_multicast_on_a,
        args.gemm_config.num_sms,
        to_string(args.gemm_config.gemm_type), args.gemm_config.with_accumulation,
        to_string(args.gemm_config.a_dtype), to_string(args.gemm_config.b_dtype), to_string(args.gemm_config.cd_dtype),
        get_default_epilogue_type(args.epilogue_type));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.grouped_layout, args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd));
    }
};

static void sm100_fp8_fp4_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                    const torch::Tensor& b, const torch::Tensor& sfb,
                                    const std::optional<torch::Tensor>& c,
                                    const torch::Tensor& d,
                                    const int& m, const int& n, const int& k,
                                    const int& gran_k_a, const int& gran_k_b,
                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                    const std::string& compiled_dims,
                                    const std::optional<std::string>& epilogue_type = std::nullopt) {
    const auto& config = get_best_config<SM100ArchSpec>(
        GemmType::Normal, KernelType::Kernel1D1D,
        m, n, k, 1, major_a, major_b,
        a.scalar_type(), b.scalar_type(),
        d.scalar_type(), c.has_value(),
        device_runtime->get_num_sms());

    const auto& cd = c.value_or(d);
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, static_cast<int>(d.size(-1)),
                                                 SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), 1,
                                                 config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, gran_k_a, 1, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                  config.block_n, gran_k_b, 1, 0);

    // Launch
    const SM100FP8FP4Gemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .num_groups = 1,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        .compiled_dims = compiled_dims,
        .epilogue_type = epilogue_type,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };
    const auto& code = SM100FP8FP4Gemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_fp8_fp4_gemm_1d1d", code);
    SM100FP8FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                         const torch::Tensor& b, const torch::Tensor& sfb,
                                                         const torch::Tensor& d,
                                                         const torch::Tensor& grouped_layout,
                                                         const int& num_groups, const int& m, const int& n, const int& k,
                                                         const int& gran_k_a, const int& gran_k_b,
                                                         const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                         const std::string& compiled_dims,
                                                         const bool& use_psum_layout,
                                                         const std::optional<int>& expected_m_for_psum_layout) {
    const auto& gemm_type = use_psum_layout ? GemmType::MGroupedContiguousWithPsumLayout : GemmType::MGroupedContiguous;

    // NOTES: If actual M is dynamic, estimate config via `num_groups` and `expected_m`.
    //        Otherwise, treat the contiguous layout as a whole.
    const auto& m_for_config = expected_m_for_psum_layout.has_value() ? expected_m_for_psum_layout.value() : m;
    const auto& num_groups_for_config = expected_m_for_psum_layout.has_value() ? num_groups : 1;

    const auto& config = get_best_config<SM100ArchSpec>(
        gemm_type, KernelType::Kernel1D1D,
        m_for_config, n, k, num_groups_for_config, major_a, major_b,
        a.scalar_type(), b.scalar_type(),
        d.scalar_type(), false,
        device_runtime->get_num_sms());

    // Create tensor descriptors
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), 1,
                                                 config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, gran_k_a, 1, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                  config.block_n, gran_k_b, num_groups, 0);

    // Launch kernel
    const SM100FP8FP4Gemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .num_groups = num_groups,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        .compiled_dims = compiled_dims,
        .epilogue_type = std::nullopt,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = grouped_layout.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };
    const auto& code = SM100FP8FP4Gemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d", code);
    SM100FP8FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm100_m_grouped_fp8_fp4_gemm_masked_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                     const torch::Tensor& b, const torch::Tensor& sfb,
                                                     const torch::Tensor& d,
                                                     const torch::Tensor& masked_m,
                                                     const int& num_groups, const int& m, const int& n, const int& k,
                                                     const int& expected_m,
                                                     const int& gran_k_a, const int& gran_k_b,
                                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                     const std::string& compiled_dims) {
    const auto& config = get_best_config<SM100ArchSpec>(
        GemmType::MGroupedMasked, KernelType::Kernel1D1D,
        expected_m, n, k, num_groups, major_a, major_b,
        a.scalar_type(), b.scalar_type(),
        d.scalar_type(), false,
        device_runtime->get_num_sms());

    // Create tensor descriptors
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), num_groups,
                                                 config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, gran_k_a, num_groups, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                  config.block_n, gran_k_b, num_groups, 0);

    // Launch kernel
    const SM100FP8FP4Gemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .num_groups = num_groups,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        .compiled_dims = compiled_dims,
        .epilogue_type = std::nullopt,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = masked_m.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };
    const auto& code = SM100FP8FP4Gemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_m_grouped_fp8_fp4_gemm_masked_1d1d", code);
    SM100FP8FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm100_k_grouped_fp8_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                          const torch::Tensor& b, const torch::Tensor& sfb,
                                          const std::optional<torch::Tensor>& c,
                                          const torch::Tensor& d,
                                          const int& m, const int& n,
                                          const std::vector<int>& ks, const torch::Tensor& ks_tensor,
                                          const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                          const std::string& compiled_dims) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::MN and major_b == cute::UMMA::Major::MN);

    int sum_k = 0, sum_sf_k = 0;
    for (const auto& k: ks) {
        sum_k += k, sum_sf_k += ceil_div(k, 512);
        DG_HOST_ASSERT(k % 128 == 0);
    }
    const auto& num_groups = static_cast<int>(ks.size());

    // Get config using max K for better performance
    const auto& max_k = *std::max_element(ks.begin(), ks.end());
    const auto& config = get_best_config<SM100ArchSpec>(
        GemmType::KGroupedContiguous, KernelType::Kernel1D1D,
        m, n, max_k, num_groups, cute::UMMA::Major::MN, cute::UMMA::Major::MN,
        a.scalar_type(), b.scalar_type(),
        d.scalar_type(), c.has_value(),
        device_runtime->get_num_sms());

    // Create tensor descriptors
    const auto& tensor_map_a = make_tma_a_desc(cute::UMMA::Major::MN, a, m, sum_k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(0)), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(cute::UMMA::Major::MN, b, n, sum_k,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(0)), 1,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(1)), num_groups,
                                                 config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, sum_sf_k * 512,
                                                  config.block_m, config.block_k, 1, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, sum_sf_k * 512,
                                                  config.block_n, config.block_k, 1, 0);

    // Launch kernel
    const SM100FP8FP4Gemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = sum_k,
        .num_groups = num_groups,
        .gran_k_a = 128,
        .gran_k_b = 128,
        .compiled_dims = compiled_dims,
        .epilogue_type = std::nullopt,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = ks_tensor.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };
    const auto& code = SM100FP8FP4Gemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_k_grouped_fp8_gemm_1d1d", code);
    SM100FP8FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm100_fp8_bmm(const torch::Tensor& a, const torch::Tensor& sfa,
                          const torch::Tensor& b, const torch::Tensor& sfb,
                          const std::optional<torch::Tensor>& c,
                          const torch::Tensor& d,
                          const int& batch_size, const int& m, const int& n, const int& k,
                          const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                          const std::string& compiled_dims) {
    const auto& config = get_best_config<SM100ArchSpec>(
        GemmType::Batched, KernelType::Kernel1D1D,
        m, n, k, batch_size, major_a, major_b,
        a.scalar_type(), b.scalar_type(),
        d.scalar_type(), c.has_value(),
        device_runtime->get_num_sms());

    const int& load_block_m = SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m);
    const auto& [inner_dim_a, outer_dim_a] = get_inner_outer_dims(major_a, k, m);
    const auto& [inner_block_a, outer_block_a] = get_inner_outer_dims(major_a, config.block_k, load_block_m);
    const auto& tensor_map_a = make_tma_3d_desc(a, inner_dim_a, outer_dim_a, batch_size,
                                                inner_block_a, outer_block_a, 1,
                                                a.stride(major_a == cute::UMMA::Major::K ? 1 : 2),
                                                a.stride(0),
                                                config.smem_config.swizzle_a_mode);

    const int& load_block_n = SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n);
    const auto& [inner_dim_b, outer_dim_b] = get_inner_outer_dims(major_b, k, n);
    const auto& [inner_block_b, outer_block_b] = get_inner_outer_dims(major_b, config.block_k, load_block_n);
    const auto& tensor_map_b = make_tma_3d_desc(b, inner_dim_b, outer_dim_b, batch_size,
                                                inner_block_b, outer_block_b, 1,
                                                b.stride(major_b == cute::UMMA::Major::K ? 1 : 2),
                                                b.stride(0),
                                                config.smem_config.swizzle_b_mode);

    const int& store_block_m = SM100ArchSpec::get_cd_store_block_m(config.block_m);
    const int& store_block_n = SM100ArchSpec::get_cd_store_block_n(config.block_n);
    const auto& tensor_map_cd = make_tma_3d_desc(d, n, m, batch_size,
                                                 store_block_n, store_block_m, 1,
                                                 d.stride(1), d.stride(0),
                                                 config.smem_config.swizzle_cd_mode);

    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, config.block_k, batch_size, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                  config.block_n, config.block_k, batch_size, 0);

    // Launch
    const SM100FP8FP4Gemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .num_groups = batch_size,
        .gran_k_a = 128,
        .gran_k_b = 128,
        .compiled_dims = compiled_dims,
        .epilogue_type = std::nullopt,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };
    const auto& code = SM100FP8FP4Gemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_fp8_gemm_1d1d", code);
    SM100FP8FP4Gemm1D1DRuntime::launch(runtime, args);
}

} // namespace deep_gemm
