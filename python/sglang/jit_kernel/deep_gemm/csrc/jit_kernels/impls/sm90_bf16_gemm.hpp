#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM90BF16GemmRuntime final: public LaunchRuntime<SM90BF16GemmRuntime> {
public:
    struct Args {
        int m, n, k, num_groups;
        const std::string& compiled_dims;

        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void *grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_bf16_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_bf16_gemm_impl<
        {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)",
        // TODO: add CD dtype
        to_string(args.gemm_config.major_a), to_string(args.gemm_config.major_b),
        get_compiled_dim(args.m, 'm', args.compiled_dims), get_compiled_dim(args.n, 'n', args.compiled_dims), get_compiled_dim(args.k, 'k', args.compiled_dims),
        args.num_groups,
        args.gemm_config.block_m, args.gemm_config.block_n, args.gemm_config.block_k,
        args.gemm_config.smem_config.swizzle_a_mode, args.gemm_config.smem_config.swizzle_b_mode, args.gemm_config.smem_config.swizzle_cd_mode,
        args.gemm_config.num_stages,
        args.gemm_config.thread_config.num_tma_threads, args.gemm_config.thread_config.num_math_threads,
        args.gemm_config.multicast_config.num_multicast, args.gemm_config.multicast_config.is_multicast_on_a,
        args.gemm_config.num_sms,
        to_string(args.gemm_config.gemm_type), args.gemm_config.with_accumulation,
        to_string(args.gemm_config.cd_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.grouped_layout,
            args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_cd));
    }
};

static void sm90_bf16_gemm(const torch::Tensor& a,
                           const torch::Tensor& b,
                           const std::optional<torch::Tensor>& c,
                           const torch::Tensor& d,
                           const int& m, const int& n, const int& k,
                           const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                           const std::string& compiled_dims) {
    const auto& config = get_best_config<SM90ArchSpec>(
        GemmType::Normal, KernelType::KernelNoSF,
        m, n, k, 1, major_a, major_b,
        a.scalar_type(), b.scalar_type(),
        d.scalar_type(), c.has_value(),
        device_runtime->get_num_sms());

    // Requires no TMA splits
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM90ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                SM90ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM90ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.smem_config.swizzle_cd_mode);

    // Launch
    const SM90BF16GemmRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .num_groups = 1,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto& code = SM90BF16GemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_bf16_gemm", code);
    SM90BF16GemmRuntime::launch(runtime, args);
}

static void sm90_m_grouped_bf16_gemm_contiguous(const torch::Tensor& a,
                                                const torch::Tensor& b,
                                                const torch::Tensor& d,
                                                const torch::Tensor& m_indices,
                                                const int& num_groups, const int& m, const int& n, const int& k,
                                                const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                const std::string& compiled_dims) {
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(k % 64 == 0);

    const auto& config = get_best_config<SM90ArchSpec>(
        GemmType::MGroupedContiguous, KernelType::KernelNoSF,
        m, n, k, 1, major_a, major_b,
        a.scalar_type(), b.scalar_type(),
        d.scalar_type(), false,
        device_runtime->get_num_sms());

    // Requires no TMA splits
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM90ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                SM90ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM90ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.smem_config.swizzle_cd_mode);

    // Launch
    const SM90BF16GemmRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = m_indices.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto& code = SM90BF16GemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_m_grouped_bf16_gemm_contiguous", code);
    SM90BF16GemmRuntime::launch(runtime, args);
}

static void sm90_bf16_m_grouped_gemm_masked(const torch::Tensor& a,
                                            const torch::Tensor& b,
                                            const torch::Tensor& d,
                                            const torch::Tensor& masked_m,
                                            const int& num_groups, const int& m, const int& n, const int& k,
                                            const int& expected_m,
                                            const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                            const std::string& compiled_dims) {
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(k % 64 == 0);

    const auto& config = get_best_config<SM90ArchSpec>(
        GemmType::MGroupedMasked, KernelType::KernelNoSF,
        expected_m, n, k, num_groups, major_a, major_b,
        a.scalar_type(), b.scalar_type(),
        d.scalar_type(), false,
        device_runtime->get_num_sms());

    // Requires no TMA splits
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM90ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                SM90ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM90ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(d.stride(-2)), num_groups,
                                                config.smem_config.swizzle_cd_mode);

    // Launch
    const SM90BF16GemmRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = masked_m.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto& code = SM90BF16GemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_bf16_m_grouped_gemm_masked", code);
    SM90BF16GemmRuntime::launch(runtime, args);
}

static void sm90_bf16_k_grouped_gemm(const torch::Tensor& a,
                                     const torch::Tensor& b,
                                     const std::optional<torch::Tensor>& c,
                                     const torch::Tensor& d,
                                     const int& m, const int& n,
                                     const std::vector<int>& ks, const torch::Tensor& ks_tensor,
                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                     const std::string& compiled_dims) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::MN and major_b == cute::UMMA::Major::MN);

    int sum_k = 0;
    for (const auto& k: ks) {
        sum_k += k;
        DG_HOST_ASSERT(k % 128 == 0);
    }
    const auto& num_groups = static_cast<int>(ks.size());

    // Get config using max K for better performance
    const auto& max_k = *std::max_element(ks.begin(), ks.end());
    const auto& config = get_best_config<SM90ArchSpec>(
        GemmType::KGroupedContiguous, KernelType::KernelNoSF,
        m, n, max_k, num_groups, cute::UMMA::Major::MN, cute::UMMA::Major::MN,
        a.scalar_type(), b.scalar_type(),
        d.scalar_type(), c.has_value(),
        device_runtime->get_num_sms());

    // Create tensor descriptors
    const auto& tensor_map_a = make_tma_a_desc(cute::UMMA::Major::MN, a, m, sum_k,
                                               SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(0)), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(cute::UMMA::Major::MN, b, n, sum_k,
                                               SM90ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(0)), 1,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM90ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM90ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(1)), num_groups,
                                                 config.smem_config.swizzle_cd_mode);

    // Launch kernel
    const SM90BF16GemmRuntime::Args& args = {
        .m = m, .n = n, .k = sum_k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = ks_tensor.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto& code = SM90BF16GemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_bf16_k_grouped_gemm", code);
    SM90BF16GemmRuntime::launch(runtime, args);
}

static void sm90_bf16_bhr_hdr_bhd(const torch::Tensor& tensor_a,
                                  const torch::Tensor& tensor_b,
                                  const torch::Tensor& tensor_d,
                                  const int& b, const int& h, const int& r, const int& d,
                                  const std::string& compiled_dims = "nk") {
    const auto& config = get_best_config<SM90ArchSpec>(
        GemmType::Batched, KernelType::KernelNoSF,
        b, d, r, h, cute::UMMA::Major::K, cute::UMMA::Major::K,
        tensor_a.scalar_type(), tensor_b.scalar_type(),
        tensor_d.scalar_type(), false,
        device_runtime->get_num_sms());

    const int& load_block_m = SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m);
    const auto& tensor_map_a = make_tma_3d_desc(tensor_a, r, b, h,
                                                config.block_k, load_block_m, 1,
                                                tensor_a.stride(0), tensor_a.stride(1),
                                                config.smem_config.swizzle_a_mode);
    const int& load_block_n = SM90ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n);
    const auto& tensor_map_b = make_tma_3d_desc(tensor_b, r, d, h,
                                                config.block_k, load_block_n, 1,
                                                tensor_b.stride(1), tensor_b.stride(0),
                                                config.smem_config.swizzle_b_mode);
    const int& store_block_m = SM90ArchSpec::get_cd_store_block_m(config.block_m);
    const int& store_block_n = SM90ArchSpec::get_cd_store_block_n(config.block_n);
    const auto& tensor_map_cd = make_tma_3d_desc(tensor_d, d, b, h,
                                                store_block_n, store_block_m, 1,
                                                tensor_d.stride(0), tensor_d.stride(1),
                                                config.smem_config.swizzle_cd_mode);
    // Launch
    const SM90BF16GemmRuntime::Args& args = {
        .m = b, .n = d, .k = r,
        .num_groups = h,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto& code = SM90BF16GemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_bf16_bhr_hdr_bhd", code);
    SM90BF16GemmRuntime::launch(runtime, args);
}

static void sm90_bf16_bhd_hdr_bhr(const torch::Tensor& tensor_a,
                                  const torch::Tensor& tensor_b,
                                  const torch::Tensor& tensor_d,
                                  const int& b, const int& h, const int& r, const int& d,
                                  const std::string& compiled_dims = "nk") {
    const auto& config = get_best_config<SM90ArchSpec>(
        GemmType::Batched, KernelType::KernelNoSF,
        b, r, d, h, cute::UMMA::Major::K, cute::UMMA::Major::MN,
        tensor_a.scalar_type(), tensor_b.scalar_type(),
        tensor_d.scalar_type(), false,
        device_runtime->get_num_sms());

    const int& load_block_m = SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m);
    const auto& tensor_map_a = make_tma_3d_desc(tensor_a, d, b, h,
                                                config.block_k, load_block_m, 1,
                                                tensor_a.stride(0), tensor_a.stride(1),
                                                config.smem_config.swizzle_a_mode);
    const int& load_block_n = SM90ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n);
    const auto& tensor_map_b = make_tma_3d_desc(tensor_b, r, d, h,
                                                load_block_n, config.block_k, 1,
                                                tensor_b.stride(1), tensor_b.stride(0),
                                                config.smem_config.swizzle_b_mode);
    const int& store_block_m = SM90ArchSpec::get_cd_store_block_m(config.block_m);
    const int& store_block_n = SM90ArchSpec::get_cd_store_block_n(config.block_n);
    const auto& tensor_map_cd = make_tma_3d_desc(tensor_d, r, b, h,
                                                store_block_n, store_block_m, 1,
                                                tensor_d.stride(0), tensor_d.stride(1),
                                                config.smem_config.swizzle_cd_mode);
    // Launch
    const SM90BF16GemmRuntime::Args& args = {
        .m = b, .n = r, .k = d,
        .num_groups = h,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto& code = SM90BF16GemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_bf16_bhd_hdr_bhr", code);
    SM90BF16GemmRuntime::launch(runtime, args);
}

} // namespace deep_gemm
