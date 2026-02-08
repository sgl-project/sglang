#pragma once

#include <torch/python.h>

#include "../../jit/kernel_runtime.hpp"
#include "../../jit/compiler.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/layout.hpp"

namespace deep_gemm {

class TransposeFP32Runtime final: public LaunchRuntime<TransposeFP32Runtime> {
public:
    struct Args {
        int mn, sf_k;
        int block_mn;
        void *sf, *out;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&transpose_fp32<
        {}, {}, {}
    >);
}};
)", args.launch_args.num_threads, args.block_mn, args.sf_k);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.sf, args.out, static_cast<uint32_t>(args.mn)));
    }
};

class TransposeAndPackFP32IntoUE8M0Runtime final: public LaunchRuntime<TransposeAndPackFP32IntoUE8M0Runtime> {
public:
    struct Args {
        int mn, sf_k;
        int block_mn;
        void *sf, *out;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&transpose_and_pack_fp32_into_ue8m0<
        {}, {}, {}
    >);
}};
)", args.launch_args.num_threads, args.block_mn, args.sf_k);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.sf, args.out, static_cast<uint32_t>(args.mn)));
    }
};

class PackFP32IntoUE8M0Runtime final: public LaunchRuntime<PackFP32IntoUE8M0Runtime> {
public:
    struct Args {
        int num_groups, mn, sf_k, packed_sf_k;
        int block_mn, block_packed_sf_k;
        void *sf, *out, *ks;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&pack_fp32_into_ue8m0<
        {}, {}, {}, {}
    >);
}};
)", args.num_groups, args.launch_args.num_threads, args.block_mn, args.block_packed_sf_k);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.sf, args.out, args.ks, args.mn, args.sf_k, args.packed_sf_k));
    }
};

static std::tuple<int, int, int, int, int, torch::Tensor> preprocess_sf(const torch::Tensor& sf) {
    // NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    const auto& dim = sf.dim();
    DG_HOST_ASSERT(dim == 2 or dim == 3);
    DG_HOST_ASSERT(sf.scalar_type() == torch::kFloat);
    const auto& batched_sf = dim == 2 ? sf.unsqueeze(0) : sf;

    const auto& [num_groups, mn, sf_k] = get_shape<3>(batched_sf);
    const auto& tma_aligned_mn = get_tma_aligned_size(mn, static_cast<int>(sf.element_size()));
    return {dim, num_groups, mn, sf_k, tma_aligned_mn, batched_sf};
}

static torch::Tensor get_mn_major_tma_aligned_tensor(const torch::Tensor& sf) {
    const auto& [dim, num_groups, mn, sf_k, tma_aligned_mn, batched_sf] = preprocess_sf(sf);

    // The last kernel already gives a column-major TMA aligned layout
    if ((batched_sf.stride(0) == tma_aligned_mn * sf_k or dim == 2) and batched_sf.stride(1) == 1 and batched_sf.stride(2) == tma_aligned_mn)
        return (dim == 2) ? batched_sf.squeeze(0) : batched_sf;

    const auto& out = torch::empty_strided({num_groups, mn, sf_k},
                                           {tma_aligned_mn * sf_k, 1, tma_aligned_mn},
                                           batched_sf.options());

    if (not batched_sf.is_contiguous()) {
        // Fallback to PyTorch's slow copy if not contiguous
        // ReSharper disable once CppExpressionWithoutSideEffects
        out.copy_(batched_sf);
    } else {
        constexpr int block_mn = 64;
        constexpr int num_threads = 512;
        const auto& smem_size = block_mn * (sf_k + (1 - (sf_k % 2))) * static_cast<int>(sizeof(float));
        const TransposeFP32Runtime::Args& args = {
            .mn = mn,
            .sf_k = sf_k,
            .block_mn = block_mn,
            .sf = batched_sf.data_ptr(),
            .out = out.data_ptr(),
            .launch_args = LaunchArgs({ceil_div(mn, block_mn), num_groups}, num_threads, smem_size)
        };

        const auto& code = TransposeFP32Runtime::generate(args);
        const auto& runtime = compiler->build("transpose_fp32", code);
        TransposeFP32Runtime::launch(runtime, args);
    }
    return (dim == 2) ? out.squeeze(0) : out;
}

static torch::Tensor get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(const torch::Tensor& sf) {
    const auto& sf_reshaped = (sf.dim() == 2) ? sf.unsqueeze(0) : sf;

    // First, convert into UE8M0 `uint8_t`
    const auto& ue8m0_tensor = sf_reshaped.view(torch::kInt32).bitwise_right_shift(23).to(torch::kUInt8);

    // Second, make padded packed tensors
    const auto& [num_groups, mn, k] = get_shape<3>(sf_reshaped);
    const auto& aligned_mn = get_tma_aligned_size(mn, 4);
    const auto& aligned_k  = align(k, 4);

    const auto& options = torch::TensorOptions().device(sf.device()).dtype(torch::kUInt8);
    auto padded = torch::zeros({num_groups, aligned_mn, aligned_k}, options);
    // ReSharper disable once CppExpressionWithoutSideEffects
    padded.slice(1, 0, mn).slice(2, 0, k).copy_(ue8m0_tensor);
    padded = padded.view(-1).view(torch::kInt32).view({num_groups, aligned_mn, aligned_k / 4});

    // Finally, transpose
    auto out = torch::empty_strided({num_groups, aligned_mn, aligned_k / 4},
                                    {aligned_mn * (aligned_k / 4), 1, aligned_mn},
                                    at::TensorOptions().device(sf.device()).dtype(torch::kInt32));
    out = out.copy_(padded).slice(1, 0, mn);
    return (sf.dim() == 2) ? out.squeeze(0) : out;
}

static torch::Tensor get_mn_major_tma_aligned_packed_ue8m0_tensor(const torch::Tensor& sf) {
    const auto& [dim, num_groups, mn, sf_k, tma_aligned_mn, batched_sf] = preprocess_sf(sf);
    const auto& packed_sf_k = ceil_div(sf_k, 4);
    const auto& out = torch::empty_strided({num_groups, mn, packed_sf_k},
                                           {packed_sf_k * tma_aligned_mn, 1, tma_aligned_mn},
                                           at::TensorOptions().device(batched_sf.device()).dtype(torch::kInt));
    // Launch the kernel
    if (batched_sf.is_contiguous()) {
        if ((mn * sf_k) % 4 != 0 and num_groups > 1)
            return get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(sf);

        constexpr int block_mn = 48;
        constexpr int num_threads = 512;
        const TransposeAndPackFP32IntoUE8M0Runtime::Args& args = {
            .mn = mn,
            .sf_k = sf_k,
            .block_mn = block_mn,
            .sf = batched_sf.data_ptr(),
            .out = out.data_ptr(),
            .launch_args = LaunchArgs({ceil_div(mn, block_mn), num_groups}, num_threads, block_mn * sf_k * 4)
        };

        const auto& code = TransposeAndPackFP32IntoUE8M0Runtime::generate(args);
        const auto& runtime = compiler->build("transpose_and_pack_fp32_into_ue8m0", code);
        TransposeAndPackFP32IntoUE8M0Runtime::launch(runtime, args);
    } else {
        if (mn % 4 != 0 or num_groups > 1)
            return get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(sf);
        DG_HOST_ASSERT(batched_sf.stride(1) == 1 and batched_sf.stride(2) == mn);

        constexpr int block_mn = 128;
        constexpr int block_packed_sf_k = 16;
        constexpr int num_threads = 512;
        const PackFP32IntoUE8M0Runtime::Args& args = {
            .num_groups = 1,
            .mn = mn,
            .sf_k = sf_k,
            .packed_sf_k = packed_sf_k,
            .block_mn = block_mn,
            .block_packed_sf_k = block_packed_sf_k,
            .sf = batched_sf.data_ptr(),
            .out = out.data_ptr(),
            .ks = nullptr,
            .launch_args = LaunchArgs({ceil_div(mn, block_mn), ceil_div(packed_sf_k, block_packed_sf_k)}, num_threads)
        };

        const auto& code = PackFP32IntoUE8M0Runtime::generate(args);
        const auto& runtime = compiler->build("pack_fp32_into_ue8m0", code);
        PackFP32IntoUE8M0Runtime::launch(runtime, args);
    }
    return (dim == 2) ? out.squeeze(0) : out;
}

static torch::Tensor get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(const torch::Tensor& sf,
                                                                            const torch::Tensor& ks_tensor,
                                                                            const std::vector<int>& ks) {
    const auto& [sf_k, mn] = get_shape<2>(sf);
    const auto& num_groups = static_cast<int>(ks.size());

    int ref_sf_k = 0, packed_sf_k = 0;
    for (const auto& k: ks)
        ref_sf_k += ceil_div(k, 128), packed_sf_k += ceil_div(k, 512);
    DG_HOST_ASSERT(sf.is_contiguous());
    DG_HOST_ASSERT(ref_sf_k == sf_k);
    DG_HOST_ASSERT(num_groups <= 128 and mn % 4 == 0);

    const auto& out = torch::empty({packed_sf_k, mn}, at::TensorOptions().device(sf.device()).dtype(torch::kInt));

    constexpr int block_mn = 128;
    constexpr int block_packed_sf_k = 16;
    constexpr int num_threads = 512;
    const PackFP32IntoUE8M0Runtime::Args& args = {
        .num_groups = num_groups,
        .mn = mn,
        .sf_k = sf_k,
        .packed_sf_k = packed_sf_k,
        .block_mn = block_mn,
        .block_packed_sf_k = block_packed_sf_k,
        .sf = sf.data_ptr(),
        .out = out.data_ptr(),
        .ks = ks_tensor.data_ptr(),
        .launch_args = LaunchArgs({ceil_div(mn, block_mn), ceil_div(packed_sf_k, block_packed_sf_k)}, num_threads)
    };

    const auto& code = PackFP32IntoUE8M0Runtime::generate(args);
    const auto& runtime = compiler->build("pack_fp32_into_ue8m0", code);
    PackFP32IntoUE8M0Runtime::launch(runtime, args);
    return out;
}

} // namespace deep_gemm
