#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SMXXPagedMQALogitsMetadataRuntime final: public LaunchRuntime<SMXXPagedMQALogitsMetadataRuntime> {
public:
    struct Args {
        int aligned_batch_size;
        int split_kv;
        int num_sms;
        
        int batch_size;
        int next_n;
        bool is_context_lens_2d;
        int* context_lens;
        int* schedule_metadata;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        const auto& arch = device_runtime->get_arch(true);

        return fmt::format(R"(
#include <deep_gemm/impls/sm{}_fp8_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_paged_mqa_logits_metadata<
        {}, {}, {}
    >);
}};
)", arch, args.aligned_batch_size, args.split_kv, args.num_sms);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.batch_size,
            args.next_n,
            args.is_context_lens_2d,
            args.context_lens,
            args.schedule_metadata
        ));
    }
};

static void smxx_paged_mqa_logits_metadata(const torch::Tensor& context_lens,
                                           const torch::Tensor& schedule_metadata,
                                           const int& batch_size, const int& next_n,
                                           const int& block_kv, const int& num_sms,
                                           const bool& is_context_lens_2d) {
    constexpr int num_math_warpgroups = 4;
    constexpr int num_threads = 32;
    const int aligned_batch_size = align(batch_size, 32);
    const int split_kv = block_kv * num_math_warpgroups;
    
    // Calculate shared memory size
    const int smem_size = aligned_batch_size * static_cast<int>(sizeof(int));
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Launch
    const SMXXPagedMQALogitsMetadataRuntime::Args& args = {
        .aligned_batch_size = aligned_batch_size,
        .split_kv = split_kv,
        .num_sms = num_sms,
        .batch_size = batch_size,
        .next_n = next_n,
        .is_context_lens_2d = is_context_lens_2d,
        .context_lens = context_lens.data_ptr<int>(),
        .schedule_metadata = schedule_metadata.data_ptr<int>(),
        .launch_args = LaunchArgs(1, num_threads, smem_size)
    };
    const auto& code = SMXXPagedMQALogitsMetadataRuntime::generate(args);
    const auto& runtime = compiler->build("smxx_paged_mqa_logits_metadata", code);
    SMXXPagedMQALogitsMetadataRuntime::launch(runtime, args);
}

class SMXXFP8PagedMQALogitsRuntime final: public LaunchRuntime<SMXXFP8PagedMQALogitsRuntime> {
public:
    struct Args {
        int batch_size;
        int next_n;
        int num_heads;
        int head_dim;
        int block_kv;
        bool is_context_lens_2d;
        int block_table_stride;
        int logits_stride;

        int num_q_stages;
        int num_kv_stages;
        int split_kv;

        int* context_lens;
        float* logits;
        int* block_table;
        int* schedule_meta;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_kv_scales;
        CUtensorMap tensor_map_weights;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // TODO: optimize performance by tuning args
        // Block sizes are fixed in this kernel
        DG_HOST_ASSERT(128 % args.num_heads == 0);
        const auto& arch = device_runtime->get_arch(true);

        return fmt::format(R"(
#include <deep_gemm/impls/sm{}_fp8_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm{}_fp8_paged_mqa_logits<
        {}, {},
        {}, {},
        {},
        {}, {},
        {},
        {}, {}
    >);
}};
)", arch, arch,
    args.next_n, args.num_heads,
    args.head_dim, args.block_kv,
    args.is_context_lens_2d,
    args.num_q_stages, args.num_kv_stages,
    args.split_kv,
    args.num_specialized_threads, args.num_math_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.batch_size,
            static_cast<uint64_t>(args.logits_stride),
            static_cast<uint64_t>(args.block_table_stride),
            args.context_lens, args.logits,
            args.block_table, args.schedule_meta,
            args.tensor_map_q, args.tensor_map_kv,
            args.tensor_map_kv_scales, args.tensor_map_weights
        ));
    }
};

static void smxx_fp8_paged_mqa_logits(const torch::Tensor& q,
                                      const torch::Tensor& kv_cache,
                                      const torch::Tensor& kv_cache_scales,
                                      const torch::Tensor& weights,
                                      const torch::Tensor& context_lens,
                                      const torch::Tensor& logits,
                                      const torch::Tensor& block_table,
                                      const torch::Tensor& schedule_meta,
                                      const int& batch_size, const int& next_n,
                                      const int& num_heads, const int& head_dim,
                                      const int& num_kv_blocks, const int& block_kv,
                                      const bool& is_context_lens_2d,
                                      const int& kv_cache_stride_bytes,
                                      const int& logits_stride,
                                      const int& block_table_stride,
                                      const int& num_sms,
                                      const int& split_kv) {
    const int num_specialized_threads = 128;
    const int mma_m = (device_runtime->get_arch_major() == 10 ? 128 : 64);
    const int num_math_warp_groups = split_kv / mma_m;
    const int num_math_threads = num_math_warp_groups * 128;
    const int num_q_stages = 3, num_kv_stages = (device_runtime->get_arch_major() == 10 ? 4 : 3);
    DG_HOST_ASSERT(split_kv % mma_m == 0 and logits_stride % split_kv == 0);

    // Construct TMAs
    DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
    const auto& tensor_map_q = make_tma_2d_desc(q, head_dim, batch_size * next_n * num_heads,
                                                head_dim, next_n * num_heads, head_dim, head_dim);
    const auto& tensor_map_kv = make_tma_3d_desc(kv_cache, head_dim, block_kv, num_kv_blocks,
                                                 head_dim, block_kv, 1,
                                                 head_dim, kv_cache_stride_bytes, head_dim);
    // TODO: use 1D TMA
    const auto& tensor_map_kv_scales = make_tma_2d_desc(kv_cache_scales, block_kv, num_kv_blocks,
                                                        block_kv, 1, kv_cache_stride_bytes / static_cast<int>(sizeof(float)), 0);
    const auto& tensor_map_weights = make_tma_2d_desc(weights, next_n * num_heads, batch_size,
                                                      next_n * num_heads, 1, next_n * num_heads, 0);

    // Calculate shared memory size
    int smem_size = 0;
    if (device_runtime->get_arch_major() == 9) {
        const int swizzle_alignment = head_dim * 8;

        const int smem_q_size_per_stage = next_n * num_heads * head_dim * static_cast<int>(q.element_size());
        const int aligned_smem_weight_size_per_stage = align(next_n * num_heads * static_cast<int>(weights.element_size()), swizzle_alignment);
        const int smem_q_pipe_size = num_q_stages * (smem_q_size_per_stage + aligned_smem_weight_size_per_stage) + align(num_q_stages * 8 * 2, swizzle_alignment);

        const int smem_kv_size_per_stage = block_kv * head_dim * static_cast<int>(kv_cache.element_size());
        const int aligned_smem_kv_scale_size_per_stage = align(block_kv * static_cast<int>(kv_cache_scales.element_size()), swizzle_alignment);
        const int smem_kv_pipe_size = num_kv_stages * (smem_kv_size_per_stage + aligned_smem_kv_scale_size_per_stage) + align(num_kv_stages * 8 * 2, swizzle_alignment);

        // Allocate some shared memory for UMMA barriers and tensor memory pointer, although it is not used in SM90
        const int smem_umma_barriers = num_math_warp_groups * 2 * 8;
        const int smem_tmem_ptr = 4;

        smem_size = smem_q_pipe_size + num_math_warp_groups * smem_kv_pipe_size + smem_umma_barriers + smem_tmem_ptr;
        DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);
    } else {
        const int smem_q_size_per_stage = next_n * num_heads * head_dim * static_cast<int>(q.element_size());
        const int smem_kv_size_per_stage = split_kv * head_dim * static_cast<int>(kv_cache.element_size());
        const int smem_kv_scale_size_per_stage = split_kv * static_cast<int>(kv_cache_scales.element_size());
        const int smem_weight_size_per_stage = next_n * num_heads * static_cast<int>(weights.element_size());

        const int smem_barriers = (num_q_stages + num_kv_stages) * 2 * 8;
        const int smem_umma_barriers = num_math_warp_groups * 2 * 8;
        const int smem_tmem_ptr = 4;

        smem_size = num_q_stages * (smem_q_size_per_stage + smem_weight_size_per_stage) + 
                    num_kv_stages * (smem_kv_size_per_stage + smem_kv_scale_size_per_stage) + 
                    smem_barriers + smem_umma_barriers + smem_tmem_ptr;
        DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);
    }

    // Launch
    const SMXXFP8PagedMQALogitsRuntime::Args& args = {
        .batch_size = batch_size,
        .next_n = next_n,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .block_kv = block_kv,
        .is_context_lens_2d = is_context_lens_2d,
        .block_table_stride = block_table_stride,
        .logits_stride = logits_stride,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .split_kv = split_kv,
        .context_lens = context_lens.data_ptr<int>(),
        .logits = logits.data_ptr<float>(),
        .block_table = block_table.data_ptr<int>(),
        .schedule_meta = schedule_meta.data_ptr<int>(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_kv_scales = tensor_map_kv_scales,
        .tensor_map_weights = tensor_map_weights,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(num_sms,
                                  num_specialized_threads + num_math_threads,
                                  smem_size)
    };
    const auto& code = SMXXFP8PagedMQALogitsRuntime::generate(args);
    const auto& runtime = compiler->build("smxx_fp8_paged_mqa_logits", code);
    SMXXFP8PagedMQALogitsRuntime::launch(runtime, args);
}

} // namespace deep_gemm
