#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

class SMXXCleanLogitsRuntime final: public LaunchRuntime<SMXXCleanLogitsRuntime> {
public:
    struct Args {
        int next_n;
        int seq_len;
        int seq_len_kv;
        uint64_t stride_logits;

        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        float* logits;

        int block_kv;
        int num_warps;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_clean_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_clean_logits<
        {}, {}, {}
    >);
}};
)", args.next_n, args.block_kv, args.num_warps);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.seq_len, args.seq_len_kv, static_cast<int64_t>(args.stride_logits),
            args.cu_seq_len_k_start, args.cu_seq_len_k_end, args.logits
        ));
    }
};

static void smxx_clean_logits(const torch::Tensor& logits,
                              const std::optional<torch::Tensor>& cu_seq_len_k_start,
                              const torch::Tensor& cu_seq_len_k_end,
                              const int& next_n,
                              const int& seq_len, const int& seq_len_kv,
                              const uint64_t &stride_logits) {
    const int block_kv = 8192;
    const int num_warps = 8;
    const int smem_size = block_kv * sizeof(float);

    // Launch
    const SMXXCleanLogitsRuntime::Args& args = {
        .next_n = next_n,
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .stride_logits = stride_logits,
        .cu_seq_len_k_start = cu_seq_len_k_start.has_value() ? cu_seq_len_k_start.value().data_ptr<int>() : nullptr,
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .logits = logits.data_ptr<float>(),
        .block_kv = block_kv,
        .num_warps = num_warps,
        .launch_args = LaunchArgs(device_runtime->get_num_sms(),
                                  num_warps * 32, smem_size)
    };
    const auto& code = SMXXCleanLogitsRuntime::generate(args);
    const auto& runtime = compiler->build("smxx_clean_logits", code);
    SMXXCleanLogitsRuntime::launch(runtime, args);
}

} // namespace deep_gemm
