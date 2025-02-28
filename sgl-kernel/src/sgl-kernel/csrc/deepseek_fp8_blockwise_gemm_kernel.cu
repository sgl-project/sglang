#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cudaTypedefs.h>

#include "deep_seek_extensions/fp8_gemm.cuh"

using namespace deep_gemm;

int get_smem_size(int num_stages, int k, int block_m, int block_n, int block_k = 128):
    int smem_d = block_m * block_n * 2
    int smem_a_per_stage = block_m * block_k
    int smem_scales_a_per_stage = block_m * 4
    int smem_b_per_stage = block_n * block_k
    int smem_scales_b = ceil_div(k, block_k) * 4
    int smem_barrier = num_stages * 8 * 2

    int smem_size = 0
    int smem_size += smem_d
    int smem_size += num_stages * smem_a_per_stage
    int smem_size += num_stages * smem_scales_a_per_stage
    int smem_size += num_stages * smem_b_per_stage
    int smem_size += ceil_div(smem_scales_b * (1 if block_k % block_n == 0 else 2), 8) * 8
    int smem_size += smem_barrier
    return smem_size

void gemm_fp8_fp8_bf16_nt(const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
                        const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
                        torch::Tensor& out) {
    '''
    fork from deep_gemm/jit_kernels/gemm.py, translate py to cu
    '''
    constexpr auto m = lhs.size(0);
    constexpr auto k = lhs.size(1);
    constexpr auto n = rhs.size(0);
    constexpr auto k_ = rhs.size(1);
    constexpr auto m_ = out.size(0);
    constexpr auto n_ = out.size(1);
    constexpr auto N = n, K = k;
    //TODO: tune configs
    constexpr auto BLOCK_M = 128;
    constexpr auto BLOCK_N = 128;
    constexpr auto kNumStages = 8;
    constexpr auto kNumTMAMulticast = {NUM_TMA_MULTICAST};

    // Make a templated GEMM
    using GemmType = Gemm<N, K, BLOCK_M, BLOCK_N, 128, 1, kNumStages, kNumTMAMulticast, GemmType::Normal>;

    // Launch kernel
    auto tma_a_desc = GemmType::make_2d_tma_a_desc(lhs, m);
    auto tma_b_desc = GemmType::make_2d_tma_b_desc(rhs);
    auto tma_scales_a_desc = GemmType::make_2d_tma_scales_a_desc(lhs_scales, m);
    auto tma_d_desc = GemmType::make_2d_tma_d_desc(out, m);
    GemmType::run(out, rhs_scales, nullptr,
                m,
                tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
                stream, num_sms, smem_size);
}
