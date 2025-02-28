#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include "deepseek_extensions/fp8_gemm.cuh"

using namespace deep_gemm;

int get_num_sms(){
    /*
    Get the current maximum limit of SM count for all GEMM kernels to use.
    If the count is never specified, the function will return the number of device SMs.
    It is equivalent to torch.cuda.get_device_properties(device='cuda').multi_processor_count.

    Returns:
        Current maximum limit of SM count for all GEMM kernels to use.
    */
    int device_idx = 0;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
        return -1;
    }

    cudaDeviceProp properties;
    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
        return -1;
    }

    int _num_sms = static_cast<int>(properties.multiProcessorCount);
    return _num_sms;
}

int get_smem_size(int num_stages, int k, int block_m, int block_n, int block_k = 128){
    // fork from deep_gemm/jit_kernels/gemm.py, translate py to cu
    int smem_d = block_m * block_n * 2;
    int smem_a_per_stage = block_m * block_k;
    int smem_scales_a_per_stage = block_m * 4;
    int smem_b_per_stage = block_n * block_k;
    int smem_scales_b = ceil_div(k, block_k) * 4;
    int smem_barrier = num_stages * 8 * 2;

    int smem_size = 0;
    smem_size += smem_d;
    smem_size += num_stages * smem_a_per_stage;
    smem_size += num_stages * smem_scales_a_per_stage;
    smem_size += num_stages * smem_b_per_stage;
    if(block_k % block_n == 0)
        smem_size += ceil_div(smem_scales_b * 1, 8) * 8;
    else
        smem_size += ceil_div(smem_scales_b * 2, 8) * 8;
    smem_size += smem_barrier;
    return smem_size;
}

void gemm_fp8_fp8_bf16_nt(const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
                        const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
                        torch::Tensor& out) {
    // fork from deep_gemm/jit_kernels/gemm.py, translate py to cu
    const uint32_t m = lhs.size(0);
    const uint32_t k = lhs.size(1);
    const uint32_t n = rhs.size(0);
    const uint32_t k_ = rhs.size(1);
    const uint32_t m_ = out.size(0);
    const uint32_t n_ = out.size(1);
    //TODO(laixinn): tune configs
    constexpr auto BLOCK_M = 128;
    constexpr auto BLOCK_N = 128;
    constexpr auto kNumStages = 8;
    constexpr auto kNumTMAMulticast = 1;

    const int smem_size = get_smem_size(kNumStages, k, BLOCK_M, BLOCK_N);
    auto stream = at::cuda::getCurrentCUDAStream(lhs.get_device());
    int num_sms = get_num_sms();

    TORCH_CHECK(n % 64 == 0 && k % 128 == 0)

    // Type and shape checks
    TORCH_CHECK(m == m_ && n == n_ && k == k_);
    TORCH_CHECK(n > 0 && k > 0);
    TORCH_CHECK(lhs_scales.size(0) == m && lhs_scales.size(1) == (k + 127) / 128);
    TORCH_CHECK(rhs_scales.size(0) == (n + 127) / 128 && rhs_scales.size(1) == (k + 127) / 128);
    TORCH_CHECK(lhs.scalar_type() == torch::kFloat8_e4m3fn && lhs_scales.scalar_type() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(rhs.scalar_type() == torch::kFloat8_e4m3fn && rhs_scales.scalar_type() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous() && out.is_contiguous());

    // Make a templated GEMM
    using GemmType = Gemm<BLOCK_M, BLOCK_N, 128, 1, kNumStages, kNumTMAMulticast, GemmType::Normal>(n, k);

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
