#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <vector>

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

int get_m_alignment_for_contiguous_layout(){
    /*
    When we do a grouped GEMM in contiguous format, LHS are grouped into several batches along the M axis.
    Since we deal with exactly one sub-matrix of RHS for each GEMM block, batch sizes above should align well
        with GEMM block shape.
    
    Returns:
        Group-level alignment requirement for grouped contiguous layout, which is always 128.
    """
    */
    return 128;
}


bool is_tma_multicast_legal(int n, int block_n, int num_tma_multicast, int num_sms) {
    if (num_tma_multicast == 1) {
        return true;
    }
    return (n % (block_n * num_tma_multicast) == 0) && 
           (num_sms % num_tma_multicast == 0);
}


std::vector<int> get_best_configs(int m, int n, int k, int num_groups, int num_sms,
                      bool is_grouped_contiguous = false){
    std::vector<int> block_ms;
    if(!is_grouped_contiguous){
        // TODO: for some cases, smaller M block is better, add them into tuning space
        block_ms.push_back(m <= 64? 64 : 128);
    }
    else{
        block_ms.push_back(get_m_alignment_for_contiguous_layout());
    }

    std::vector<int> block_ns;
    for (int bn = 16; bn <= 128; bn += 8) {
        block_ns.push_back(bn);
    }

    auto fix_wave_saturate = [num_sms](int x) { return x == 0 ? num_sms : x; };
    auto get_num_waves = [=](int bm, int bn) {
        return ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms);
    };
    auto get_last_wave_util = [=, &fix_wave_saturate](int bm, int bn) {
        return fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms);
    };

    // Decide block sizes by waves
    int best_block_m = -1, best_block_n = -1;
    for (auto block_m : block_ms) {
        for (auto block_n : block_ns) {
            bool success = false;
            
            if (best_block_m != -1 || best_block_n != -1) {
                success = true;
            } else {
                const int num_waves = get_num_waves(block_m, block_n);
                const int best_num_waves = get_num_waves(best_block_m, best_block_n);
                
                if (num_waves < best_num_waves) {
                    success = true;
                } else if (num_waves == best_num_waves) {
                    const int util = get_last_wave_util(block_m, block_n);
                    const int best_util = get_last_wave_util(best_block_m, best_block_n);
                    
                    if (util > best_util) {
                        success = true;
                    } else if (util == best_util) {
                        if (block_m > best_block_m || 
                           (block_m == best_block_m && block_n < best_block_n)) {
                            success = true;
                        }
                    }
                }
            }
            
            if (success) {
                best_block_m = block_m;
                best_block_n = block_n;
            }
        }
    }
    TORCH_CHECK(best_block_m != -1 && best_block_n != -1);

    // Always pick the longest one
    // NOTES: for double B scales, the best number of stages may be reduced
    const int sm90_capacity = 232448;
    int best_num_stages = -1;
    int best_smem_size = -1;

    std::vector<int> num_stages_options;
    if (128 % best_block_n != 0) {
        num_stages_options = {6, 5, 4};
    } else {
        num_stages_options = {8, 7, 6, 5, 4};
    }

    for (auto num_stages : num_stages_options) {
        best_smem_size = get_smem_size(num_stages, k, best_block_m, best_block_n);
        if (best_smem_size <= sm90_capacity) {
            best_num_stages = num_stages;
            break;
        }
    }
    TORCH_CHECK(best_num_stages != -1);

    // Determine TMA multicast
    int best_num_tma_multicast = 1;
    if (m >= 1024 && 
        is_tma_multicast_legal(n, best_block_n, 2, num_sms) &&
        num_groups == 1) {
        best_num_tma_multicast = 2;
    }

    return {
        best_block_m,
        best_block_n,
        best_num_stages,
        best_num_tma_multicast,
        best_smem_size
    };
}


int get_tma_aligned_size(int x, int element_size){
    /*
    Global memory address of TMA must be 16-byte aligned.
    Since we use column-major layout for the LHS scaling tensor,
        the M-axis of the LHS scaling tensor needs to be padded to a multiple of 16 bytes.

    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.

    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    */
    int tma_alignment_bytes = 16;
    TORCH_CHECK(tma_alignment_bytes % element_size == 0);
    int alignment = tma_alignment_bytes / element_size;
    return ceil_div(x, alignment) * alignment;
}


torch::Tensor get_col_major_tma_aligned_tensor(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3);
    bool remove_dim = false;
    
    if (x.dim() == 2) {
        x = x.unsqueeze(0);
        remove_dim = true;
    }

    auto sizes = x.sizes();
    int b = sizes[0];
    int m = sizes[1];
    int n = sizes[2];
    int aligned_m = get_tma_aligned_size(m, x.element_size());

    // Check if the tensor is already in the desired layout
    if (x.stride(0) == aligned_m * n && x.stride(1) == 1 && x.stride(2) == aligned_m) {
        return remove_dim ? x.squeeze(0) : x;
    }

    // Allocate aligned tensor
    auto options = x.options();
    torch::Tensor aligned_x = torch::empty({b, n, aligned_m}, options).transpose(1, 2);

    // Copy data to aligned tensor
    aligned_x.index({"...", torch::indexing::Slice(0, m), "..."}) = x;
    aligned_x = aligned_x.index({"...", torch::indexing::Slice(0, m), "..."});

    return remove_dim ? aligned_x.squeeze(0) : aligned_x;
}

template<int N, int K, int BLOCK_M, int BLOCK_N, int kNumStages, int kNumTMAMulticast>
void gemm_fp8_fp8_bf16_nt_N_K(torch::Tensor& lhs, torch::Tensor& lhs_scales,
                              torch::Tensor& rhs, torch::Tensor& rhs_scales,
                              torch::Tensor& out, int m, at::cuda::CUDAStream stream,
                              int num_sms, int smem_size){
    // Make a templated GEMM
    using GemmType = Gemm<N, K, BLOCK_M, BLOCK_N, 128, 1, kNumStages, kNumTMAMulticast, GemmType::Normal>;

    // Launch kernel
    auto tma_a_desc = GemmType::make_2d_tma_a_desc(lhs.data_ptr<__nv_fp8_e4m3>(), m);
    auto tma_b_desc = GemmType::make_2d_tma_b_desc(rhs.data_ptr<__nv_fp8_e4m3>());
    auto tma_scales_a_desc = GemmType::make_2d_tma_scales_a_desc(lhs_scales.data_ptr<float>(), m);
    auto tma_d_desc = GemmType::make_2d_tma_d_desc(out.data_ptr<__nv_bfloat16>(), m);
    GemmType::run(
                out.data_ptr<__nv_bfloat16>(), rhs_scales.data_ptr<float>(), nullptr,
                m,
                tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
                stream, num_sms, smem_size);
}


void gemm_fp8_fp8_bf16_nt(torch::Tensor& lhs, torch::Tensor& lhs_scales,
                          torch::Tensor& rhs, torch::Tensor& rhs_scales,
                          torch::Tensor& out) {
    uint32_t m = lhs.size(0);
    uint32_t k = lhs.size(1);
    uint32_t n = rhs.size(0);
    uint32_t k_ = rhs.size(1);
    uint32_t m_ = out.size(0);
    uint32_t n_ = out.size(1);

    TORCH_CHECK(n % 64 == 0 && k % 128 == 0);

    // Type and shape checks
    TORCH_CHECK(m == m_ && n == n_ && k == k_);
    TORCH_CHECK(n > 0 && k > 0);
    TORCH_CHECK(lhs_scales.size(0) == m && lhs_scales.size(1) == (k + 127) / 128);
    TORCH_CHECK(rhs_scales.size(0) == (n + 127) / 128 && rhs_scales.size(1) == (k + 127) / 128);
    TORCH_CHECK(lhs.scalar_type() == torch::kFloat8_e4m3fn && lhs_scales.scalar_type() == torch::kFloat32);
    TORCH_CHECK(rhs.scalar_type() == torch::kFloat8_e4m3fn && rhs_scales.scalar_type() == torch::kFloat32);
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous() && out.is_contiguous());

    // LHS scales must be transposed for TMA load, but not for RHS scales
    // NOTES: `get_tma_aligned_lhs_scales` may launch a kernel if not processed by previous kernels
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales);
    TORCH_CHECK(rhs_scales.is_contiguous());

    if(m==0)
        return;

    constexpr auto BLOCK_M = 128;
    constexpr auto BLOCK_N = 128;
    constexpr auto kNumStages = 8;
    constexpr auto kNumTMAMulticast = 1;
    int num_sms = get_num_sms();
    const int smem_size = get_smem_size(kNumStages, k, BLOCK_M, BLOCK_N);
    // auto [BLOCK_M, BLOCK_N, kNumStages, kNumTMAMulticast, smem_size] = get_best_configs(m, n, k, 1, num_sms);
    auto stream = at::cuda::getCurrentCUDAStream(lhs.get_device());

    // dispatch
    // constexpr int TP = 16;
    // if (n == 36 && k == 7168)
    //     // q_proj for q: (512+64)/tp, 7168
    //     gemm_fp8_fp8_bf16_nt_N_K<36, 7168, BLOCK_M, BLOCK_N, kNumStages, kNumTMAMulticast>(
    //         lhs, lhs_scales, rhs, rhs_scales, out, m, stream, num_sms, smem_size
    //     );
    if (n == 1536 && k == 7168)
        // q_a_proj
        gemm_fp8_fp8_bf16_nt_N_K<1536, 7168, BLOCK_M, BLOCK_N, kNumStages, kNumTMAMulticast>(
            lhs, lhs_scales, rhs, rhs_scales, out, m, stream, num_sms, smem_size
        );
    else if(n == 1536 && k == 1536)
        // q_b_proj when q_lora: 24576/tp, 1536
        gemm_fp8_fp8_bf16_nt_N_K<1536, 7168, BLOCK_M, BLOCK_N, kNumStages, kNumTMAMulticast>(
            lhs, lhs_scales, rhs, rhs_scales, out, m, stream, num_sms, smem_size
        );
    else if(n == 24576 && k == 7168)
        // kv_a_proj_with_mqa for kv: (128 + 64) * 128, 7168
        gemm_fp8_fp8_bf16_nt_N_K<24576, 7168, BLOCK_M, BLOCK_N, kNumStages, kNumTMAMulticast>(
            lhs, lhs_scales, rhs, rhs_scales, out, m, stream, num_sms, smem_size
        );
    else if(n == 2048 && k == 512)
        // kv_b_proj for k: 128 * (128 + 128)/tp, 512
        gemm_fp8_fp8_bf16_nt_N_K<2048, 512, BLOCK_M, BLOCK_N, kNumStages, kNumTMAMulticast>(
            lhs, lhs_scales, rhs, rhs_scales, out, m, stream, num_sms, smem_size
        );
    else if(n == 7168 && k == 1024)
        // o_proj: 7168, 16384/tp
        gemm_fp8_fp8_bf16_nt_N_K<7168, 1024, BLOCK_M, BLOCK_N, kNumStages, kNumTMAMulticast>(
            lhs, lhs_scales, rhs, rhs_scales, out, m, stream, num_sms, smem_size
        );
    else if(n == 2304 && k == 7168)
        // gate_up_proj for FFN: 18432 * 2/tp, 7168
        gemm_fp8_fp8_bf16_nt_N_K<2304, 7168, BLOCK_M, BLOCK_N, kNumStages, kNumTMAMulticast>(
            lhs, lhs_scales, rhs, rhs_scales, out, m, stream, num_sms, smem_size
        );
    else if(n == 256 && k == 7168)
        // gate_up_proj for MoE: 4096/tp, 7168
        gemm_fp8_fp8_bf16_nt_N_K<256, 7168, BLOCK_M, BLOCK_N, kNumStages, kNumTMAMulticast>(
            lhs, lhs_scales, rhs, rhs_scales, out, m, stream, num_sms, smem_size
        );
    else if(n == 7168 && k == 1152)
        // down_proj for FFN: 7168, 18432/tp
        gemm_fp8_fp8_bf16_nt_N_K<7168, 1152, BLOCK_M, BLOCK_N, kNumStages, kNumTMAMulticast>(
            lhs, lhs_scales, rhs, rhs_scales, out, m, stream, num_sms, smem_size
        );
    else if(n == 7168 && k == 256)
        // down_proj for MoE: 7168, 2048/tp
        gemm_fp8_fp8_bf16_nt_N_K<7168, 256, BLOCK_M, BLOCK_N, kNumStages, kNumTMAMulticast>(
            lhs, lhs_scales, rhs, rhs_scales, out, m, stream, num_sms, smem_size
        );
}
