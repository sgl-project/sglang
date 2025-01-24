#pragma once
#include <vector>

#include <torch/extension.h>
#include "utils.h"

// trt_reduce
using fptr_t = int64_t;
fptr_t init_custom_ar(int64_t rank_id, int64_t world_size, torch::Tensor& rank_data, const std::vector<fptr_t>& buffers,
                      const std::vector<fptr_t>& tmp_result_buffers, const std::vector<fptr_t>& barrier_in,
                      const std::vector<fptr_t>& barrier_out);
void dispose(fptr_t _fa);
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out);
std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa);
void register_graph_buffers(fptr_t _fa, const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);

// moe_align_block_size
void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
                          torch::Tensor sorted_token_ids, torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad,
                          torch::Tensor token_cnts_buffer, torch::Tensor cumsum_buffer);

// sampling_scaling_penalties
torch::Tensor sampling_scaling_penalties(const torch::Tensor& logits, const torch::Tensor& scaling_penalties);

// int8_scaled_mm
torch::Tensor int8_scaled_mm(const torch::Tensor& mat_a, const torch::Tensor& mat_b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b, const torch::Dtype& out_dtype,
                             const c10::optional<torch::Tensor>& bias);

// lightning_attention_decode
void lightning_attention_decode(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                                const torch::Tensor& past_kv, const torch::Tensor& slope, torch::Tensor output,
                                torch::Tensor new_kv);

// rotary embedding
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox);

// rms norm
void rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps, int64_t cuda_stream);

// fused rms norm
void fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps, int64_t cuda_stream);

// gemma rms norm
void gemma_rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps, int64_t cuda_stream);

// fused gemma rms norm
void gemma_fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps,
                             int64_t cuda_stream);

// silu and mul
void silu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);

// gelu tanh and mul
void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);

// gelu and mul
void gelu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);

// bmm fp8
void bmm_fp8(at::Tensor A, at::Tensor B, at::Tensor D, at::Tensor A_scale, at::Tensor B_scale,
             at::Tensor workspace_buffer, int64_t cublas_handle, int64_t cuda_stream);

// min p sampling from probs
void min_p_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples, at::Tensor samples,
                               std::optional<at::Tensor> maybe_min_p_arr, double min_p_val, bool deterministic,
                               int64_t cuda_stream);

// top k renorm probs
// patch here, cause flashinfer use unsigned int. but torch must use int64_t for extension.
void top_k_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_k_arr, unsigned int top_k_val,
                        int64_t cuda_stream); 

// patch here, cause flashinfer use unsigned int. but torch must use int64_t for extension.
// wrapper for binding
inline void top_k_renorm_probs_wrapper(at::Tensor probs, 
                               at::Tensor renorm_probs,
                               std::optional<at::Tensor> maybe_top_k_arr, 
                               int64_t top_k_val,
                               int64_t cuda_stream) {
    top_k_renorm_probs(probs, 
                       renorm_probs, 
                       maybe_top_k_arr,
                       static_cast<unsigned int>(top_k_val),
                       cuda_stream);
}

// top p renorm probs
void top_p_renorm_probs(at::Tensor probs, at::Tensor renorm_probs, std::optional<at::Tensor> maybe_top_p_arr,
                        double top_p_val, int64_t cuda_stream);

// top k top p sampling from probs
void top_k_top_p_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples, at::Tensor samples,
                                     at::Tensor success, std::optional<at::Tensor> maybe_top_k_arr, double top_k_val,
                                     std::optional<at::Tensor> maybe_top_p_arr, double top_p_val, bool deterministic,
                                     int64_t cuda_stream);

// top p sampling from probs
void top_p_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples, at::Tensor samples, at::Tensor success,
                               std::optional<at::Tensor> maybe_top_p_arr, double top_p_val, bool deterministic,
                               int64_t cuda_stream);
