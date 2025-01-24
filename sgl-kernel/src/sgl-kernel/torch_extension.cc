#include <torch/script.h>
#include "sgl_kernels_ops.h"

TORCH_LIBRARY(sgl_kernels, m) {
    // trt_reduce
    m.def("init_custom_ar(int64_t rank_id, int64_t world_size, Tensor rank_data, int64_t[] buffers, int64_t[] tmp_result_buffers, int64_t[] barrier_in, int64_t[] barrier_out) -> int64_t");
    m.def("dispose(int64_t fa) -> ()");
    m.def("all_reduce(int64_t fa, Tensor inp, Tensor out) -> ()");
    m.def("get_graph_buffer_ipc_meta(int64_t fa) -> (int64_t[], int64_t[])");
    m.def("register_graph_buffers(int64_t fa, int64_t[][] handles, int64_t[][] offsets) -> ()");

    // moe_align_block_size
    m.def("moe_align_block_size(Tensor topk_ids, int64_t num_experts, int64_t block_size, Tensor sorted_token_ids, Tensor experts_ids, Tensor num_tokens_post_pad, Tensor token_cnts_buffer, Tensor cumsum_buffer) -> ()");

    // sampling_scaling_penalties
    m.def("sampling_scaling_penalties(Tensor logits, Tensor scaling_penalties) -> Tensor");

    // int8_scaled_mm
    m.def("int8_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype, Tensor? bias) -> Tensor");

    // lightning_attention_decode
    m.def("lightning_attention_decode(Tensor q, Tensor k, Tensor v, Tensor past_kv, Tensor slope, Tensor output, Tensor new_kv) -> ()");

    // rotary embedding
    m.def("rotary_embedding(Tensor positions, Tensor query, Tensor key, int64_t head_size, Tensor cos_sin_cache, bool is_neox) -> ()");

    // rms norm
    m.def("rmsnorm(Tensor output, Tensor input, Tensor weight, double eps, int64_t cuda_stream) -> ()");

    // fused rms norm
    m.def("fused_add_rmsnorm(Tensor input, Tensor residual, Tensor weight, double eps, int64_t cuda_stream) -> ()");

    // gemma rms norm
    m.def("gemma_rmsnorm(Tensor output, Tensor input, Tensor weight, double eps, int64_t cuda_stream) -> ()");

    // fused gemma rms norm
    m.def("gemma_fused_add_rmsnorm(Tensor input, Tensor residual, Tensor weight, double eps, int64_t cuda_stream) -> ()");

    // silu and mul
    m.def("silu_and_mul(Tensor out, Tensor input, int64_t cuda_stream) -> ()");

    // gelu tanh and mul
    m.def("gelu_tanh_and_mul(Tensor out, Tensor input, int64_t cuda_stream) -> ()");

    // gelu and mul
    m.def("gelu_and_mul(Tensor out, Tensor input, int64_t cuda_stream) -> ()");

    // bmm fp8
    m.def("bmm_fp8(Tensor A, Tensor B, Tensor D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, int64_t cublas_handle, int64_t cuda_stream) -> ()");

    // min p sampling from probs
    m.def("min_p_sampling_from_probs(Tensor probs, Tensor uniform_samples, Tensor samples, Tensor? maybe_min_p_arr, double min_p_val, bool deterministic, int64_t cuda_stream) -> ()");

    // top k renorm probs
    m.def("top_k_renorm_probs_wrapper(Tensor probs, Tensor renorm_probs, Tensor? maybe_top_k_arr, int64_t top_k_val, int64_t cuda_stream) -> ()");

    // top p renorm probs
    m.def("top_p_renorm_probs(Tensor probs, Tensor renorm_probs, Tensor? maybe_top_p_arr, double top_p_val, int64_t cuda_stream) -> ()");

    // top k top p sampling from probs
    m.def("top_k_top_p_sampling_from_probs(Tensor probs, Tensor uniform_samples, Tensor samples, Tensor success, Tensor? maybe_top_k_arr, double top_k_val, Tensor? maybe_top_p_arr, double top_p_val, bool deterministic, int64_t cuda_stream) -> ()");

    // top p sampling from probs
    m.def("top_p_sampling_from_probs(Tensor probs, Tensor uniform_samples, Tensor samples, Tensor success, Tensor? maybe_top_p_arr, float top_p_val, bool deterministic, int64_t cuda_stream) -> ()");
}

TORCH_LIBRARY_IMPL(sgl_kernels, CUDA, m) {
    m.impl("init_custom_ar", &init_custom_ar);
    m.impl("dispose", &dispose);
    m.impl("all_reduce", &all_reduce);
    m.impl("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
    m.impl("register_graph_buffers", &register_graph_buffers);
    m.impl("moe_align_block_size", &moe_align_block_size);
    m.impl("sampling_scaling_penalties", &sampling_scaling_penalties);
    m.impl("int8_scaled_mm", &int8_scaled_mm);
    m.impl("lightning_attention_decode", &lightning_attention_decode);
    m.impl("rotary_embedding", &rotary_embedding);
    m.impl("rmsnorm", &rmsnorm);
    m.impl("fused_add_rmsnorm", &fused_add_rmsnorm);
    m.impl("gemma_rmsnorm", &gemma_rmsnorm);
    m.impl("gemma_fused_add_rmsnorm", &gemma_fused_add_rmsnorm);
    m.impl("silu_and_mul", &silu_and_mul);
    m.impl("gelu_tanh_and_mul", &gelu_tanh_and_mul);
    m.impl("gelu_and_mul", &gelu_and_mul);
    m.impl("bmm_fp8", &bmm_fp8);
    m.impl("min_p_sampling_from_probs", &min_p_sampling_from_probs);
    m.impl("top_k_renorm_probs", &top_k_renorm_probs);
    m.impl("top_p_renorm_probs", &top_p_renorm_probs);
    m.impl("top_k_top_p_sampling_from_probs", &top_k_top_p_sampling_from_probs);
    m.impl("top_p_sampling_from_probs", &top_p_sampling_from_probs);
}
