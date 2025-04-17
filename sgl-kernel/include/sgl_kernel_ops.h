/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <Python.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <tuple>
#include <vector>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define REGISTER_EXTENSION(NAME)                                                                      \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                                                  \
  }

using fptr_t = int64_t;

/*
 * From csrc/allreduce
 */
#ifdef USE_ROCM
// ROCM custom allreduce
fptr_t init_custom_ar(
    torch::Tensor& meta,
    torch::Tensor& rank_data,
    const std::vector<std::string>& handles,
    const std::vector<int64_t>& offsets,
    int64_t rank,
    bool full_nvlink);
void all_reduce_reg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out);
void all_reduce_unreg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& reg_buffer, torch::Tensor& out);
void dispose(fptr_t _fa);
int64_t meta_size();
void register_buffer(
    fptr_t _fa, torch::Tensor& t, const std::vector<std::string>& handles, const std::vector<int64_t>& offsets);
std::tuple<torch::Tensor, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa);
void register_graph_buffers(
    fptr_t _fa, const std::vector<std::string>& handles, const std::vector<std::vector<int64_t>>& offsets);
torch::Tensor allocate_meta_buffer(int64_t size);
torch::Tensor get_meta_buffer_ipc_handle(torch::Tensor& inp);
#else
// custom allreduce
fptr_t
init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs, torch::Tensor& rank_data, int64_t rank, bool full_nvlink);
void dispose(fptr_t _fa);
int64_t meta_size();
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes);
std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa);
void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs);
void register_graph_buffers(
    fptr_t _fa, const std::vector<std::vector<int64_t>>& handles, const std::vector<std::vector<int64_t>>& offsets);
#endif

/*
 * From csrc/attention
 */
void lightning_attention_decode(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& past_kv,
    const torch::Tensor& slope,
    torch::Tensor output,
    torch::Tensor new_kv);
void merge_state(
    at::Tensor v_a, at::Tensor s_a, at::Tensor v_b, at::Tensor s_b, at::Tensor v_merged, at::Tensor s_merged);
void merge_state_v2(
    at::Tensor v_a, at::Tensor s_a, at::Tensor v_b, at::Tensor s_b, at::Tensor v_merged, at::Tensor s_merged);
void cutlass_mla_decode(
    torch::Tensor const& out,
    torch::Tensor const& q_nope_and_q_pe,
    torch::Tensor const& kv_c_and_k_pe_cache,
    torch::Tensor const& seq_lens,
    torch::Tensor const& page_table,
    torch::Tensor const& workspace);
int64_t cutlass_mla_get_workspace_size(int64_t max_seq_len, int64_t num_batches, int64_t sm_count = 0);
/*
 * From csrc/elementwise
 */
void rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps, int64_t cuda_stream);
void sgl_fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps);
void gemma_rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps, int64_t cuda_stream);
void gemma_fused_add_rmsnorm(
    at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps, int64_t cuda_stream);
void silu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);
void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);
void gelu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);
void apply_rope_pos_ids_cos_sin_cache(
    at::Tensor q,
    at::Tensor k,
    at::Tensor q_rope,
    at::Tensor k_rope,
    at::Tensor cos_sin_cache,
    at::Tensor pos_ids,
    bool interleave,
    int64_t cuda_stream);

/*
 * From csrc/gemm
 */
torch::Tensor awq_dequantize(torch::Tensor qweight, torch::Tensor scales, torch::Tensor qzeros);
void cutlass_scaled_fp4_mm(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha);
torch::Tensor int8_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype,
    const c10::optional<torch::Tensor>& bias);
torch::Tensor fp8_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype,
    const c10::optional<torch::Tensor>& bias);
torch::Tensor fp8_blockwise_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype);
void scaled_fp4_quant(
    torch::Tensor& output, torch::Tensor const& input, torch::Tensor& output_scale, torch::Tensor const& input_scale);
void sgl_per_token_group_quant_fp8(
    at::Tensor input,
    at::Tensor output_q,
    at::Tensor output_s,
    int64_t group_size,
    double eps,
    double fp8_min,
    double fp8_max);
void sgl_per_token_group_quant_int8(
    at::Tensor input,
    at::Tensor output_q,
    at::Tensor output_s,
    int64_t group_size,
    double eps,
    double int8_min,
    double int8_max);
void sgl_per_tensor_quant_fp8(at::Tensor input, at::Tensor output_q, at::Tensor output_s, bool is_static);
void sgl_per_token_quant_fp8(at::Tensor input, at::Tensor output_q, at::Tensor output_s);
void bmm_fp8(
    at::Tensor A,
    at::Tensor B,
    at::Tensor D,
    at::Tensor A_scale,
    at::Tensor B_scale,
    at::Tensor workspace_buffer,
    int64_t cublas_handle,
    int64_t cuda_stream);

/*
 * From csrc/moe
 */
void moe_align_block_size(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad,
    torch::Tensor token_cnts_buffer,
    torch::Tensor cumsum_buffer);

void topk_softmax(
    torch::Tensor& topk_weights,
    torch::Tensor& topk_indices,
    torch::Tensor& token_expert_indices,
    torch::Tensor& gating_output);

std::vector<at::Tensor>
moe_fused_gate(at::Tensor& input, at::Tensor& bias, int64_t num_expert_group, int64_t topk_group, int64_t topk);

/*
 * From csrc/speculative
 */
void tree_speculative_sampling_target_only(
    at::Tensor predicts,          // mutable
    at::Tensor accept_index,      // mutable
    at::Tensor accept_token_num,  // mutable
    at::Tensor candidates,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    at::Tensor uniform_samples,
    at::Tensor target_probs,
    at::Tensor draft_probs,
    double threshold_single = 1,
    double threshold_acc = 1,
    bool deterministic = true,
    int64_t cuda_stream = 0);

void verify_tree_greedy(
    at::Tensor predicts,          // mutable
    at::Tensor accept_index,      // mutable
    at::Tensor accept_token_num,  // mutable
    at::Tensor candidates,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    at::Tensor target_predict,
    int64_t cuda_stream = 0);

void build_tree_kernel_efficient(
    at::Tensor parent_list,
    at::Tensor selected_index,
    at::Tensor verified_seq_len,
    at::Tensor tree_mask,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num);

void segment_packbits(
    at::Tensor x, at::Tensor input_indptr, at::Tensor output_indptr, at::Tensor y, int64_t cuda_stream);

/*
 * From FlashInfer
 */
void min_p_sampling_from_probs(
    at::Tensor probs,
    at::Tensor uniform_samples,
    at::Tensor samples,
    std::optional<at::Tensor> maybe_min_p_arr,
    double min_p_val,
    bool deterministic,
    int64_t cuda_stream);

void top_k_renorm_probs(
    at::Tensor probs,
    at::Tensor renorm_probs,
    std::optional<at::Tensor> maybe_top_k_arr,
    int64_t top_k_val,
    int64_t cuda_stream);

void top_p_renorm_probs(
    at::Tensor probs,
    at::Tensor renorm_probs,
    std::optional<at::Tensor> maybe_top_p_arr,
    double top_p_val,
    int64_t cuda_stream);

void top_k_top_p_sampling_from_probs(
    at::Tensor probs,
    at::Tensor uniform_samples,
    at::Tensor samples,
    at::Tensor success,
    std::optional<at::Tensor> maybe_top_k_arr,
    double top_k_val,
    std::optional<at::Tensor> maybe_top_p_arr,
    double top_p_val,
    bool deterministic,
    int64_t cuda_stream);

void top_p_sampling_from_probs(
    at::Tensor probs,
    at::Tensor uniform_samples,
    at::Tensor samples,
    at::Tensor success,
    std::optional<at::Tensor> maybe_top_p_arr,
    double top_p_val,
    bool deterministic,
    int64_t cuda_stream);

namespace flash {
/*
 * From fa2 sparse
 */
std::vector<at::Tensor> mha_fwd_sparse(
    at::Tensor& q,        // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& k,  // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& v,  // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& block_count,
    const at::Tensor& block_offset,
    const at::Tensor& column_count,
    const at::Tensor& column_index,
    const std::optional<at::Tensor>& out_,           // batch_size x seqlen_q x num_heads x head_size
    const std::optional<at::Tensor>& alibi_slopes_,  // num_heads or batch_size x num_heads
    const double p_dropout,
    const double softmax_scale,
    bool is_causal,
    const double softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_);

std::vector<at::Tensor> mha_varlen_fwd_sparse(
    at::Tensor& q,        // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor& k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i.
    const at::Tensor& v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i.
    const at::Tensor& block_count,
    const at::Tensor& block_offset,
    const at::Tensor& column_count,
    const at::Tensor& column_index,
    const c10::optional<at::Tensor>& out_,  // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q,         // b+1
    const at::Tensor& cu_seqlens_k,         // b+1
    const c10::optional<at::Tensor>&
        seqused_k,  // b. If given, only this many elements of each batch element's keys are used.
    const c10::optional<at::Tensor>& alibi_slopes_,  // num_heads or b x num_heads
    int64_t max_seqlen_q,
    const int64_t max_seqlen_k,
    const double p_dropout,
    const double softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    const double softcap,
    const bool return_softmax,
    c10::optional<at::Generator> gen_);
}  // namespace flash
