#include <vector>

#include "utils.hpp"

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

// rotary embedding
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox);

// rms norm
void rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps, int64_t cuda_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // trt_reduce
  m.def("init_custom_ar", &init_custom_ar, "init custom allreduce meta (CUDA)");
  m.def("dispose", &dispose, "dispose custom allreduce meta");
  m.def("all_reduce", &all_reduce, "custom all reduce (CUDA)");
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta, "custom all reduce get graph ipc meta");
  m.def("register_graph_buffers", &register_graph_buffers, "custom all reduce register graph buffers");
  // moe_align_block_size
  m.def("moe_align_block_size", &moe_align_block_size, "MOE Align Block Size (CUDA)");
  // sampling_scaling_penalties
  m.def("sampling_scaling_penalties", &sampling_scaling_penalties, "Sampling scaling penalties (CUDA)");
  // int8_scaled_mm
  m.def("int8_scaled_mm", &int8_scaled_mm, "INT8 scaled matmul (CUDA)");
  // rotary embedding
  m.def("rotary_embedding", &rotary_embedding, "Rotary Embedding (CUDA)");
  // rms norm
  m.def("rmsnorm", &rmsnorm, "RMSNorm (CUDA)");
}
