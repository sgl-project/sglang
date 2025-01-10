#include "utils.hpp"

// trt_reduce
using fptr_t = int64_t;
fptr_t init_custom_ar(int64_t rank_id, int64_t world_size, const std::vector<fptr_t>& buffers,
                      const std::vector<fptr_t>& barrier_in, const std::vector<fptr_t>& barrier_out);
void dispose(fptr_t _fa);
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out);

// moe_align_block_size
void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
                          torch::Tensor sorted_token_ids, torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad,
                          torch::Tensor token_cnts_buffer, torch::Tensor cumsum_buffer);
// moe_align_block_size stage functions
void moe_align_block_size_stage1(torch::Tensor topk_ids,
                                torch::Tensor token_cnts_buffer,
                                int64_t num_experts);

void moe_align_block_size_stage2(torch::Tensor token_cnts_buffer,
                                int64_t num_experts);

void moe_align_block_size_stage3(torch::Tensor topk_ids,
                                torch::Tensor sorted_token_ids,
                                torch::Tensor num_tokens_post_pad,
                                torch::Tensor token_cnts_buffer,
                                torch::Tensor cumsum_buffer,
                                int64_t num_experts,
                                int64_t block_size);

void moe_align_block_size_stage4(torch::Tensor topk_ids,
                                torch::Tensor sorted_token_ids,
                                torch::Tensor expert_ids,
                                torch::Tensor token_cnts_buffer,
                                torch::Tensor cumsum_buffer,
                                int64_t num_experts,
                                int64_t block_size,
                                int64_t total_blocks);

void moe_align_block_size_stage5(torch::Tensor topk_ids,
                                torch::Tensor sorted_token_ids,
                                torch::Tensor token_cnts_buffer,
                                torch::Tensor cumsum_buffer,
                                int64_t num_experts);

// int8_scaled_mm
torch::Tensor int8_scaled_mm(const torch::Tensor& mat_a, const torch::Tensor& mat_b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b, const torch::Dtype& out_dtype,
                             const c10::optional<torch::Tensor>& bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // trt_reduce
  m.def("init_custom_ar", &init_custom_ar, "init custom allreduce meta (CUDA)");
  m.def("dispose", &dispose, "dispose custom allreduce meta");
  m.def("all_reduce", &all_reduce, "custom all reduce (CUDA)");
  // moe_align_block_size
  m.def("moe_align_block_size", &moe_align_block_size, "MOE Align Block Size (CUDA)");
  // moe_align_block_size stages
  m.def("moe_align_block_size_stage1", &moe_align_block_size_stage1, 
        "MOE Align Block Size Stage 1: Count tokens per expert (CUDA)");
  m.def("moe_align_block_size_stage2", &moe_align_block_size_stage2,
        "MOE Align Block Size Stage 2: Compute prefix sum (CUDA)");
  m.def("moe_align_block_size_stage3", &moe_align_block_size_stage3,
        "MOE Align Block Size Stage 3: Compute total padded tokens (CUDA)");
  m.def("moe_align_block_size_stage4", &moe_align_block_size_stage4,
        "MOE Align Block Size Stage 4: Assign tokens to experts (CUDA)");
  m.def("moe_align_block_size_stage5", &moe_align_block_size_stage5,
        "MOE Align Block Size Stage 5: Assign tokens to experts (CUDA)");
  // int8_scaled_mm
  m.def("int8_scaled_mm", &int8_scaled_mm, "INT8 scaled matmul (CUDA)");
}
