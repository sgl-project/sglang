#include "utils.hpp"

// moe_align_block_size
void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
                          torch::Tensor sorted_token_ids, torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad,
                          torch::Tensor token_cnts_buffer, torch::Tensor cumsum_buffer);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // moe_align_block_size
  m.def("moe_align_block_size", &moe_align_block_size, "MOE Align Block Size (CUDA)");
}
