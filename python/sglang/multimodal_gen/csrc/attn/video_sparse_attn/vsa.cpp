#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>

#include <cuda_runtime.h>

#ifdef TK_COMPILE_BLOCK_SPARSE
extern std::vector<torch::Tensor> block_sparse_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor q2k_block_sparse_index, torch::Tensor q2k_block_sparse_num,
    torch::Tensor block_size);
extern std::vector<torch::Tensor> block_sparse_attention_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
    torch::Tensor l_vec, torch::Tensor og, torch::Tensor k2q_block_sparse_index,
    torch::Tensor k2q_block_sparse_num, torch::Tensor block_size);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Video Sparse Attention Kernels"; // optional module docstring

#ifdef TK_COMPILE_BLOCK_SPARSE
  m.def("block_sparse_fwd",
        torch::wrap_pybind_function(block_sparse_attention_forward),
        "block sparse attention");
  m.def("block_sparse_bwd",
        torch::wrap_pybind_function(block_sparse_attention_backward),
        "block sparse attention backward");
#endif
}
