#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>

#include <cuda_runtime.h>

#ifdef TK_COMPILE_ST_ATTN
extern torch::Tensor sta_forward(torch::Tensor q, torch::Tensor k,
                                 torch::Tensor v, torch::Tensor o,
                                 int kernel_t_size, int kernel_w_size,
                                 int kernel_h_size, int text_length,
                                 bool process_text, bool has_text,
                                 int kernel_aspect_ratio_flag);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Sliding Block Attention Kernels"; // optional module docstring

#ifdef TK_COMPILE_ST_ATTN
  m.def("sta_fwd", torch::wrap_pybind_function(sta_forward),
        "sliding tile attention, assuming tile size is (6,8,8)");
#endif
}
