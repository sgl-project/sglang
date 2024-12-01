#include <torch/extension.h>

torch::Tensor warp_reduce_cuda(torch::Tensor input);

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

torch::Tensor warp_reduce(torch::Tensor input) {
  CHECK_INPUT(input);
  return warp_reduce_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduce", &warp_reduce, "Warp Reduce (CUDA)");
}
