#include <torch/extension.h>

#include "utils.hpp"

torch::Tensor warp_reduce_cuda(torch::Tensor input);

torch::Tensor warp_reduce(torch::Tensor input) {
  CHECK_CUDA_INPUT(input);
  return warp_reduce_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduce", &warp_reduce, "Warp Reduce (CUDA)");
}
