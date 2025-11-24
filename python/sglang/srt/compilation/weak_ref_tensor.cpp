// Adapted from: https://github.com/vllm-project/vllm/blob/main/csrc/ops.h

#include <torch/extension.h>
#include <vector>

static at::Tensor weak_ref_tensor(at::Tensor &tensor) {
  TORCH_CHECK(tensor.is_cuda(), "weak_ref_tensor expects a CUDA tensor");

  void *data_ptr = tensor.data_ptr();
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();

  auto options = tensor.options();

  auto new_tensor = torch::from_blob(data_ptr, sizes, strides, options);

  return new_tensor;
}

TORCH_LIBRARY(jit_weak_ref_tensor, ops) {
  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(jit_weak_ref_tensor, CUDA, ops) {
  ops.impl("weak_ref_tensor", weak_ref_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
