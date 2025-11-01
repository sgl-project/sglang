#include <torch/extension.h>
#include <torch_npu/csrc/aten/common/from_blob.h>
#include <vector>

torch::Tensor weak_ref_tensor(torch::Tensor &tensor) {
  if (!tensor.is_privateuseone()) {
    throw std::runtime_error("Tensor must be on NPU device");
  }
  // Get the raw data pointer
  void *data_ptr = tensor.data_ptr();
  // Get tensor sizes and strides
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();
  // Get tensor options (dtype, device)
  auto options = tensor.options();
  // Create a new tensor from the raw data pointer
  auto new_tensor =
      at_npu::native::from_blob(data_ptr, sizes, strides, options);
  return new_tensor;
}

TORCH_LIBRARY(jit_weak_ref_tensor_npu, ops) {
  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
  ops.impl("weak_ref_tensor", torch::kPrivateUse1, weak_ref_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
