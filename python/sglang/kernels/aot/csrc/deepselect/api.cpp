// Adapted from DeepSelect commit 382d62a (MIT License).
// Copyright (c) 2025 DeepSeek

#include <ATen/cuda/CUDAContext.h>
#include <Python.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cmath>
#include <exception>
#include <vector>

#include "cuda_kernels/config.h"
#include "cuda_kernels/v3_fp32/topk_select.h"
#include "structs.h"

namespace {

std::vector<int64_t> deepselect_compiled_architectures() {
  std::vector<int64_t> architectures;
#if DEEPSELECT_HAS_SM90
  architectures.push_back(90);
#endif
#if DEEPSELECT_HAS_SM100
  architectures.push_back(100);
#endif
#if DEEPSELECT_HAS_SM103
  architectures.push_back(103);
#endif
  return architectures;
}

bool deepselect_supports_device(const cudaDeviceProp& device_prop) {
  const int64_t compute_capability = device_prop.major * 10 + device_prop.minor;
#if DEEPSELECT_HAS_SM90
  if (compute_capability == 90) return true;
#endif
#if DEEPSELECT_HAS_SM100
  if (compute_capability == 100) return true;
#endif
#if DEEPSELECT_HAS_SM103
  if (compute_capability == 103) return true;
#endif
  return false;
}

void deepselect_topk_fp32(
    const torch::Tensor& input, torch::Tensor& output_value, torch::Tensor& output_index, int64_t topk) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
  TORCH_CHECK(input.stride(1) == 1, "input's last dimension must be contiguous");
  TORCH_CHECK(
      input.stride(0) * input.element_size() % INPUT_STRIDE_ALIGNMENT_REQUIREMENT == 0,
      "input row stride must be a multiple of 1024 bytes");
  TORCH_CHECK(input.size(1) < MAX_VOCAB_SIZE, "input width must be less than 2^23");
  TORCH_CHECK(topk > 0 && topk <= 4096, "topk must be in [1, 4096]");
  TORCH_CHECK(topk <= input.size(1), "topk cannot exceed input width");

  TORCH_CHECK(output_value.is_cuda() && output_index.is_cuda(), "outputs must be CUDA tensors");
  TORCH_CHECK(
      output_value.device() == input.device() && output_index.device() == input.device(),
      "input and outputs must be on the same device");
  TORCH_CHECK(output_value.scalar_type() == at::kFloat, "output_value must be float32");
  TORCH_CHECK(output_index.scalar_type() == at::kInt, "output_index must be int32");
  TORCH_CHECK(
      output_value.dim() == 2 && output_value.size(0) == input.size(0) && output_value.size(1) == topk,
      "output_value has an invalid shape");
  TORCH_CHECK(
      output_index.dim() == 2 && output_index.size(0) == input.size(0) && output_index.size(1) == topk,
      "output_index has an invalid shape");
  TORCH_CHECK(
      output_value.stride(1) == 1 && output_index.stride(1) == 1, "outputs' last dimensions must be contiguous");
  TORCH_CHECK(
      output_value.stride(0) * output_value.element_size() % OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT == 0,
      "output_value row stride must be a multiple of 32 bytes");
  TORCH_CHECK(
      output_index.stride(0) * output_index.element_size() % OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT == 0,
      "output_index row stride must be a multiple of 32 bytes");

  const auto* device_prop = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(device_prop != nullptr, "failed to query CUDA device properties");
  TORCH_CHECK(
      deepselect_supports_device(*device_prop),
      "deepselect_topk_fp32 was not compiled for SM",
      device_prop->major,
      device_prop->minor);
  const at::cuda::CUDAGuard device_guard(input.device());

  TopkSelectArgs args{
      static_cast<uint32_t>(input.size(0)),
      static_cast<uint32_t>(input.size(1)),
      static_cast<uint32_t>(topk),
      input.data_ptr(),
      output_value.data_ptr(),
      output_index.data_ptr(),
      nullptr,
      nullptr,
      nullptr,
      static_cast<uint64_t>(input.stride(0)),
      static_cast<uint64_t>(output_value.stride(0)),
      static_cast<uint64_t>(output_index.stride(0)),
      false,
      false,
      true,
      -1,
      -INFINITY,
      true,
      device_prop->sharedMemPerBlockOptin,
      at::cuda::getCurrentCUDAStream(input.get_device()).stream()};

  if (topk <= 512) {
    topk_select_fp32::run_topk_select_kernel<
        TopkSelectConfig<float, int32_t, false, false, true, 512, 512, 1, 8192, 4096, 3>>(args);
  } else if (topk <= 1024) {
    topk_select_fp32::run_topk_select_kernel<
        TopkSelectConfig<float, int32_t, false, false, true, 1024, 512, 1, 8192, 4096, 3>>(args);
  } else {
    topk_select_fp32::run_topk_select_kernel<
        TopkSelectConfig<float, int32_t, false, false, true, 4096, 256, 1, 4096, 4096, 3>>(args);
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("deepselect_topk_fp32(Tensor input, Tensor! values, Tensor! indices, int topk) -> ()");
  m.impl("deepselect_topk_fp32", torch::kCUDA, &deepselect_topk_fp32);
}

PyObject* get_supported_architectures(PyObject*, PyObject*) {
  const auto architectures = deepselect_compiled_architectures();
  PyObject* result = PyTuple_New(architectures.size());
  if (result == nullptr) {
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(architectures.size()); ++i) {
    PyObject* architecture = PyLong_FromLong(architectures[i]);
    if (architecture == nullptr || PyTuple_SetItem(result, i, architecture) < 0) {
      Py_XDECREF(architecture);
      Py_DECREF(result);
      return nullptr;
    }
  }
  return result;
}

PyObject* is_supported_device(PyObject*, PyObject* args) {
  int device_index;
  if (!PyArg_ParseTuple(args, "i", &device_index)) {
    return nullptr;
  }
  try {
    const auto* device_prop = at::cuda::getDeviceProperties(device_index);
    if (device_prop == nullptr) {
      Py_RETURN_FALSE;
    }
    return PyBool_FromLong(deepselect_supports_device(*device_prop));
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

PyMODINIT_FUNC PyInit_deepselect_ops() {
  static PyMethodDef methods[] = {
      {"get_supported_architectures",
       get_supported_architectures,
       METH_NOARGS,
       "Return the CUDA compute capabilities compiled into DeepSelect."},
      {"is_supported_device",
       is_supported_device,
       METH_VARARGS,
       "Return whether DeepSelect contains code for the CUDA device."},
      {nullptr, nullptr, 0, nullptr}};
  static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "deepselect_ops", nullptr, 0, methods};
  return PyModule_Create(&module);
}
