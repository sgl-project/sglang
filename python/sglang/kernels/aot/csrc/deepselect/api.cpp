// Adapted from DeepSelect commit 382d62a (MIT License).
// Copyright (c) 2025 DeepSeek

#include <ATen/cuda/CUDAContext.h>
#include <Python.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <vector>

#include "cuda_kernels/config.h"
#include "cuda_kernels/v3/topk_select.h"
#include "cuda_kernels/v3_cluster/topk_select.h"
#include "cuda_kernels/v3_fp32/topk_select.h"
#include "dispatch_utils.h"
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
#if DEEPSELECT_HAS_SM90
  if (device_prop.major == 9 && device_prop.minor == 0) return true;
#endif
#if DEEPSELECT_HAS_SM100
  if (device_prop.major == 10 && device_prop.minor == 0) return true;
#endif
#if DEEPSELECT_HAS_SM103
  if (device_prop.major == 10 && device_prop.minor == 3) return true;
#endif
  return false;
}

void deepselect_topk(
    const torch::Tensor& input,
    int64_t topk,
    const std::optional<torch::Tensor>& begin,
    const std::optional<torch::Tensor>& end,
    bool sorted_value,
    bool sorted_index,
    const std::optional<torch::Tensor>& output_value,
    torch::Tensor& output_index,
    const std::optional<torch::Tensor>& output_idx_offset,
    int64_t idx_oob_fill_value,
    double value_oob_fill_value,
    bool return_value,
    bool abort_when_nan_found) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(
      input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kFloat,
      "input dtype must be bfloat16 or float32");
  TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
  TORCH_CHECK(input.stride(1) == 1, "input's last dimension must be contiguous");
  TORCH_CHECK(
      input.stride(0) * input.element_size() % INPUT_STRIDE_ALIGNMENT_REQUIREMENT == 0,
      "input row stride must be a multiple of 1024 bytes");
  TORCH_CHECK(input.size(1) < MAX_VOCAB_SIZE, "input width must be less than 2^23");
  TORCH_CHECK(topk > 0 && topk <= 4096, "topk must be in [1, 4096]");
  TORCH_CHECK(!begin.has_value(), "`begin` is not supported currently");
  TORCH_CHECK(!(sorted_value && !return_value), "`return_value` must be enabled when `sorted` is True");
  TORCH_CHECK(!(sorted_value && sorted_index), "`sorted` and `sorted_index` cannot both be True");
  TORCH_CHECK(!(sorted_value && input.scalar_type() == at::kBFloat16), "`sorted` is only supported for float32 input");
  TORCH_CHECK(
      output_index.scalar_type() == at::kInt || output_index.scalar_type() == at::kLong,
      "output_index dtype must be int32 or int64");
  TORCH_CHECK(
      idx_oob_fill_value >= std::numeric_limits<int32_t>::min() &&
          idx_oob_fill_value <= std::numeric_limits<int32_t>::max(),
      "idx_oob_fill_value must fit in int32");

  TORCH_CHECK(output_index.is_cuda(), "output_index must be a CUDA tensor");
  TORCH_CHECK(output_index.device() == input.device(), "input and output_index must be on the same device");
  TORCH_CHECK(
      output_index.dim() == 2 && output_index.size(0) == input.size(0) && output_index.size(1) == topk,
      "output_index has an invalid shape");
  TORCH_CHECK(output_index.stride(1) == 1, "output_index's last dimension must be contiguous");
  TORCH_CHECK(
      output_index.stride(0) * output_index.element_size() % OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT == 0,
      "output_index row stride must be a multiple of 32 bytes");

  auto check_optional_vector = [&](const std::optional<torch::Tensor>& tensor, const char* name, at::ScalarType dtype) {
    if (!tensor.has_value()) return;
    TORCH_CHECK(tensor->is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor->device() == input.device(), name, " must be on the input device");
    TORCH_CHECK(tensor->scalar_type() == dtype, name, " has an invalid dtype");
    TORCH_CHECK(tensor->dim() == 1 && tensor->size(0) == input.size(0), name, " must have shape [batch_size]");
    TORCH_CHECK(tensor->is_contiguous(), name, " must be contiguous");
  };
  check_optional_vector(end, "end", at::kInt);
  check_optional_vector(output_idx_offset, "output_idx_offset", at::kInt);
  if (return_value) {
    TORCH_CHECK(output_value.has_value(), "output_value must be provided when return_value is True");
  }
  if (output_value.has_value()) {
    TORCH_CHECK(output_value->is_cuda(), "output_value must be a CUDA tensor");
    TORCH_CHECK(output_value->device() == input.device(), "output_value must be on the input device");
    TORCH_CHECK(output_value->scalar_type() == input.scalar_type(), "output_value dtype must match input dtype");
    TORCH_CHECK(
        output_value->dim() == 2 && output_value->size(0) == input.size(0) && output_value->size(1) == topk,
        "output_value has an invalid shape");
    TORCH_CHECK(output_value->stride(1) == 1, "output_value's last dimension must be contiguous");
    TORCH_CHECK(
        output_value->stride(0) * output_value->element_size() % OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT == 0,
        "output_value row stride must be a multiple of 32 bytes");
  }

  const at::cuda::CUDAGuard device_guard(input.device());
  const auto* device_prop = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(device_prop != nullptr, "failed to query CUDA device properties");
  TORCH_CHECK(
      deepselect_supports_device(*device_prop),
      "DeepSelect was not compiled for SM",
      device_prop->major,
      device_prop->minor);
  if (input.size(0) == 0) return;

  TopkSelectArgs args{
      static_cast<uint32_t>(input.size(0)),
      static_cast<uint32_t>(input.size(1)),
      static_cast<uint32_t>(topk),
      input.data_ptr(),
      output_value.has_value() ? output_value->data_ptr() : nullptr,
      output_index.data_ptr(),
      nullptr,
      end.has_value() ? end->data_ptr<int>() : nullptr,
      output_idx_offset.has_value() ? output_idx_offset->data_ptr<int>() : nullptr,
      static_cast<uint64_t>(input.stride(0)),
      output_value.has_value() ? static_cast<uint64_t>(output_value->stride(0)) : 0,
      static_cast<uint64_t>(output_index.stride(0)),
      sorted_value,
      sorted_index,
      return_value,
      static_cast<int>(idx_oob_fill_value),
      static_cast<float>(value_oob_fill_value),
      abort_when_nan_found,
      device_prop->sharedMemPerBlockOptin,
      at::cuda::getCurrentCUDAStream(input.get_device()).stream()};

  const uint32_t num_sm = device_prop->multiProcessorCount;
  const uint32_t num_waves = (input.size(0) + num_sm - 1) / num_sm;
  const auto output_index_type = output_index.scalar_type();
  if (input.scalar_type() == at::kBFloat16) {
    const uint32_t cluster_min_vocab_size = device_prop->major == 9 ? 128u * 1024u : 512u * 1024u;
    if (input.size(0) <= 6 && input.size(1) >= cluster_min_vocab_size && topk <= 1024) {
      INTEGER_TYPE_SWITCH(output_index_type, OutIdxT, [&]() {
        BOOL_SWITCH(sorted_index, SORTED_INDEX, [&]() {
          BOOL_SWITCH(return_value, RETURN_VALUE, [&]() {
            if (device_prop->major == 9) {
              topk_select_bf16_cluster::run_topk_select_kernel<TopkSelectConfig<
                  nv_bfloat16,
                  OutIdxT,
                  false,
                  SORTED_INDEX,
                  RETURN_VALUE,
                  1024,
                  256,
                  1,
                  4096,
                  4096,
                  16,
                  512,
                  8>>(args);
            } else {
              topk_select_bf16_cluster::run_topk_select_kernel<TopkSelectConfig<
                  nv_bfloat16,
                  OutIdxT,
                  false,
                  SORTED_INDEX,
                  RETURN_VALUE,
                  1024,
                  256,
                  1,
                  4096,
                  4096,
                  16,
                  512,
                  16>>(args);
            }
          });
        });
      });
    } else {
      INTEGER_TYPE_SWITCH(output_index_type, OutIdxT, [&]() {
        BOOL_SWITCH(sorted_index, SORTED_INDEX, [&]() {
          BOOL_SWITCH(return_value, RETURN_VALUE, [&]() {
            auto dispatch = [&]<uint32_t MAX_TOPK>() {
              if (num_waves == 1) {
                topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<
                    nv_bfloat16,
                    OutIdxT,
                    false,
                    SORTED_INDEX,
                    RETURN_VALUE,
                    MAX_TOPK,
                    512,
                    1,
                    8192,
                    4096,
                    5>>(args);
              } else if constexpr (MAX_TOPK <= 512) {
                topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<
                    nv_bfloat16,
                    OutIdxT,
                    false,
                    SORTED_INDEX,
                    RETURN_VALUE,
                    MAX_TOPK,
                    256,
                    2,
                    4096,
                    4096,
                    4>>(args);
              } else {
                topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<
                    nv_bfloat16,
                    OutIdxT,
                    false,
                    SORTED_INDEX,
                    RETURN_VALUE,
                    MAX_TOPK,
                    256,
                    2,
                    4096,
                    4096,
                    3>>(args);
              }
            };
            if (topk <= 512) {
              dispatch.template operator()<512>();
            } else if (topk <= 1024) {
              dispatch.template operator()<1024>();
            } else {
              topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<
                  nv_bfloat16,
                  OutIdxT,
                  false,
                  SORTED_INDEX,
                  RETURN_VALUE,
                  4096,
                  512,
                  1,
                  8192,
                  4096,
                  3>>(args);
            }
          });
        });
      });
    }
  } else {
    INTEGER_TYPE_SWITCH(output_index_type, OutIdxT, [&]() {
      auto dispatch = [&]<bool SORTED_VALUE, bool SORTED_INDEX, bool RETURN_VALUE>() {
        auto launch = [&]<uint32_t MAX_TOPK, uint32_t NUM_THREADS, uint32_t B>() {
          topk_select_fp32::run_topk_select_kernel<TopkSelectConfig<
              float,
              OutIdxT,
              SORTED_VALUE,
              SORTED_INDEX,
              RETURN_VALUE,
              MAX_TOPK,
              NUM_THREADS,
              1,
              B,
              4096,
              3>>(args);
        };
        if (topk <= 512) {
          launch.template operator()<512, 512, 8192>();
        } else if (topk <= 1024) {
          launch.template operator()<1024, 512, 8192>();
        } else {
          launch.template operator()<4096, 256, 4096>();
        }
      };
      if (sorted_value) {
        dispatch.template operator()<true, false, true>();
      } else {
        BOOL_SWITCH(sorted_index, SORTED_INDEX, [&]() {
          BOOL_SWITCH(
              return_value, RETURN_VALUE, [&]() { dispatch.template operator()<false, SORTED_INDEX, RETURN_VALUE>(); });
        });
      }
    });
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "deepselect_topk(Tensor input, int topk, Tensor? begin, Tensor? end, bool sorted, bool sorted_index, "
      "Tensor(a!)? output_value, Tensor(b!) output_index, Tensor? output_idx_offset, int idx_oob_fill_value, "
      "float value_oob_fill_value, bool return_value, bool abort_when_nan_found) -> ()");
  m.impl("deepselect_topk", torch::kCUDA, &deepselect_topk);
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
