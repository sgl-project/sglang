#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <Python.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <vector>

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define REGISTER_EXTENSION(NAME)                                                                      \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                                                  \
  }

namespace machete {
/*
 * Machete
 */
std::vector<std::string> supported_schedules(
    at::ScalarType a_type,
    int64_t b_type_id,
    std::optional<at::ScalarType> maybe_group_scales_type,
    std::optional<at::ScalarType> maybe_group_zeros_type,
    std::optional<at::ScalarType> maybe_channel_scales_type,
    std::optional<at::ScalarType> maybe_token_scales_type,
    std::optional<at::ScalarType> maybe_out_type);

torch::Tensor
mm(torch::Tensor const& A,
   torch::Tensor const& B,
   torch::Tensor& D,
   int64_t b_type_id,
   std::optional<at::ScalarType> const& maybe_out_type,
   std::optional<torch::Tensor> const& maybe_group_scales,
   std::optional<torch::Tensor> const& maybe_group_zeros,
   std::optional<int64_t> maybe_group_size,
   std::optional<torch::Tensor> const& maybe_channel_scales,
   std::optional<torch::Tensor> const& maybe_token_scales,
   std::optional<std::string> maybe_schedule,
   std::optional<torch::Tensor> const& maybe_group_layout,
   std::optional<torch::Tensor> const& maybe_valid_len,
   std::optional<int64_t> group_stride);

torch::Tensor prepack_B(
    torch::Tensor const& B,
    at::ScalarType const& a_type,
    int64_t b_type_id,
    std::optional<at::ScalarType> const& maybe_group_scales_type);
}  // namespace machete
