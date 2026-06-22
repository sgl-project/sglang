// SPDX-License-Identifier: Apache-2.0
// C++ direct-call shim into sgl-kernel's sage3_ops (exported symbols, default
// visibility). sage3 was contributed to sgl-kernel as a separate SM120a-gated
// sage3_ops library (cmake/sage3.cmake). The upstream SageAttention kernels
// (mha_fwd, scaled_fp4_quant / _permute / _trans) are now compiled ONCE inside
// sgl-kernel and exported with C++ linkage; this shim declares those symbols so
// OmniDreams' sage3_attention.cu glue can call them via a cross-.so link
// (same pattern as sgl_gemm_shim.cuh for fp8_scaled_mm).
//
// The singleview_loader links sage3_ops.so via -L/-l:/-Wl,-rpath (like
// common_ops.abi3.so for the fp8 GEMM). Declarations are in the GLOBAL
// namespace to match sgl-kernel's exported ::mha_fwd / ::scaled_fp4_quant_*.
#pragma once
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/types.h>
#include <vector>

// SageAttention-3 upstream symbols, now exported from sgl_kernel::sage3_ops.so.
// Signatures match the #include'd upstream definitions 1:1 (see sage3.cu in
// sgl-kernel/csrc/attention/).
void scaled_fp4_quant(torch::Tensor const& input,
                      torch::Tensor const& output,
                      torch::Tensor const& output_sf,
                      int tensor_layout);
void scaled_fp4_quant_permute(torch::Tensor const& input,
                              torch::Tensor const& output,
                              torch::Tensor const& output_sf,
                              int tensor_layout);
void scaled_fp4_quant_trans(torch::Tensor const& input,
                            torch::Tensor const& output,
                            torch::Tensor const& output_sf,
                            int tensor_layout);
std::vector<at::Tensor> mha_fwd(at::Tensor& q,
                                const at::Tensor& k,
                                const at::Tensor& v,
                                const at::Tensor& sfq,
                                const at::Tensor& sfk,
                                const at::Tensor& sfv,
                                const at::Tensor& delta_s,
                                int unpadded_k,
                                c10::optional<at::Tensor>& out_,
                                const float softmax_scale,
                                bool is_causal,
                                bool per_block_mean,
                                bool is_bf16);
