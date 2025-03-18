/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>

#include "device/fmha.hpp"
#include "kernel/sm100_fmha_mla_tma_warpspecialized.hpp"
#include "kernel/mla_options.hpp"

constexpr auto FP8 = at::ScalarType::Float8_e4m3fn;
constexpr auto HALF = at::ScalarType::Half;
constexpr auto FLOAT = at::ScalarType::Float;
constexpr auto INT = at::ScalarType::Int;

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, \
                cutlassGetStatusString(error));     \
  }

using namespace cute;
using namespace cutlass::fmha::kernel;

template <typename T, typename... KernelOptions>
struct MlaSm100 {
  using Element = T;
  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;
  using ElementOut = cutlass::half_t;

  using TileShape = Shape<_128, _128, Shape<_512, _64>>;
  using TileShapeH = cute::tuple_element_t<0, TileShape>;
  using TileShapeD = cute::tuple_element_t<2, TileShape>;

  using StrideQ = cute::tuple<int, _1, int>;  // H D B
  using StrideK = cute::tuple<int, _1, int>;  // K D B
  using StrideO = StrideQ;                    // H D B
  using StrideLSE = cute::tuple<_1, int>;     // H B

  static constexpr bool kIsPersistent = find_option_t<
      Tag::kIsPersistent, true_type, KernelOptions...>::value;
  using TileScheduler = std::conditional_t<
      kIsPersistent, cutlass::fmha::kernel::PersistentTileScheduler, 
      cutlass::fmha::kernel::IndividualTileScheduler>;

  using FmhaKernel =
      cutlass::fmha::kernel::Sm100FmhaMlaKernelTmaWarpspecialized<
          TileShape, Element, ElementAccumulatorQK, ElementOut,
          ElementAccumulatorPV, TileScheduler>;
  using Fmha = cutlass::fmha::device::FMHA<FmhaKernel>;
};


template <typename T>
typename T::Fmha::Arguments args_from_options(at::Tensor& out,
                                              at::Tensor& lse,
                                              at::Tensor const& q_absorbed,
                                              at::Tensor const& ckv_kpe_cache,
                                              at::Tensor const& seq_lens,
                                              at::Tensor const& page_table) {
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = q_absorbed.device().index();
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  int batches = q_absorbed.sizes()[0];
  int page_count_per_seq = page_table.sizes()[1];
  int page_count_total = ckv_kpe_cache.sizes()[0];
  int page_size = ckv_kpe_cache.sizes()[1];
  int max_seq_len = page_size * page_count_per_seq;
  using TileShapeH = typename T::TileShapeH;
  using TileShapeD = typename T::TileShapeD;
  auto problem_shape = cute::make_tuple(
      TileShapeH{}, max_seq_len, TileShapeD{}, batches);

  auto [H, K, D, B] = problem_shape;
  auto [D_latent, D_rope] = D;

  // the scale is based on the non-absorbed sizes, change as appropriate
  // we can't determine this parameter from the info we have, it's an input
  int D_non_latent = 128;
  float scale = 1.0 / sqrt(1.0 * (D_non_latent + D_rope));

  using StrideQ = typename T::StrideQ;
  using StrideK = typename T::StrideK;
  using StrideO = typename T::StrideO;
  using StrideLSE = typename T::StrideLSE;

  StrideQ stride_Q = cute::make_tuple(0 + D_latent + D_rope, _1{},
                                      H * (0 + D_latent + D_rope));
  StrideK stride_C = cute::make_tuple(0 + D_latent + D_rope, _1{},
                                      page_size * (D_latent + D_rope));
  StrideLSE stride_PT = cute::make_stride(_1{}, page_count_per_seq);
  StrideLSE stride_LSE = cute::make_tuple(_1{}, 0 + H);
  StrideO stride_O = cute::make_tuple(0 + D_latent, _1{}, 0 + H * D_latent);

  using Element = typename T::Element;
  using ElementOut = typename T::ElementOut;
  using ElementAccumulatorPV = typename T::ElementAccumulatorPV;
  auto Q_ptr = static_cast<Element*>(q_absorbed.data_ptr());
  auto C_ptr = static_cast<Element*>(ckv_kpe_cache.data_ptr());
  typename T::Fmha::Arguments arguments{
    problem_shape,
    { scale,
      Q_ptr, stride_Q,
      Q_ptr + D_latent, stride_Q,
      C_ptr, stride_C,
      C_ptr + D_latent, stride_C,
      static_cast<int*>(seq_lens.data_ptr()),
      static_cast<int*>(page_table.data_ptr()), stride_PT,
      page_count_total, page_size},
    { static_cast<ElementOut*>(out.data_ptr()), stride_O,
      // static_cast<ElementAccumulatorPV*>(lse.data_ptr()), stride_LSE},
      static_cast<ElementAccumulatorPV*>(nullptr), stride_LSE},
    hw_info
  };
  return arguments;
}

template <typename Element>
void runMla(at::Tensor& out,
            at::Tensor& lse,
            at::Tensor const& q_absorbed,
            at::Tensor const& ckv_kpe_cache,
            at::Tensor const& seq_lens,
            at::Tensor const& page_table,
            cudaStream_t stream) {
  using MlaSm100Type = MlaSm100<Element, Option<Tag::kIsPersistent, true_type>>;
  typename MlaSm100Type::Fmha fmha;
  auto arguments =
      args_from_options<MlaSm100Type>(out, lse, q_absorbed, ckv_kpe_cache,
                                      seq_lens, page_table);
  size_t workspace_size = MlaSm100Type::Fmha::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(q_absorbed.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(fmha.can_implement(arguments));

  CUTLASS_CHECK(fmha.initialize(arguments, workspace.data_ptr(), stream));

  CUTLASS_CHECK(fmha.run(arguments, workspace.data_ptr(), stream));
}

void cutlass_mla(torch::Tensor& out,
                 torch::Tensor& lse,
                 torch::Tensor const& q_absorbed,
                 torch::Tensor const& ckv_kpe_cache,
                 torch::Tensor const& seq_lens,
                 torch::Tensor const& page_table) {
  auto in_dtype = q_absorbed.dtype();
  at::cuda::CUDAGuard device_guard{(char)q_absorbed.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
                                  q_absorbed.get_device());
  if (in_dtype == at::ScalarType::Half) {
    runMla<cutlass::half_t>(
        out, lse, q_absorbed, ckv_kpe_cache, seq_lens, page_table, stream);
  } else if (in_dtype == at::ScalarType::Float8_e4m3fn) {
    runMla<cutlass::float_e4m3_t>(
        out, lse, q_absorbed, ckv_kpe_cache, seq_lens, page_table, stream);
  } else {
    TORCH_CHECK(false, "Unsupported input data type of MLA");
  }
}

torch::Tensor cutlass_mla_entry(torch::Tensor const& q_absorbed,
                                torch::Tensor const& ckv_kpe_cache,
                                torch::Tensor const& seq_lens,
                                torch::Tensor const& page_table) {
  const int D_latent = 512;
  const int D_rope = 64;
  auto out_type = HALF;

  TORCH_CHECK(q_absorbed.is_cuda(), "q_absorbed must be a CUDA tensor");
  TORCH_CHECK(q_absorbed.is_contiguous(), "q_absorbed must be contiguous");
  TORCH_CHECK(ckv_kpe_cache.is_cuda(), "ckv_kpe_cache must be a CUDA tensor");
  TORCH_CHECK(ckv_kpe_cache.is_contiguous(), "ckv_kpe_cache must be contiguous");


  TORCH_CHECK(q_absorbed.scalar_type() == FP8 || q_absorbed.scalar_type() == HALF,
              "Inconsistency of Tensor type for q_absorbed")
  TORCH_CHECK(ckv_kpe_cache.scalar_type() == q_absorbed.scalar_type(),
              "Inconsistency of Tensor type for ckv_kpe_cache")
  TORCH_CHECK(seq_lens.scalar_type() == INT,
              "Inconsistency of Tensor type for seq_lens")
  TORCH_CHECK(page_table.scalar_type() == INT,
              "Inconsistency of Tensor type for page_table")

  TORCH_CHECK(q_absorbed.dim() == 3 && ckv_kpe_cache.dim() == 3);
  auto B_q = q_absorbed.size(0);
  auto H = q_absorbed.size(1);
  // TODO(kaixih): should relax this when cutlass improves.
  TORCH_CHECK(H == 128, "The num_heads has to be 128 for now");
  auto D_q = q_absorbed.size(2);
  auto D_ckv = ckv_kpe_cache.size(2);
  TORCH_CHECK(D_q == D_ckv && D_q == D_latent + D_rope);

  auto out = torch::empty({B_q, H, D_latent},
                          torch::TensorOptions()
                              .dtype(out_type)
                              .device(q_absorbed.device()));
  // auto lse = torch::empty({B_q, H},
  auto lse = torch::empty({0},
                          torch::TensorOptions()
                              .dtype(FLOAT)
                              .device(q_absorbed.device()));
  cutlass_mla(out, lse, q_absorbed, ckv_kpe_cache, seq_lens, page_table);
  return out;
}
