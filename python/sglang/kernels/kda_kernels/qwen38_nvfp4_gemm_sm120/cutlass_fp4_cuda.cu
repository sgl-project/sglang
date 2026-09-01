/*
 * KDA provenance: this kernel was automatically optimized by the Humanize2
 * workflow (https://github.com/PolyArch/humanize) and Kernel Design Agents
 * (https://github.com/mit-han-lab/kernel-design-agents).
 * Source: https://github.com/BBuf/KDA-Pilot/pull/195 @
 * 516c976cee824a236679adf6eb525275a0a9a120.
 */
/* Copyright 2026 SGLang Team. All Rights Reserved.

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

// The kernel configuration and argument layout are adapted from SGLang
// PR #21314, specialized here to this task's BF16 large-M path.

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace qwen38 {

struct Sm120Fp4LargeConfig {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShapeMNK = Shape<_256, _128, _128>;
};

template <typename Config>
struct Fp4GemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = cutlass::bfloat16_t;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShapeMNK = typename Config::PerSmTileShapeMNK;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      PerSmTileShapeMNK,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      void,
      LayoutDTag,
      AlignmentD,
      ElementD,
      LayoutDTag,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutATag,
      AlignmentA,
      ElementB,
      LayoutBTag,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm>
cutlass::Status run(
    void* output,
    const void* input,
    const void* weight,
    const void* input_scales,
    const void* weight_scales,
    const void* alpha,
    int32_t m,
    int32_t n,
    int32_t k,
    cudaStream_t stream,
    DLDevice device) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using BlockScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  const auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  const auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  const auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  const auto layout_sfa = BlockScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  const auto layout_sfb = BlockScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {static_cast<ElementA const*>(input),
       stride_a,
       static_cast<ElementB const*>(weight),
       stride_b,
       static_cast<ElementSFA const*>(input_scales),
       layout_sfa,
       static_cast<ElementSFB const*>(weight_scales),
       layout_sfb},
      {{}, nullptr, stride_d, static_cast<ElementD*>(output), stride_d}};
  arguments.epilogue.thread.alpha_ptr = static_cast<float const*>(alpha);
  // Group nearby work tiles so the 18x40 down-projection grid reuses its
  // high-hit-rate operands more effectively across the final device waves.
  arguments.scheduler.max_swizzle_size = 4;

  Gemm gemm;
  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace_tensor = sglang::host::ffi::alloc_workspace_tensor(workspace_size, device);
  void* workspace = workspace_size == 0 ? nullptr : workspace_tensor.data_ptr();
  auto status = gemm.can_implement(arguments);
  if (status == cutlass::Status::kSuccess) {
    status = gemm.initialize(arguments, workspace, stream);
  }
  if (status == cutlass::Status::kSuccess) {
    status = gemm.run(arguments, workspace, stream);
  }
  return status;
}

}  // namespace qwen38

namespace sglang {

void cutlass_fp4_gemm(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView input_scales,
    tvm::ffi::TensorView weight_scales,
    tvm::ffi::TensorView alpha) {
  auto m = host::SymbolicSize{"m"};
  auto packed_k = host::SymbolicSize{"packed_k"};
  auto n = host::SymbolicSize{"n"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();

  host::TensorMatcher({m, packed_k}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(input);
  host::TensorMatcher({n, packed_k}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(weight);
  host::TensorMatcher({m, n}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(output);
  host::TensorMatcher({1}).with_dtype<float>().with_device<kDLCUDA>(device).verify(alpha);
  CHECK_HOST(input_scales.device().device_type == kDLCUDA) << "input_scales must be a CUDA tensor";
  CHECK_HOST(weight_scales.device().device_type == kDLCUDA) << "weight_scales must be a CUDA tensor";
  CHECK_HOST(input_scales.device().device_id == device.unwrap().device_id)
      << "input_scales must live on the same device as input";
  CHECK_HOST(weight_scales.device().device_id == device.unwrap().device_id)
      << "weight_scales must live on the same device as input";

  const int32_t m_i32 = static_cast<int32_t>(m.unwrap());
  const int32_t packed_k_i32 = static_cast<int32_t>(packed_k.unwrap());
  const int32_t n_i32 = static_cast<int32_t>(n.unwrap());
  const int32_t k_i32 = packed_k_i32 * 2;
  const cudaStream_t stream = host::LaunchKernel::resolve_device(device.unwrap());

  using Gemm = qwen38::Fp4GemmSm120<qwen38::Sm120Fp4LargeConfig>::Gemm;
  const auto status = qwen38::run<Gemm>(
      output.data_ptr(),
      input.data_ptr(),
      weight.data_ptr(),
      input_scales.data_ptr(),
      weight_scales.data_ptr(),
      alpha.data_ptr(),
      m_i32,
      n_i32,
      k_i32,
      stream,
      device.unwrap());
  CHECK_HOST(status == cutlass::Status::kSuccess)
      << "CUTLASS SM120 FP4 GEMM failed with status " << static_cast<int>(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cutlass_fp4_gemm, cutlass_fp4_gemm);

}  // namespace sglang
