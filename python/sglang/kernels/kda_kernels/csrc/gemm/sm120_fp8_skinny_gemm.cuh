// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// KDA provenance: this kernel was automatically optimized by the Humanize2
// workflow (https://github.com/PolyArch/humanize) and Kernel Design Agents
// (https://github.com/mit-han-lab/kernel-design-agents).

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// This file integrates the SM120 FP8 skinny GEMM with SGLang's JIT kernel and
// KDA backend interfaces.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include <cuda_runtime.h>

namespace kda_sm120_fp8_skinny {

using namespace cute;
using namespace sglang;

using ElementInput = cutlass::float_e4m3_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using ClusterShape = Shape<_1, _1, _1>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;
constexpr int AlignmentInput = 16;
constexpr int AlignmentOutput = 8;

using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
using OutputScale = cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>;
using Multiply = cutlass::epilogue::fusion::
    Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using ScaleOutput = cutlass::epilogue::fusion::Sm90EVT<Multiply, OutputScale, Accum>;

template <int SwizzleBytes>
CUTLASS_HOST_DEVICE constexpr auto select_b_smem_layout() {
  if constexpr (SwizzleBytes == 128) {
    return cute::UMMA::Layout_K_SW128_Atom<uint8_t>{};
  } else if constexpr (SwizzleBytes == 64) {
    return cute::UMMA::Layout_K_SW64_Atom<uint8_t>{};
  } else if constexpr (SwizzleBytes == 32) {
    return cute::UMMA::Layout_K_SW32_Atom<uint8_t>{};
  } else {
    static_assert(SwizzleBytes == 0, "unsupported B shared-memory swizzle");
    return cute::UMMA::Layout_K_INTER_Atom<uint8_t>{};
  }
}

// FlashInfer's cuBLAS backend combines the static activation and weight scales
// before applying them to the accumulator. Use the same single FP32 multiply
// before BF16 conversion to minimize speculative target-logit drift.
class OneScaleLinearCombination {
 public:
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementOutput = kda_sm120_fp8_skinny::ElementOutput;
  using ElementC = ElementOutput;
  using ElementD = ElementOutput;
  static constexpr int kCount = 1;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;

  struct Params {
    const float* scale = nullptr;
  };

  CUTLASS_HOST_DEVICE
  explicit OneScaleLinearCombination(const Params& params) : scale_(*params.scale) {}

  CUTLASS_HOST_DEVICE bool is_source_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE FragmentOutput operator()(const FragmentAccumulator& accumulator) const {
    cutlass::NumericConverter<ElementOutput, ElementCompute> convert;
    FragmentOutput output;
    output[0] = convert(accumulator[0] * scale_);
    return output;
  }

  CUTLASS_HOST_DEVICE ElementD operator()(ElementAccumulator accumulator) const {
    cutlass::NumericConverter<ElementD, ElementCompute> convert;
    return convert(accumulator * scale_);
  }

  CUTLASS_HOST_DEVICE ElementD operator()(ElementAccumulator accumulator, ElementC) const {
    return (*this)(accumulator);
  }

 private:
  ElementCompute scale_;
};

// D.T = W @ A.T moves the tiny token dimension to the narrow N axis. A
// 16-wide token tile minimizes padded MMA work for every qualified M <= 9. The generic
// warp-specialized schedule needs only one MMA consumer group for this skinny
// tile; unlike ping-pong, it compiles to 256 threads without register spills.
template <
    int TileM,
    int TileK,
    int MainloopStages = 0,
    bool UseCacheHints = (TileM == 32 && TileK == 512),
    int BSmemSwizzleBytes = 128,
    int TmaLoopUnroll = 0>
struct Fp8Gemm {
  using TileShape = Shape<Int<TileM>, _16, Int<TileK>>;
  // Preserve the tuned mainloop's stage count by reserving the same carveout
  // used by the former TMA epilogue.  Only the executed epilogue changes.
  using StagingEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      Shape<Int<TileM>, _16>,
      ElementAccumulator,
      float,
      void,
      LayoutC,
      AlignmentOutput,
      ElementOutput,
      LayoutD,
      AlignmentOutput,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      ScaleOutput>::CollectiveOp;
  using DirectEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      void,
      cutlass::detail::TagToStrideC_t<LayoutD>,
      cutlass::detail::TagToStrideC_t<LayoutD>,
      OneScaleLinearCombination,
      cutlass::gemm::EpilogueDefault>;
  using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<DirectEpilogue>;
  using AutoStageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename StagingEpilogue::SharedStorage))>;
  using StageCount =
      cute::conditional_t<MainloopStages == 0, AutoStageCount, cutlass::gemm::collective::StageCount<MainloopStages>>;
  // The SM120 builder currently exposes ping-pong/cooperative schedules only.
  // Use it as a type/layout factory, then rebuild the collective with the
  // generic one-consumer-group dispatch policy below.
  using BuiltMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementInput,
      LayoutA,
      AlignmentInput,
      ElementInput,
      LayoutB,
      AlignmentInput,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      StageCount,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  // CUTLASS's generic SM120 builder selects ldmatrix.x4 for both operands.
  // A 16-wide B tile contains only enough values for ldmatrix.x2, so retain
  // every generated policy/layout while narrowing just that shared-to-register
  // copy atom.
  using MainloopDispatch = cutlass::gemm::MainloopSm120TmaWarpSpecialized<
      BuiltMainloop::DispatchPolicy::Stages,
      2,
      ClusterShape,
      cutlass::gemm::KernelTmaWarpSpecialized>;
  using SmemLayoutAtomB = decltype(select_b_smem_layout<BSmemSwizzleBytes>());
  using BaseCollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      MainloopDispatch,
      typename BuiltMainloop::TileShape,
      typename BuiltMainloop::ElementA,
      typename BuiltMainloop::StrideA,
      typename BuiltMainloop::ElementB,
      typename BuiltMainloop::StrideB,
      typename BuiltMainloop::TiledMma,
      typename BuiltMainloop::GmemTiledCopyA,
      typename BuiltMainloop::SmemLayoutAtomA,
      typename BuiltMainloop::SmemCopyAtomA,
      typename BuiltMainloop::TransformA,
      typename BuiltMainloop::GmemTiledCopyB,
      SmemLayoutAtomB,
      cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, uint8_t>,
      typename BuiltMainloop::TransformB>;
  // The generic GemmUniversal kernel calls the legacy mma overload. SM120's
  // collective also accepts a block coordinate for block-scaled variants; this
  // scalar-scale kernel does not use it, so bridge the two interfaces.
  struct CollectiveMainloop : BaseCollectiveMainloop {
    template <class TensorA, class TensorB, class KTileIterator, class BlockCoord>
    CUTLASS_DEVICE void load(
        typename BaseCollectiveMainloop::Params const& mainloop_params,
        typename BaseCollectiveMainloop::MainloopPipeline pipeline,
        typename BaseCollectiveMainloop::PipelineState smem_pipe_write,
        cute::tuple<TensorA, TensorB> const& load_inputs,
        BlockCoord const& block_coord,
        KTileIterator k_tile_iter,
        int k_tile_count,
        int thread_idx,
        uint32_t block_rank_in_cluster,
        typename BaseCollectiveMainloop::TensorStorage& shared_tensors) {
      if constexpr (!UseCacheHints) {
        BaseCollectiveMainloop::load(
            mainloop_params,
            pipeline,
            smem_pipe_write,
            load_inputs,
            block_coord,
            k_tile_iter,
            k_tile_count,
            thread_idx,
            block_rank_in_cluster,
            shared_tensors);
        return;
      }

      if (cute::elect_one_sync()) {
        auto shared_a = cute::make_tensor(
            cute::make_smem_ptr(shared_tensors.smem_A.data()), typename BaseCollectiveMainloop::SmemLayoutA{});
        auto shared_b = cute::make_tensor(
            cute::make_smem_ptr(shared_tensors.smem_B.data()), typename BaseCollectiveMainloop::SmemLayoutB{});

        // This kernel always uses a 1x1 cluster, so the TMA slice is fixed.
        auto block_tma_a = mainloop_params.tma_load_a.get_slice(0);
        auto block_tma_b = mainloop_params.tma_load_b.get_slice(0);
        auto [m_coord, n_coord, k_coord, l_coord] = block_coord;
        auto global_a = cute::get<0>(load_inputs)(cute::_, cute::_, m_coord, cute::_, l_coord);
        auto global_b = cute::get<1>(load_inputs)(cute::_, cute::_, n_coord, cute::_, l_coord);
        auto tiled_global_a = block_tma_a.partition_S(global_a);
        auto tiled_shared_a = block_tma_a.partition_D(shared_a);
        auto tiled_global_b = block_tma_b.partition_S(global_b);
        auto tiled_shared_b = block_tma_b.partition_D(shared_b);

        if constexpr (TmaLoopUnroll == 0) {
          CUTLASS_PRAGMA_NO_UNROLL
          for (; k_tile_count > 0; --k_tile_count) {
            pipeline.producer_acquire(smem_pipe_write);
            using Barrier = typename BaseCollectiveMainloop::MainloopPipeline::ProducerBarrierType;
            Barrier* barrier = pipeline.producer_get_barrier(smem_pipe_write);
            const int stage = smem_pipe_write.index();
            // The wide projection streams 80 MiB of weights exactly once.
            // Mark that traffic evict-first so it does not displace the
            // freshly quantized 45 KiB activation reused by every output tile.
            cute::copy(
                mainloop_params.tma_load_a.with(*barrier, 0, cute::TMA::CacheHintSm90::EVICT_FIRST),
                tiled_global_a(cute::_, cute::_, cute::_, *k_tile_iter),
                tiled_shared_a(cute::_, cute::_, cute::_, stage));
            cute::copy(
                mainloop_params.tma_load_b.with(*barrier, 0, cute::TMA::CacheHintSm90::EVICT_NORMAL),
                tiled_global_b(cute::_, cute::_, cute::_, *k_tile_iter),
                tiled_shared_b(cute::_, cute::_, cute::_, stage));
            ++k_tile_iter;
            ++smem_pipe_write;
          }
        } else {
          auto issue_stage = [&] {
            pipeline.producer_acquire(smem_pipe_write);
            using Barrier = typename BaseCollectiveMainloop::MainloopPipeline::ProducerBarrierType;
            Barrier* barrier = pipeline.producer_get_barrier(smem_pipe_write);
            const int stage = smem_pipe_write.index();
            cute::copy(
                mainloop_params.tma_load_a.with(*barrier, 0, cute::TMA::CacheHintSm90::EVICT_FIRST),
                tiled_global_a(cute::_, cute::_, cute::_, *k_tile_iter),
                tiled_shared_a(cute::_, cute::_, cute::_, stage));
            cute::copy(
                mainloop_params.tma_load_b.with(*barrier, 0, cute::TMA::CacheHintSm90::EVICT_NORMAL),
                tiled_global_b(cute::_, cute::_, cute::_, *k_tile_iter),
                tiled_shared_b(cute::_, cute::_, cute::_, stage));
            ++k_tile_iter;
            ++smem_pipe_write;
          };

          if constexpr (TmaLoopUnroll == 2) {
#pragma unroll 2
            for (; k_tile_count > 0; --k_tile_count) {
              issue_stage();
            }
          } else if constexpr (TmaLoopUnroll == 5) {
#pragma unroll 5
            for (; k_tile_count > 0; --k_tile_count) {
              issue_stage();
            }
          } else {
#pragma unroll 10
            for (; k_tile_count > 0; --k_tile_count) {
              issue_stage();
            }
          }
        }
      }
    }

    template <class FrgTensorC>
    CUTLASS_DEVICE void
    mma(typename BaseCollectiveMainloop::MainloopPipeline pipeline,
        typename BaseCollectiveMainloop::PipelineState state,
        FrgTensorC& accumulators,
        int k_tile_count,
        int thread_idx,
        typename BaseCollectiveMainloop::TensorStorage& shared_tensors,
        typename BaseCollectiveMainloop::Params const& params) {
      cute::Underscore unused_block_coord;
      BaseCollectiveMainloop::mma(
          pipeline, state, accumulators, k_tile_count, thread_idx, shared_tensors, params, unused_block_coord);
    }
  };
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <
    int TileM,
    int TileK,
    int MainloopStages = 0,
    bool UseCacheHints = (TileM == 32 && TileK == 512),
    int BSmemSwizzleBytes = 128,
    int TmaLoopUnroll = 0>
void run(
    void* output,
    const void* activation,
    const void* weight,
    const void* output_scale,
    int n,
    int k,
    cudaStream_t stream,
    int m = 9) {
  using GemmKernel =
      typename Fp8Gemm<TileM, TileK, MainloopStages, UseCacheHints, BSmemSwizzleBytes, TmaLoopUnroll>::Kernel;
  using Gemm = typename Fp8Gemm<TileM, TileK, MainloopStages, UseCacheHints, BSmemSwizzleBytes, TmaLoopUnroll>::Device;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(n, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(m, k, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, 1));

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementInput*>(weight), stride_a, static_cast<const ElementInput*>(activation), stride_b};
  auto* result = static_cast<ElementOutput*>(output);
  typename GemmKernel::EpilogueArguments epilogue_args{
      {static_cast<const float*>(output_scale)}, nullptr, stride_d, result, stride_d};

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, {n, m, k, 1}, mainloop_args, epilogue_args};

  Gemm gemm;
  CHECK_HOST(gemm.can_implement(arguments) == cutlass::Status::kSuccess)
      << "KDA SM120 FP8 GEMM cannot implement M=" << m << ", N=" << n << ", K=" << k;
  CHECK_HOST(gemm.initialize(arguments, nullptr, stream) == cutlass::Status::kSuccess)
      << "KDA SM120 FP8 GEMM initialization failed";
  CHECK_HOST(gemm.run(stream, nullptr, false) == cutlass::Status::kSuccess) << "KDA SM120 FP8 GEMM launch failed";
}

void launch_gemm(
    void* output,
    const void* activation,
    const void* weight,
    const void* output_scale,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  if (n >= 16384) {
    kda_sm120_fp8_skinny::run<32, 512>(output, activation, weight, output_scale, n, k, stream, m);
  } else if (n >= 8192) {
    kda_sm120_fp8_skinny::run<64, 256>(output, activation, weight, output_scale, n, k, stream, m);
  } else {
    kda_sm120_fp8_skinny::run<32, 256>(output, activation, weight, output_scale, n, k, stream, m);
  }
}

}  // namespace kda_sm120_fp8_skinny

namespace sglang {

/**
 * \brief SM120 per-tensor FP8 skinny GEMM for the KDA backend.
 *
 * SGLang performs its reference static quantization before this entry so the
 * full path remains within BF16 tolerance of the FlashInfer fallback.
 */
struct KdaSm120Fp8SkinnyGemm {
  static void run_quantized(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output_scale,
      const tvm::ffi::TensorView output) {
    using namespace host;

    auto M = SymbolicSize{"M"};
    auto K = SymbolicSize{"K"};
    auto N = SymbolicSize{"N"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, K}).with_dtype<fp8_e4m3_t>().with_device<kDLCUDA>(device).verify(input);
    TensorMatcher({K, N}).with_strides({1, K}).with_dtype<fp8_e4m3_t>().with_device<kDLCUDA>(device).verify(weight);
    TensorMatcher({}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(output_scale);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(output);

    const int m = static_cast<int>(M.unwrap());
    const int k = static_cast<int>(K.unwrap());
    const int n = static_cast<int>(N.unwrap());
    CHECK_HOST(m == 1 || m == 2 || m == 4 || m == 8 || m == 9) << "M must be one of {1, 2, 4, 8, 9}, got " << m;
    CHECK_HOST(k >= 256 && k % 256 == 0) << "K must be a positive multiple of 256, got " << k;
    CHECK_HOST(n >= 32 && n % 32 == 0) << "N must be a positive multiple of 32, got " << n;

    const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
    kda_sm120_fp8_skinny::launch_gemm(
        output.data_ptr(), input.data_ptr(), weight.data_ptr(), output_scale.data_ptr(), m, n, k, stream);
    RuntimeDeviceCheck();
  }
};

}  // namespace sglang
