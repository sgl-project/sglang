/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/numeric_conversion.h"

namespace cutlass::epilogue::collective {

// Experimental Humming GEMM1 epilogue. W1 rows must be physically ordered as
// [gate0, up0, gate1, up1, ...]. Each CTA produces one 128-channel physical
// tile, writes 64 rounded BF16 SwiGLU values, and publishes its tile-local amax.
// The last CTA arriving for a token row finalizes the complete row to E4M3.
template <
    class CtaTileShapeMNK_,
    class ElementC_,
    class StrideC_,
    class ElementD_,
    class StrideD_,
    class ElementAccumulator_,
    class ElementScalar_>
class SmemEpilogueArraySwiGLUQuant {
 public:
  using CtaTileShapeMNK = CtaTileShapeMNK_;
  using EpilogueSchedule = PtrArrayNoSmemWarpSpecialized;
  using DispatchPolicy = EpilogueSchedule;
  using ElementOutput = ElementD_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementAccumulator;
  using ElementScalar = ElementScalar_;
  using ElementC = ElementC_;
  using StrideC = StrideC_;
  using InternalStrideC = cute::remove_pointer_t<StrideC>;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using InternalStrideD = cute::remove_pointer_t<StrideD>;
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  struct ThreadEpilogueOp {
    using ElementOutput = ElementD_;
    using ElementD = ElementD_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementAccumulator_;
  };

  static constexpr int TileM = cute::size<0>(CtaTileShapeMNK{});
  static constexpr int TileN = cute::size<1>(CtaTileShapeMNK{});
  static constexpr int TileElements = TileM * TileN;
  static constexpr int LogicalTileM = TileM / 2;
  static constexpr int ElementsPerVector = 128 / cute::sizeof_bits_v<ElementD>;
  static constexpr int kOutputAlignment = ElementsPerVector;

  static_assert(TileM >= 64 && TileM % 2 == 0, "The fused GEMM1 epilogue requires an even channel tile.");
  static_assert(TileN % 8 == 0);
  static_assert(cute::is_same_v<decltype(cute::get<0>(InternalStrideD{})), cute::Int<1>>);

  struct SharedStorage {
    alignas(16) ElementD physical_output[TileElements];
    alignas(16) ElementCompute token_scale[TileN > 8 ? TileN : 1];
  };
  using TensorMapStorage = SharedStorage;

  struct ThreadArguments {
    ElementScalar token_scale_default = ElementScalar(1);
    ElementScalar const* const* token_scale_ptr_array = nullptr;
    cutlass::float_e4m3_t* output_fp8 = nullptr;
    float* output_scale = nullptr;
    float* row_amax = nullptr;
    int32_t* row_arrivals = nullptr;
    float const* expert_residual = nullptr;
    int64_t total_rows = 0;
    int64_t logical_channel_extent = 0;
    float swiglu_limit = 0.0f;
    bool has_swiglu_limit = false;
  };

  struct Arguments {
    ThreadArguments thread{};
    ElementC const** ptr_C = nullptr;
    StrideC dC{};
    ElementD** ptr_D = nullptr;
    StrideD dD{};
    ElementD* output_base = nullptr;
    int64_t output_channel_extent = 0;
    int64_t output_row_stride = 0;
    ElementCompute beta = ElementCompute(0);
  };

  struct Params {
    ThreadArguments thread{};
    ElementD** ptr_D = nullptr;
    StrideD dD{};
    ElementD* output_base = nullptr;
    int64_t output_channel_extent = 0;
    int64_t output_row_stride = 0;
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return {args.thread, args.ptr_D, args.dD, args.output_base, args.output_channel_extent, args.output_row_stride};
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&, int) {
    return 0;
  }

  template <class ProblemShape>
  static Status
  initialize_workspace(ProblemShape const&, Arguments const&, void*, cudaStream_t, CudaHostAdapter* = nullptr) {
    return Status::kSuccess;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape problem_shapes, Arguments const& args) {
    bool valid = args.ptr_C == nullptr && args.beta == ElementCompute(0) && args.ptr_D != nullptr &&
                 args.output_base != nullptr && args.thread.output_fp8 != nullptr &&
                 args.thread.output_scale != nullptr && args.thread.row_amax != nullptr &&
                 args.thread.row_arrivals != nullptr && args.thread.expert_residual != nullptr &&
                 args.output_channel_extent > 0 && args.output_channel_extent == args.thread.logical_channel_extent &&
                 args.output_row_stride == args.output_channel_extent &&
                 args.thread.logical_channel_extent % LogicalTileM == 0;
    if (problem_shapes.is_host_problem_shape_available()) {
      for (int group = 0; group < problem_shapes.groups(); ++group) {
        auto problem = problem_shapes.get_host_problem_shape(group);
        valid = valid && int64_t(cute::get<0>(problem)) == 2 * args.thread.logical_channel_extent;
      }
    }
    return valid;
  }

  CUTLASS_HOST_DEVICE
  explicit SmemEpilogueArraySwiGLUQuant(Params const& params) : params_(params) {}

  CUTLASS_DEVICE
  bool is_source_needed() const {
    return false;
  }

  CUTLASS_DEVICE
  static float warp_max(float value) {
    for (int delta = 16; delta > 0; delta /= 2) {
      value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, delta));
    }
    return value;
  }

  CUTLASS_DEVICE
  float swiglu(ElementD gate_bf16, ElementD up_bf16) const {
    float gate = static_cast<float>(gate_bf16);
    float up = static_cast<float>(up_bf16);
    if (params_.thread.has_swiglu_limit) {
      gate = fminf(gate, params_.thread.swiglu_limit);
      up = fmaxf(fminf(up, params_.thread.swiglu_limit), -params_.thread.swiglu_limit);
    }
    float const silu_gate = gate / (1.0f + expf(-gate));
    // Match the unfused path: round the FP32 product once to BF16 before amax.
    return static_cast<float>(ElementD(silu_gate * up));
  }

  template <
      class ProblemShapeMNKL,
      class BlockShapeMNK,
      class BlockCoordMNKL,
      class FrgEngine,
      class FrgLayout,
      class TiledMma,
      class ResidueMNK>
  CUTLASS_DEVICE void operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK block_shape_mnk,
      BlockCoordMNKL block_coord_mnkl,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      ResidueMNK,
      int thread_idx,
      char* shared_storage_ptr) {
    using namespace cute;
    static_assert(is_same_v<BlockShapeMNK, CtaTileShapeMNK>);

    auto physical_m = get<0>(problem_shape_mnkl);
    auto tokens = get<1>(problem_shape_mnkl);
    auto [m_coord, n_coord, k_coord, group_coord] = block_coord_mnkl;
    int const tile_m_origin = int(m_coord) * TileM;
    int const tile_n_origin = int(n_coord) * TileN;

    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor output_coordinates = make_identity_tensor(make_shape(physical_m, tokens));
    Tensor tile_coordinates = local_tile(output_coordinates, take<0, 2>(block_shape_mnk), make_coord(m_coord, n_coord));
    Tensor thread_coordinates = thread_mma.partition_C(tile_coordinates);
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(shared_storage_ptr);
    auto synchronize_consumer_warpgroup = [&]() {
      cutlass::arch::NamedBarrier::sync(size(TiledMma{}), cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    };

    if constexpr (TileN > 8) {
      if (thread_idx < TileN) {
        ElementScalar const* token_scales =
            params_.thread.token_scale_ptr_array ? params_.thread.token_scale_ptr_array[group_coord] : nullptr;
        int const global_n = tile_n_origin + thread_idx;
        shared.token_scale[thread_idx] = global_n < tokens && token_scales
                                             ? static_cast<ElementCompute>(token_scales[global_n])
                                             : static_cast<ElementCompute>(params_.thread.token_scale_default);
      }
      synchronize_consumer_warpgroup();
    }

    NumericConverter<ElementD, ElementCompute> convert;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accumulators); ++i) {
      auto coordinate = thread_coordinates(i);
      if (get<0>(coordinate) < physical_m && get<1>(coordinate) < tokens) {
        int const local_m = int(get<0>(coordinate)) - tile_m_origin;
        int const local_n = int(get<1>(coordinate)) - tile_n_origin;
        ElementScalar const* token_scales =
            params_.thread.token_scale_ptr_array ? params_.thread.token_scale_ptr_array[group_coord] : nullptr;
        ElementCompute scale = TileN > 8
                                   ? shared.token_scale[local_n]
                                   : (token_scales ? static_cast<ElementCompute>(token_scales[int(get<1>(coordinate))])
                                                   : static_cast<ElementCompute>(params_.thread.token_scale_default));
        shared.physical_output[local_n * TileM + local_m] =
            convert(static_cast<ElementCompute>(accumulators(i)) * scale);
      }
    }
    synchronize_consumer_warpgroup();

    int const warp = thread_idx / 32;
    int const lane = thread_idx % 32;
    int const warps = NumWarpsPerWarpGroup;
    int64_t const group_row_base = (params_.ptr_D[group_coord] - params_.output_base) / params_.output_row_stride;
    int64_t const logical_tile_origin = int64_t(tile_m_origin) / 2;

    for (int local_n = warp; local_n < TileN; local_n += warps) {
      int64_t const global_token = tile_n_origin + local_n;
      if (global_token >= tokens) {
        continue;
      }
      int64_t const row = group_row_base + global_token;
      if (row >= params_.thread.total_rows) {
        continue;
      }

      for (int pair = lane; pair < LogicalTileM; pair += 32) {
        ElementD const gate = shared.physical_output[local_n * TileM + pair * 2];
        ElementD const up = shared.physical_output[local_n * TileM + pair * 2 + 1];
        float const value = swiglu(gate, up);
        params_.output_base[row * params_.output_row_stride + logical_tile_origin + pair] = ElementD(value);
      }
    }
  }

 private:
  Params params_;
};

}  // namespace cutlass::epilogue::collective
