/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <cstdint>

#include "cute/tensor.hpp"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/numeric_conversion.h"

namespace cutlass::epilogue::collective {

// Shared-memory exchange variant matching the small-K Humming writeback shape:
// accumulator owners scatter BF16 values into a compact token-major tile, then
// all 128 threads issue coalesced 16B global stores.
template <
    class CtaTileShapeMNK_,
    class ElementC_,
    class StrideC_,
    class ElementD_,
    class StrideD_,
    class ElementAccumulator_,
    class ElementScalar_>
class SmemEpilogueArrayPerTokenScale {
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
  static constexpr int OutputAlignmentBits = 128;
  static constexpr int RequiredChannelMultiple = 128;
  static constexpr int ElementsPerVector = 128 / cute::sizeof_bits_v<ElementD>;
  static constexpr int VectorCount = TileElements / ElementsPerVector;
  static constexpr int VectorsPerThread = VectorCount / NumThreadsPerWarpGroup;
  static constexpr int kOutputAlignment = ElementsPerVector;

  static_assert(OutputAlignmentBits % cute::sizeof_bits_v<ElementD> == 0);
  static_assert(TileElements % ElementsPerVector == 0);
  static_assert(TileN % 8 == 0);
  static_assert(TileM % ElementsPerVector == 0, "Each vector store must remain inside one output row.");
  static_assert(
      VectorCount % NumThreadsPerWarpGroup == 0,
      "The compact SMEM epilogue expects an integer number of output vectors per thread.");
  static_assert(
      cute::is_same_v<decltype(cute::get<0>(InternalStrideD{})), cute::Int<1>>,
      "The compact SMEM epilogue requires a unit-stride channel dimension.");
  static_assert(cute::rank(InternalStrideC{}) == 3, "StrideC must be rank-3.");
  static_assert(cute::rank(InternalStrideD{}) == 3, "StrideD must be rank-3.");

  struct SharedStorage {
    alignas(16) ElementD output[TileElements];
    alignas(16) ElementCompute token_scale[TileN > 8 ? TileN : 1];
  };
  using TensorMapStorage = SharedStorage;

  struct ThreadArguments {
    ElementScalar token_scale_default = ElementScalar(1);
    ElementScalar const* const* token_scale_ptr_array = nullptr;
  };

  struct Arguments {
    ThreadArguments thread{};
    ElementC const** ptr_C = nullptr;
    StrideC dC{};
    ElementD** ptr_D = nullptr;
    StrideD dD{};
    // ptr_D entries must address rows within this contiguous output allocation.
    ElementD* output_base = nullptr;
    int64_t output_channel_extent = 0;
    int64_t output_row_stride = 0;
    ElementCompute beta = ElementCompute(0);
  };

  struct Params {
    ThreadArguments thread{};
    ElementD** ptr_D = nullptr;
    StrideD dD{};
    int64_t output_channel_extent = 0;
    int64_t output_row_stride = 0;
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return {args.thread, args.ptr_D, args.dD, args.output_channel_extent, args.output_row_stride};
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
    bool const no_source_or_beta = args.ptr_C == nullptr && args.beta == ElementCompute(0);
    bool const valid_output_storage =
        args.ptr_D != nullptr && args.dD != nullptr && args.output_base != nullptr &&
        (reinterpret_cast<std::uintptr_t>(args.output_base) % (OutputAlignmentBits / 8)) == 0;
    bool const valid_output_shape = args.output_channel_extent > 0 &&
                                    (args.output_channel_extent % RequiredChannelMultiple) == 0 &&
                                    (args.output_channel_extent % TileM) == 0;
    bool const valid_output_stride =
        args.output_row_stride >= args.output_channel_extent && (args.output_row_stride % ElementsPerVector) == 0;

    bool host_shapes_match = true;
    if (problem_shapes.is_host_problem_shape_available()) {
      for (int group = 0; group < problem_shapes.groups(); ++group) {
        auto const problem = problem_shapes.get_host_problem_shape(group);
        int64_t const channel_extent = static_cast<int64_t>(cute::get<0>(problem));
        host_shapes_match = host_shapes_match && channel_extent == args.output_channel_extent &&
                            (channel_extent % RequiredChannelMultiple) == 0;
      }
    }

    if (!no_source_or_beta) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Compact epilogue does not support source C or beta.\n");
    }
    if (!valid_output_storage) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Output storage must provide a 16B-aligned base and pointer/stride "
          "arrays.\n");
    }
    if (!valid_output_shape) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Output channel extent must be a positive multiple of 128 and TileM.\n");
    }
    if (!valid_output_stride) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Output row stride must cover the channel extent and be 16B aligned.\n");
    }
    if (!host_shapes_match) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Host problem shapes do not match the compact epilogue channel "
          "contract.\n");
    }

    return no_source_or_beta && valid_output_storage && valid_output_shape && valid_output_stride && host_shapes_match;
  }

  CUTLASS_HOST_DEVICE
  explicit SmemEpilogueArrayPerTokenScale(Params const& params) : params_(params) {}

  CUTLASS_DEVICE
  bool is_source_needed() const {
    return false;
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

    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto [m_coord, n_coord, k_coord, group_coord] = block_coord_mnkl;
    int const tile_m_origin = int(m_coord) * TileM;
    int const tile_n_origin = int(n_coord) * TileN;

    auto stride_d = [&, group = group_coord]() {
      if constexpr (!is_same_v<InternalStrideD, StrideD>) {
        return detail::get_epilogue_stride<EpilogueSchedule>(params_.dD[group]);
      } else {
        return detail::get_epilogue_stride<EpilogueSchedule>(params_.dD);
      }
    }();

    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor output_coordinates = make_identity_tensor(make_shape(M, N));
    Tensor tile_coordinates = local_tile(output_coordinates, take<0, 2>(block_shape_mnk), make_coord(m_coord, n_coord));
    Tensor thread_coordinates = thread_mma.partition_C(tile_coordinates);

    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(shared_storage_ptr);

    if constexpr (TileN > 8) {
      if (thread_idx < TileN) {
        ElementScalar const* token_scales =
            params_.thread.token_scale_ptr_array ? params_.thread.token_scale_ptr_array[group_coord] : nullptr;
        int const global_n = tile_n_origin + thread_idx;
        shared.token_scale[thread_idx] = global_n < N && token_scales
                                             ? static_cast<ElementCompute>(token_scales[global_n])
                                             : static_cast<ElementCompute>(params_.thread.token_scale_default);
      }
      __syncthreads();
    }

    if constexpr (TileN == 8) {
      ElementScalar const* token_scales =
          params_.thread.token_scale_ptr_array ? params_.thread.token_scale_ptr_array[group_coord] : nullptr;
      NumericConverter<ElementD, ElementCompute> convert;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        auto coordinate = thread_coordinates(i);
        if (get<1>(coordinate) < N) {
          int const local_m = int(get<0>(coordinate)) - tile_m_origin;
          int const local_n = int(get<1>(coordinate)) - tile_n_origin;
          ElementCompute const token_scale = token_scales
                                                 ? static_cast<ElementCompute>(token_scales[int(get<1>(coordinate))])
                                                 : static_cast<ElementCompute>(params_.thread.token_scale_default);
          shared.output[local_n * TileM + local_m] =
              convert(static_cast<ElementCompute>(accumulators(i)) * token_scale);
        }
      }
    } else {
      static_assert(decltype(size(accumulators))::value % 4 == 0);
      NumericArrayConverter<ElementD, ElementCompute, 2> convert;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); i += 4) {
        auto coordinate_0 = thread_coordinates(i);
        auto coordinate_1 = thread_coordinates(i + 1);
        int const local_m_0 = int(get<0>(coordinate_0)) - tile_m_origin;
        int const local_n_0 = int(get<1>(coordinate_0)) - tile_n_origin;
        int const local_m_1 = int(get<0>(coordinate_1)) - tile_m_origin;
        int const local_n_1 = int(get<1>(coordinate_1)) - tile_n_origin;
        ElementCompute const token_scale_0 = shared.token_scale[local_n_0];
        ElementCompute const token_scale_1 = shared.token_scale[local_n_1];
        cutlass::Array<ElementCompute, 2> scaled_accumulators_01{
            static_cast<ElementCompute>(accumulators(i)) * token_scale_0,
            static_cast<ElementCompute>(accumulators(i + 1)) * token_scale_1};
        auto converted_01 = convert(scaled_accumulators_01);

        if (get<1>(coordinate_0) < N) {
          shared.output[local_n_0 * TileM + local_m_0] = converted_01[0];
        }
        if (get<1>(coordinate_1) < N) {
          shared.output[local_n_1 * TileM + local_m_1] = converted_01[1];
        }

        auto coordinate_2 = thread_coordinates(i + 2);
        auto coordinate_3 = thread_coordinates(i + 3);
        int const local_m_2 = int(get<0>(coordinate_2)) - tile_m_origin;
        int const local_m_3 = int(get<0>(coordinate_3)) - tile_m_origin;
#if !defined(NDEBUG)
        // The SM90 GMMA C fragment repeats each token pair across two adjacent
        // channel octets. Reuse those two scales for all four accumulators.
        CUTLASS_ASSERT(int(get<1>(coordinate_2)) - tile_n_origin == local_n_0);
        CUTLASS_ASSERT(int(get<1>(coordinate_3)) - tile_n_origin == local_n_1);
#endif
        cutlass::Array<ElementCompute, 2> scaled_accumulators_23{
            static_cast<ElementCompute>(accumulators(i + 2)) * token_scale_0,
            static_cast<ElementCompute>(accumulators(i + 3)) * token_scale_1};
        auto converted_23 = convert(scaled_accumulators_23);
        if (get<1>(coordinate_2) < N) {
          shared.output[local_n_0 * TileM + local_m_2] = converted_23[0];
        }
        if (get<1>(coordinate_3) < N) {
          shared.output[local_n_1 * TileM + local_m_3] = converted_23[1];
        }
      }
    }

    __syncthreads();

    using OutputVector = cutlass::Array<ElementD, ElementsPerVector>;
    auto const* shared_vectors = reinterpret_cast<OutputVector const*>(shared.output);
    ElementD* output = params_.ptr_D[group_coord];
    int64_t const stride_n = int64_t(get<1>(stride_d));

#if !defined(NDEBUG)
    CUTLASS_ASSERT(int64_t(M) == params_.output_channel_extent);
    CUTLASS_ASSERT((int64_t(M) % RequiredChannelMultiple) == 0);
    CUTLASS_ASSERT(output != nullptr);
    CUTLASS_ASSERT((reinterpret_cast<std::uintptr_t>(output) % (OutputAlignmentBits / 8)) == 0);
    CUTLASS_ASSERT(stride_n == params_.output_row_stride);
    CUTLASS_ASSERT((stride_n % ElementsPerVector) == 0);
#endif

    CUTLASS_PRAGMA_UNROLL
    for (int vector_group = 0; vector_group < VectorsPerThread; ++vector_group) {
      int const vector_idx = thread_idx + vector_group * NumThreadsPerWarpGroup;
      int const element_idx = vector_idx * ElementsPerVector;
      int const local_n = element_idx / TileM;
      int const local_m = element_idx % TileM;
      int const global_n = tile_n_origin + local_n;

      if (global_n < N) {
        auto* output_vector =
            reinterpret_cast<OutputVector*>(output + int64_t(tile_m_origin + local_m) + int64_t(global_n) * stride_n);
        *output_vector = shared_vectors[vector_idx];
      }
    }

    // Larger token tiles synchronize before the next tile scatters output, so
    // that token-scale barrier also protects this output scratch from reuse.
    if constexpr (TileN == 8) {
      __syncthreads();
    }
  }

 private:
  Params params_;
};

// Register-to-global epilogue specialized for the SM90 128x32 C fragment used
// by Humming GEMM2. Each warp transposes its fragment with shuffles into
// coalesced channel-major stores. There is no shared memory scratch and no CTA
// or named-barrier synchronization.
template <
    class CtaTileShapeMNK_,
    class ElementC_,
    class StrideC_,
    class ElementD_,
    class StrideD_,
    class ElementAccumulator_,
    class ElementScalar_,
    bool UsePackedStore_ = false,
    bool PreferMaxMmaRegisters_ = false>
class WarpShuffleEpilogueArrayPerTokenScale {
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
  static constexpr bool UsePackedStore = UsePackedStore_;
  static constexpr bool PreferMaxMmaRegisters = PreferMaxMmaRegisters_;
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
  static constexpr int TileK = cute::size<2>(CtaTileShapeMNK{});
  static constexpr int OutputAlignmentBits = 128;
  static constexpr int RequiredChannelMultiple = 128;
  static constexpr int ElementsPerVector = 128 / cute::sizeof_bits_v<ElementD>;
  static constexpr int kOutputAlignment = ElementsPerVector;

  static_assert(
      TileM == 128 && TileN == 32 && TileK == 512,
      "The warp-shuffle epilogue is specialized for config320's 128x32x512 tile.");
  static_assert(cute::is_same_v<ElementAccumulator, float>, "The warp-shuffle epilogue requires FP32 accumulators.");
  static_assert(
      cute::is_same_v<ElementCompute, float>, "The warp-shuffle epilogue requires FP32 scale multiplication.");
  static_assert(cute::is_same_v<ElementScalar, float>, "The warp-shuffle epilogue requires FP32 per-token scales.");
  static_assert(cute::is_same_v<ElementD, cutlass::bfloat16_t>, "The warp-shuffle epilogue writes BF16 output.");
  static_assert(ElementsPerVector == 8, "One warp-shuffle output sector must contain eight BF16 values.");
  static_assert(
      cute::is_same_v<decltype(cute::get<0>(InternalStrideD{})), cute::Int<1>>,
      "The warp-shuffle epilogue requires a unit-stride channel dimension.");
  static_assert(cute::rank(InternalStrideC{}) == 3, "StrideC must be rank-3.");
  static_assert(cute::rank(InternalStrideD{}) == 3, "StrideD must be rank-3.");

  struct SharedStorage {};
  using TensorMapStorage = SharedStorage;

  struct ThreadArguments {
    ElementScalar token_scale_default = ElementScalar(1);
    ElementScalar const* const* token_scale_ptr_array = nullptr;
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
    int64_t output_channel_extent = 0;
    int64_t output_row_stride = 0;
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return {args.thread, args.ptr_D, args.dD, args.output_channel_extent, args.output_row_stride};
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
    bool const no_source_or_beta = args.ptr_C == nullptr && args.beta == ElementCompute(0);
    bool const valid_output_storage =
        args.ptr_D != nullptr && args.dD != nullptr && args.output_base != nullptr &&
        (reinterpret_cast<std::uintptr_t>(args.output_base) % (OutputAlignmentBits / 8)) == 0;
    bool const valid_output_shape =
        args.output_channel_extent > 0 && (args.output_channel_extent % RequiredChannelMultiple) == 0;
    bool const valid_output_stride =
        args.output_row_stride >= args.output_channel_extent && (args.output_row_stride % ElementsPerVector) == 0;

    bool host_shapes_match = true;
    if (problem_shapes.is_host_problem_shape_available()) {
      for (int group = 0; group < problem_shapes.groups(); ++group) {
        auto const problem = problem_shapes.get_host_problem_shape(group);
        int64_t const channel_extent = static_cast<int64_t>(cute::get<0>(problem));
        host_shapes_match = host_shapes_match && channel_extent == args.output_channel_extent &&
                            (channel_extent % RequiredChannelMultiple) == 0;
      }
    }

    if (!no_source_or_beta) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Warp-shuffle epilogue does not support source C or beta.\n");
    }
    if (!valid_output_storage) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Output storage must provide a 16B-aligned base and pointer/stride "
          "arrays.\n");
    }
    if (!valid_output_shape) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Output channel extent must be a positive multiple of 128.\n");
    }
    if (!valid_output_stride) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Output row stride must cover the channel extent and be 16B aligned.\n");
    }
    if (!host_shapes_match) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Host problem shapes do not match the warp-shuffle epilogue "
          "contract.\n");
    }

    return no_source_or_beta && valid_output_storage && valid_output_shape && valid_output_stride && host_shapes_match;
  }

  CUTLASS_HOST_DEVICE
  explicit WarpShuffleEpilogueArrayPerTokenScale(Params const& params) : params_(params) {}

  CUTLASS_DEVICE
  bool is_source_needed() const {
    return false;
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
      char*) {
    using namespace cute;
    static_assert(is_same_v<BlockShapeMNK, CtaTileShapeMNK>);
    static_assert(
        decltype(size(accumulators))::value == 32, "The SM90 128x32 C fragment must hold 32 FP32 values per thread.");

    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto [m_coord, n_coord, k_coord, group_coord] = block_coord_mnkl;
    int const tile_m_origin = int(m_coord) * TileM;
    int const tile_n_origin = int(n_coord) * TileN;

    auto stride_d = [&, group = group_coord]() {
      if constexpr (!is_same_v<InternalStrideD, StrideD>) {
        return detail::get_epilogue_stride<EpilogueSchedule>(params_.dD[group]);
      } else {
        return detail::get_epilogue_stride<EpilogueSchedule>(params_.dD);
      }
    }();

    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor output_coordinates = make_identity_tensor(make_shape(M, N));
    Tensor tile_coordinates = local_tile(output_coordinates, take<0, 2>(block_shape_mnk), make_coord(m_coord, n_coord));
    Tensor thread_coordinates = thread_mma.partition_C(tile_coordinates);

    ElementScalar const* token_scales =
        params_.thread.token_scale_ptr_array ? params_.thread.token_scale_ptr_array[group_coord] : nullptr;
    ElementD* output = params_.ptr_D[group_coord];
    int64_t const stride_n = int64_t(get<1>(stride_d));
    int const lane = thread_idx % NumThreadsPerWarp;
    int const warp = (thread_idx % NumThreadsPerWarpGroup) / NumThreadsPerWarp;
    int const lane_token_pair = lane % 4;

#if !defined(NDEBUG)
    CUTLASS_ASSERT(int64_t(M) == params_.output_channel_extent);
    CUTLASS_ASSERT((int64_t(M) % RequiredChannelMultiple) == 0);
    CUTLASS_ASSERT(output != nullptr);
    CUTLASS_ASSERT(stride_n == params_.output_row_stride);
    CUTLASS_ASSERT((stride_n % ElementsPerVector) == 0);
#endif

    NumericArrayConverter<ElementD, ElementCompute, 2> convert;

    // SM90 GMMA CLayout_64xN maps lane=(q*4+r) to channel q in each
    // 8-channel octet and token pair r. The tiled 128x32 MMA contributes:
    //   i = parity + 2*m_octet + 4*n_octet + 16*m_half.
    // Pack both channel octets before shuffling so one shuffle transports two
    // independently rounded BF16 values. Each pass remaps all 32 lanes onto
    // two token rows and 16 consecutive channels, yielding full 32B sectors
    // without shared memory or a warpgroup barrier.
    CUTLASS_PRAGMA_UNROLL
    for (int m_half = 0; m_half < 2; ++m_half) {
      CUTLASS_PRAGMA_UNROLL
      for (int n_octet = 0; n_octet < 4; ++n_octet) {
        CUTLASS_PRAGMA_UNROLL
        for (int parity = 0; parity < 2; ++parity) {
          int const fragment_idx = parity + 4 * n_octet + 16 * m_half;
          int const source_local_n = 2 * lane_token_pair + parity + 8 * n_octet;
          int const source_global_n = tile_n_origin + source_local_n;
          ElementCompute const token_scale = source_global_n < N && token_scales
                                                 ? static_cast<ElementCompute>(token_scales[source_global_n])
                                                 : static_cast<ElementCompute>(params_.thread.token_scale_default);
          cutlass::Array<ElementCompute, 2> scaled{
              static_cast<ElementCompute>(accumulators(fragment_idx)) * token_scale,
              static_cast<ElementCompute>(accumulators(fragment_idx + 2)) * token_scale};
          auto converted = convert(scaled);
          ElementD const converted_0 = converted[0];
          ElementD const converted_1 = converted[1];
          uint32_t packed = uint32_t(converted_0.storage) | (uint32_t(converted_1.storage) << 16);

#if !defined(NDEBUG)
          auto coordinate_0 = thread_coordinates(fragment_idx);
          auto coordinate_1 = thread_coordinates(fragment_idx + 2);
          CUTLASS_ASSERT(int(get<1>(coordinate_0)) == source_global_n);
          CUTLASS_ASSERT(int(get<1>(coordinate_1)) == source_global_n);
          CUTLASS_ASSERT(int(get<0>(coordinate_0)) == tile_m_origin + warp * 16 + m_half * 64 + lane / 4);
          CUTLASS_ASSERT(int(get<0>(coordinate_1)) == tile_m_origin + warp * 16 + m_half * 64 + 8 + lane / 4);
#endif

          if constexpr (UsePackedStore) {
            // One lane owns two adjacent channels of one token row. Two
            // shuffles collect the already-rounded BF16 bit patterns, then a
            // single aligned 32-bit store replaces two scalar BF16 stores.
            int const target_token_pair = lane / 8;
            int const channel_pair = lane % 8;
            int const source_channel_0 = (2 * channel_pair) % ElementsPerVector;
            int const source_lane_0 = target_token_pair + 4 * source_channel_0;
            int const source_lane_1 = source_lane_0 + 4;
            uint32_t const shuffled_0 = __shfl_sync(0xffffffffu, packed, source_lane_0);
            uint32_t const shuffled_1 = __shfl_sync(0xffffffffu, packed, source_lane_1);
            bool const upper_octet = channel_pair >= 4;
            uint16_t const bits_0 = upper_octet ? uint16_t(shuffled_0 >> 16) : uint16_t(shuffled_0);
            uint16_t const bits_1 = upper_octet ? uint16_t(shuffled_1 >> 16) : uint16_t(shuffled_1);
            uint32_t const output_bits = uint32_t(bits_0) | (uint32_t(bits_1) << 16);
            int const target_local_n = 2 * target_token_pair + parity + 8 * n_octet;
            if (tile_n_origin + target_local_n < N) {
              int const target_m = tile_m_origin + warp * 16 + m_half * 64 + 2 * channel_pair;
              auto* output_pair = reinterpret_cast<uint32_t*>(
                  output + int64_t(target_m) + int64_t(tile_n_origin + target_local_n) * stride_n);
              *output_pair = output_bits;
            }
          } else {
            CUTLASS_PRAGMA_UNROLL
            for (int pass = 0; pass < 2; ++pass) {
              int const target_token_pair = 2 * pass + lane / 16;
              int const target_channel = lane % 16;
              int const source_lane = target_token_pair + 4 * (target_channel % ElementsPerVector);
              uint32_t const shuffled = __shfl_sync(0xffffffffu, packed, source_lane);
              uint16_t const bits = target_channel < ElementsPerVector ? uint16_t(shuffled) : uint16_t(shuffled >> 16);
              int const target_local_n = 2 * target_token_pair + parity + 8 * n_octet;
              if (tile_n_origin + target_local_n < N) {
                int const target_m = tile_m_origin + warp * 16 + m_half * 64 + target_channel;
                output[int64_t(target_m) + int64_t(tile_n_origin + target_local_n) * stride_n] =
                    ElementD::bitcast(bits);
              }
            }
          }
        }
      }
    }
  }

 private:
  Params params_;
};

}  // namespace cutlass::epilogue::collective
