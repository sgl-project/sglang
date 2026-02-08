/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_traits_sm90_im2col.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"

#include "cutlass/conv/detail.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/util/packed_stride.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  conv::Operator ConvOp,
  int Stages,
  int NumSpatialDims,
  class ClusterShape,
  class KernelSchedule,
  int PipelineAsyncMmaStages,
  class TileShape_,
  class ElementA_,
  class ElementB_,
  class TiledMma_,
  class TileTraitsA_,
  class TileTraitsB_>
struct CollectiveConv<
    MainloopSm90TmaGmmaWarpSpecializedImplicitGemm<
        ConvOp, Stages, NumSpatialDims, ClusterShape, KernelSchedule, PipelineAsyncMmaStages>,
    TileShape_,
    ElementA_,
    ElementB_,
    TiledMma_,
    TileTraitsA_,
    TileTraitsB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecializedImplicitGemm<
      ConvOp, Stages, NumSpatialDims, ClusterShape, KernelSchedule, PipelineAsyncMmaStages>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = typename TileTraitsA_::GmemTiledCopy;
  using GmemTiledCopyB = typename TileTraitsB_::GmemTiledCopy;
  using SmemLayoutA = typename TileTraitsA_::SmemLayout;
  using SmemLayoutB = typename TileTraitsB_::SmemLayout;
  using ArchTag = typename DispatchPolicy::ArchTag;
  static constexpr int NumSpatialDimensions = DispatchPolicy::NumSpatialDimensions;
  static constexpr int NumTensorDimensions = NumSpatialDimensions + 2;
  // Deduce the kernel-facing stride tuple types based on the dispatch policy
  // (which is a function of the number of spatial dimensions, the algorithm, etc.)
  using StrideA = decltype(detail::sm90_dispatch_policy_to_stride_A<DispatchPolicy>());
  using StrideB = decltype(detail::sm90_dispatch_policy_to_stride_B<DispatchPolicy>());

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState  = typename cutlass::PipelineState<DispatchPolicy::Stages>;

  using ProblemShape = ConvProblemShape<ConvOp, NumSpatialDimensions>;

  static_assert(rank(SmemLayoutA{}) == 3, "SmemLayout must be rank 3 (M/N, K, PIPE)");
  static_assert((size<0>(TileShape{}) == size<0>(SmemLayoutA{})), "SmemLayout must be compatible with the tile shape.");
  static_assert((size<2>(TileShape{}) == size<1>(SmemLayoutA{})), "SmemLayout must be compatible with the tile shape.");

  static_assert(rank(SmemLayoutB{}) == 3, "SmemLayout must be rank 3 (M/N, K, PIPE)");
  static_assert((size<1>(TileShape{}) == size<0>(SmemLayoutB{})), "SmemLayout must be compatible with the tile shape.");
  static_assert((size<2>(TileShape{}) == size<1>(SmemLayoutB{})), "SmemLayout must be compatible with the tile shape.");

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
  static_assert(cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");

  // The tma load mode of wgrad is tiled for tensor A and im2col for tensor B while the tma load mode of fprop and dgrad
  // kernel is im2col for tensor A and tiled for tensor B.
  static_assert((ConvOp == conv::Operator::kWgrad
             && (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>))
             || (ConvOp != conv::Operator::kWgrad
             && (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_IM2COL> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_IM2COL_MULTICAST>)),
      "GmemTiledCopyA - invalid SM90 TMA copy atom specified.");
  static_assert((ConvOp == conv::Operator::kWgrad
             && (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_IM2COL> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_IM2COL_MULTICAST>))
             || (ConvOp != conv::Operator::kWgrad
             && (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>)),
      "GmemTiledCopyB - invalid SM90 TMA copy atom specified.");

  static constexpr bool is_im2col_A = detail::is_im2col_load<GmemTiledCopyA>::value;
  static constexpr bool is_im2col_B = detail::is_im2col_load<GmemTiledCopyB>::value;

  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using InternalElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using InternalElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;

  struct SharedStorage
  {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = DispatchPolicy::PipelineAsyncMmaStages;
  static constexpr uint32_t TmaTransactionBytes =
      (size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof(InternalElementA)))+
      (size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof(InternalElementB)));

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A{nullptr};
    ElementB const* ptr_B{nullptr};
  };

private:
  // Note that for fprop and dgrad kernel, the tma load mode is im2col for tensor A and tiled for
  // tensor B while for wgrad kernel, the tma load mode is tiled for tensor A and im2col for tensor
  // B since operand A, B is swapped.
  // Get tma_load_a instantce.
  template <class TensorA>
  static constexpr auto
  get_tma_load_a_instance(TensorA const& tensor_a, ProblemShape const& problem_shape) {
    if constexpr (is_im2col_A) {
      // compute the upper and lower corners based on the conv padding
      auto lower_corner_whd = detail::compute_lower_corner_whd(problem_shape);
      auto upper_corner_whd = detail::compute_upper_corner_whd(problem_shape);
      auto lower_srt = detail::compute_lower_srt(problem_shape);

      // The calculation of gbasis strides for dgrad kernel needs perform negate for dilation values.
      cute::array<int32_t, NumSpatialDimensions> stride_srt{};
      for (int i = 0; i < NumSpatialDimensions; ++i) {
        stride_srt[i] = ConvOp == conv::Operator::kDgrad ?
            -problem_shape.dilation[NumSpatialDimensions-1-i] :
            problem_shape.dilation[NumSpatialDimensions-1-i];
      }

      return make_im2col_tma_copy(
          GmemTiledCopyA{},
          tensor_a,
          SmemLayoutA{}(_,_,_0{}),
          product_each(shape(SmemLayoutA{}(_,_,_0{}))),
          size<1>(ClusterShape{}),
          shape(lower_corner_whd),
          shape(upper_corner_whd),
          cute::reverse(shape(problem_shape.lower_padding)),
          cute::reverse(shape(problem_shape.upper_padding)),
          cute::reverse(shape(problem_shape.traversal_stride)),
          shape(lower_srt),
          shape(stride_srt));
    }
    // TMA tiled mode for tensor A in wgrad kernel.
    else {
      return make_tma_copy(
          GmemTiledCopyA{},
          tensor_a,
          SmemLayoutA{}(_,_,_0{}),
          make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
          size<1>(ClusterShape{}));
    }
  }

  // Get tma_load_b instantce.
  template <class TensorB>
  static constexpr auto
  get_tma_load_b_instance(TensorB const& tensor_b, ProblemShape const& problem_shape) {
    // TMA im2col mode for tensor B in wgrad kernel.
    if constexpr (is_im2col_B) {
      // compute the upper and lower corners based on the conv padding
      auto lower_corner_whd = detail::compute_lower_corner_whd(problem_shape);
      auto upper_corner_whd = detail::compute_upper_corner_whd(problem_shape);
      auto lower_srt = detail::compute_lower_srt(problem_shape);

      return make_im2col_tma_copy(
          GmemTiledCopyB{},
          tensor_b,
          SmemLayoutB{}(_,_,_0{}),
          product_each(shape(SmemLayoutB{}(_,_,_0{}))),
          size<0>(ClusterShape{}),
          shape(lower_corner_whd),
          shape(upper_corner_whd),
          cute::reverse(shape(problem_shape.lower_padding)),
          cute::reverse(shape(problem_shape.upper_padding)),
          cute::reverse(shape(problem_shape.traversal_stride)),
          shape(lower_srt),
          cute::reverse(shape(problem_shape.dilation)));
    }
    else {
      return make_tma_copy(
          GmemTiledCopyB{},
          tensor_b,
          SmemLayoutB{}(_,_,_0{}),
          make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
          size<0>(ClusterShape{}));
    }
  }

public:

  // Performs im2col transformations on the input of type ConvProblemShape
  static constexpr auto
  get_problem_shape_MNKL(ProblemShape const& problem_shape) {

    if constexpr (is_im2col_A || is_im2col_B) {
      // transformation + im2col linearization
      return cutlass::conv::detail::get_linearized_problem_shape_MNKL(problem_shape);
    }
    else {
      // transformation
      return cutlass::conv::detail::get_transformed_problem_shape_MNKL(problem_shape);
    }
  }

  // Device side kernel params
  struct Params {
    using _Submode = decltype(take<0,NumTensorDimensions-1>(typename ProblemShape::TensorExtent{}));

    // Assumption: StrideA is congruent with Problem_MK
    // Select TMA load type according to convolution operator.
    using TensorShapeA = cute::conditional_t<ConvOp == conv::Operator::kWgrad,
        decltype(repeat_like(StrideA{}, int32_t(0))),
        decltype(make_shape(_Submode{}, int(0)))>;

    using TensorShapeB = cute::conditional_t<ConvOp == conv::Operator::kWgrad,
        decltype(make_shape(int(0), _Submode{})),
        decltype(repeat_like(StrideB{}, int32_t(0)))>;

    using TMA_A = decltype(get_tma_load_a_instance(
        make_tensor(
            make_gmem_ptr(static_cast<InternalElementA const*>(nullptr)),
            make_layout(TensorShapeA{}, StrideA{})),
        ConvProblemShape<ConvOp, NumSpatialDimensions>{}));

    using TMA_B = decltype(get_tma_load_b_instance(
        make_tensor(
            make_gmem_ptr(static_cast<InternalElementB const*>(nullptr)),
            make_layout(TensorShapeB{}, StrideB{})),
        ConvProblemShape<ConvOp, NumSpatialDimensions>{}));

    // Members
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
  };

  //
  // Methods
  //

  // Lowers the host side user facing arguments to the kernel facing lauch params
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;
    // from the flat problem shape arrays of ConvProblemShape<ConvOp, N>, create a rank-3 MNK problem shape tuple
    // tma desc creation depends on the original untransformed domain.

    // A extents.
    auto shape_A_orig = problem_shape.get_shape_A();
    // B extents.
    auto shape_B_orig = problem_shape.get_shape_B();

    // Fill inferred cute strides from flat stride arrays
    auto dA = make_cute_packed_stride(StrideA{}, problem_shape.stride_A, ConvOp);
    auto dB = make_cute_packed_stride(StrideB{}, problem_shape.stride_B, ConvOp);

    auto ptr_A = reinterpret_cast<InternalElementA const*>(args.ptr_A);
    auto ptr_B = reinterpret_cast<InternalElementB const*>(args.ptr_B);

    Tensor tensor_a = make_tensor(make_gmem_ptr(ptr_A), make_layout(shape_A_orig, dA));
    Tensor tensor_b = make_tensor(make_gmem_ptr(ptr_B), make_layout(shape_B_orig, dB));

    auto tma_load_a = get_tma_load_a_instance(tensor_a, problem_shape);
    auto tma_load_b = get_tma_load_b_instance(tensor_b, problem_shape);

    return {
      tma_load_a,
      tma_load_b,
      TmaTransactionBytes
    };
  }

  template <class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      Arguments const& args) {
    // Activation and Filter channel mode extents much match
    bool implementable = true;
    // channel mode is major
    implementable &= problem_shape.stride_A[NumTensorDimensions-1] == 1;
    implementable &= problem_shape.stride_B[NumTensorDimensions-1] == 1;

    constexpr int tma_alignment_bits = 128;
    // A extents.
    auto shape_A_orig = problem_shape.get_shape_A();
    // B extents.
    auto shape_B_orig = problem_shape.get_shape_B();
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(shape_A_orig, StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(shape_B_orig, StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
      return false;
    }

    // Check valid padding values for TMA_LOAD_IM2COL
    constexpr int padding_limit = (ProblemShape::RankS == 1) ? 65536 : (ProblemShape::RankS == 2 ? 256 : 16);
    for (int i = 0; i < problem_shape.RankS; ++i) {
      implementable = implementable && problem_shape.lower_padding[i] <= padding_limit && problem_shape.lower_padding[i] >= 0;
      implementable = implementable && problem_shape.upper_padding[i] <= padding_limit && problem_shape.upper_padding[i] >= 0;
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Padding values don't meet requirements for TMA LOAD IM2COL.\n");
      return false;
    }

    if (is_im2col_A || is_im2col_B) {
      // Check valid corner values for TMA_LOAD_IM2COL, signed int ranging from [-corner_limit, corner_limit - 1]
      constexpr int32_t corner_limit = 1 << (16 / NumSpatialDimensions - 1);
      auto lower_corner_whd = detail::compute_lower_corner_whd(problem_shape);
      for (int i = 0; i < problem_shape.RankS; ++i) {
        implementable = implementable && lower_corner_whd[i] >= -corner_limit && lower_corner_whd[i] <= (corner_limit - 1);
      }
      auto upper_corner_whd = detail::compute_upper_corner_whd(problem_shape);
      for (int i = 0; i < problem_shape.RankS; ++i) {
        implementable = implementable && upper_corner_whd[i] >= -corner_limit && upper_corner_whd[i] <= (corner_limit - 1);
      }

      if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Padding values don't meet requirements for TMA LOAD IM2COL.\n");
        return false;
      }
    }

    if (is_im2col_A || is_im2col_B) {
      // Check valid filter offsets for TMA_LOAD_IM2COL, unsigned int ranging from [0, offset_limit - 1]
      constexpr int32_t offset_limit = (1 << (16 / NumSpatialDimensions)) - 1;
      auto flt_data = (ConvOp == conv::Operator::kWgrad) ? problem_shape.shape_C : problem_shape.shape_B;
      for (int i = 0; i < problem_shape.RankS; ++i) {
        // flt_data array contains [K, T, R, S, C], so pure filter [T, R, S] starts from the second position in the array
        implementable = implementable && ((flt_data[i+1] - 1) * problem_shape.dilation[i] >= 0)
                                      && ((flt_data[i+1] - 1) * problem_shape.dilation[i] < offset_limit);
      }

      if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: tensor coordinate offset values don't meet requirements for TMA LOAD IM2COL.\n");
        return false;
      }
    }

    // Wgrad kernels don't support non-packed output strides, non-packed tensor A stride (linearized)
    if constexpr (ConvOp == conv::Operator::kWgrad) {
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      std::ostringstream os;
#endif
      const auto & input_shape  = problem_shape.shape_A;
      const auto & input_stride  = problem_shape.stride_A;

      implementable &= input_stride[ProblemShape::RankT - 1] == 1;
      int64_t input_shape_size = 1;
      for (int i = ProblemShape::RankT - 2; i >= 0; --i) {
        input_shape_size *= input_shape[i + 1];
        implementable &= input_stride[i] == input_shape_size;
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        if (input_stride[i] != input_shape_size) {
          os << "\n    *** input_stride[" << i << "] = " << input_stride[i] << " != input_shape_size = " << input_shape_size << " ***";
        }
#endif
      }

      if (!implementable) {
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        os << "\n    input_shape_size: " << input_shape_size
           << "\n    input_shape: " << input_shape
           << "\n    input_stride: " << input_stride
           << "\n";
#endif
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Wgrad kernels don't support non-packed input strides.\n");
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        CUTLASS_TRACE_HOST(os.str());
#endif
        return false;
      }

      const auto & output_shape  = problem_shape.shape_C;
      const auto & output_stride  = problem_shape.stride_C;

      implementable &= output_stride[ProblemShape::RankT - 1] == 1;
      int64_t output_shape_size = 1;
      for (int i = ProblemShape::RankT - 2; i >= 0; --i) {
        output_shape_size *= output_shape[i + 1];
        implementable &= output_stride[i] == output_shape_size;
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        if (output_stride[i] != output_shape_size) {
          os << "\n    *** output_stride[" << i << "] = " << output_stride[i] << " != output_shape_size = " << output_shape_size << " ***";
        }
#endif
      }

      if (!implementable) {
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        os << "\n    output_shape_size: " << input_shape_size
           << "\n    output_shape: " << input_shape
           << "\n    output_stride: " << input_stride
           << "\n";
#endif
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Wgrad kernels don't support non-packed output strides.\n");
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        CUTLASS_TRACE_HOST(os.str());
#endif
        return false;
      }
    }

    // Conv kernels only support cross correlation mode currently.
    implementable &= problem_shape.mode == cutlass::conv::Mode::kCrossCorrelation;

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Conv kernels only support cross correlation mode currently.\n");
      return false;
    }

    if (problem_shape.groups > 1) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: This kernel does not support conv groups > 1.\n");
      return false;
    }

    if constexpr (is_im2col_A || is_im2col_B) {
      auto [M, N, K, L] = cutlass::conv::detail::get_transformed_problem_shape_MNKL(problem_shape);
      auto to_64b = [](auto S) { return transform_leaf(S, [](auto s) { return static_cast<int64_t>(s); }); };

      if constexpr (ConvOp == conv::Operator::kFprop || ConvOp == conv::Operator::kDgrad) {
        implementable &= (cute::product(to_64b(M)) <= cutlass::platform::numeric_limits<int32_t>::max()) &
                         (cute::product(to_64b(L)) <= cutlass::platform::numeric_limits<int32_t>::max());
      }
      else if constexpr (ConvOp == conv::Operator::kWgrad) {
        implementable &= (cute::product(to_64b(K)) <= cutlass::platform::numeric_limits<int32_t>::max());
      }

      if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: the extents exceed the maximum number.\n");
        return false;
      }
    }

    return true;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract
  /// Returned tuple must contain at least two elements, with the first two elements being:
  /// gA_mk - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k)
  /// gB_nk - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k)
  /// The rest of the tensors can be specified as needed by this collective.
  /// The dimensions of gA_mk and gA_nk do not contain L to maintain consistency with
  /// StrideA and StrideB set up for TMA
  template <class ProblemShapeMNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShapeMNKL const& problem_shape_MNKL, Params const& mainloop_params){
  //load_init(ProblemShapeMNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mk = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K));                            // (m,k)
    Tensor mB_nk = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K));                            // (n,k)

    // Make tiled views, defer the slice
    Tensor gA_mk = local_tile(mA_mk, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k)
    Tensor gB_nk = local_tile(mB_nk, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k)

    return cute::make_tuple(gA_mk, gB_nk);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TensorB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_producer_state,
      cute::tuple<TensorA, TensorB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {

    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A and B
      //
      constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());

      uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
      auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

      auto [gA_mk, gB_nk] = load_inputs;

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

      Tensor gA = gA_mk(_,_,m_coord,_);                                                     // (BLK_M,BLK_K,k)
      Tensor gB = gB_nk(_,_,n_coord,_);                                                     // (BLK_N,BLK_K,k)

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;

      // Issue TmaLoads
      // Maps the tile -> block, value
      if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_IM2COL_MULTICAST> ||
                    cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int n = 0; n < size<1>(block_layout); ++n) {
          mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
        }
      }

      if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_IM2COL_MULTICAST> ||
                    cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int m = 0; m < size<0>(block_layout); ++m) {
          mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
        }
      }

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_producer_state for _writing_
        pipeline.producer_acquire(smem_pipe_producer_state);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_producer_state);

        int write_stage = smem_pipe_producer_state.index();

        copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        ++k_tile_iter;

        // Advance smem_pipe_producer_state
        ++smem_pipe_producer_state;
      }
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_producer_state) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was
       * still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_producer_state);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <class FrgTensorC>
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_consumer_state,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCsA = thread_mma.partition_A(sA);                                                 // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                                 // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                                                         // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_consumer_state;

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);

    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

    warpgroup_fence_operand(accum);
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count; k_tile_prologue > 0; --k_tile_prologue) {
      // WAIT on smem_pipe_consumer_state until its data are available (phase bit flips from rdPhaseBit value)
      pipeline.consumer_wait(smem_pipe_consumer_state);

      int read_stage = smem_pipe_consumer_state.index();
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }

      warpgroup_commit_batch();

      ++smem_pipe_consumer_state;
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      // WAIT on smem_pipe_consumer_state until its data are available (phase bit flips from rdPhaseBit value)
      pipeline.consumer_wait(smem_pipe_consumer_state);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_consumer_state.index();
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_producer_state is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);

      // UNLOCK smem_pipe_release, done _computing_ on it
      pipeline.consumer_release(smem_pipe_release);

      // Advance smem_pipe_consumer_state and smem_pipe_release
      ++smem_pipe_consumer_state;
      ++smem_pipe_release;
    }

    warpgroup_fence_operand(accum);
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
