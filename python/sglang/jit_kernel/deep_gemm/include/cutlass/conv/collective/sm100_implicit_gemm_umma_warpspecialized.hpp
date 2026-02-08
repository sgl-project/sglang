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
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/detail/cluster.hpp"

#include "cutlass/conv/detail.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cutlass/trace.h"

#if (! defined(__CUDA_ARCH__)) && (CUTLASS_DEBUG_TRACE_LEVEL > 0)
#  include <sstream>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
// Both DMA Load and MMA methods of this class must be run by a single thread that's picked by elect_one
template <
  conv::Operator ConvOp,
  int Stages,
  int NumSpatialDims,
  int SchedulerPipelineStageCount,
  int AccumulatorPipelineStageCount,
  class ClusterShape,    // Static cluster shape or dynamic (int, int, _1)
  class TileShapeMNKL_,  // (MmaAtomShapeM, MmaAtomShapeN, TileK, optional: TileL)
  class ElementA_,
  class ElementB_,
  class TiledMma_,
  class TileTraitsA_,
  class TileTraitsB_>
struct CollectiveConv<
    MainloopSm100TmaUmmaWarpSpecializedImplicitGemm<
      ConvOp,
      Stages,
      NumSpatialDims,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape>,
    TileShapeMNKL_,
    ElementA_,
    ElementB_,
    TiledMma_,
    TileTraitsA_,
    TileTraitsB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm100TmaUmmaWarpSpecializedImplicitGemm<
                           ConvOp,
                           Stages,
                           NumSpatialDims,
                           SchedulerPipelineStageCount,
                           AccumulatorPipelineStageCount,
                           ClusterShape>;
  using TileShape = decltype(cute::take<0,3>(TileShapeMNKL_{})); // (MmaAtomShapeM, MmaAtomShapeN, TileK)
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = typename TileTraitsA_::GmemTiledCopy;
  using GmemTiledCopyB = typename TileTraitsB_::GmemTiledCopy;
  using SmemLayoutAtomA = typename TileTraitsA_::SmemLayoutAtom;
  using SmemLayoutAtomB = typename TileTraitsB_::SmemLayoutAtom;
  using ArchTag = typename DispatchPolicy::ArchTag;
  static constexpr int NumSpatialDimensions = DispatchPolicy::NumSpatialDimensions;
  static constexpr int NumTensorDimensions = NumSpatialDimensions + 2;
  // deducde the kernel facing stride tuple types based on the dispatch policy (spatial dim, algo, etc.)
  using StrideA = decltype(detail::sm100_dispatch_policy_to_stride_A<DispatchPolicy>());
  using StrideB = decltype(detail::sm100_dispatch_policy_to_stride_B<DispatchPolicy>());

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using TmaInternalElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, cute::uint_bit_t<cute::sizeof_bits_v<ElementA>>>;
  using TmaInternalElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, cute::uint_bit_t<cute::sizeof_bits_v<ElementB>>>;

  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  // Determine MMA type: MMA_1SM vs MMA_2SM
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma_::ThrLayoutVMNK{})), _1, _1>;

  using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
                             DispatchPolicy::Stages,
                             ClusterShape,
                             AtomThrShapeMNK>;
  using MainloopPipelineState = typename MainloopPipeline::PipelineState;

  using ProblemShape = ConvProblemShape<ConvOp, NumSpatialDimensions>;

  CUTE_STATIC_ASSERT_V(evenly_divides(shape<0>(TileShape{}), tile_size<0>(TiledMma{})), "TileShape_M should be evenly divided by TiledMma_M");
  CUTE_STATIC_ASSERT_V(evenly_divides(shape<1>(TileShape{}), tile_size<1>(TiledMma{})) || (ConvOp == conv::Operator::kWgrad), "TileShape_N should be evenly divided by TiledMma_N");

  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));

  // Define A and B block shapes for reduced size TMA_LOADs
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert(((size<0,0>(MmaShapeA_MK{}) * size<1>(MmaShapeA_MK{})) % size<0>(SmemLayoutAtomA{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(((size<0,1>(MmaShapeA_MK{}) * size<2>(MmaShapeA_MK{})) % size<1>(SmemLayoutAtomA{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert(((size<0,0>(MmaShapeB_NK{}) * size<1>(MmaShapeB_NK{})) % size<0>(SmemLayoutAtomB{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(((size<0,1>(MmaShapeB_NK{}) * size<2>(MmaShapeB_NK{})) % size<1>(SmemLayoutAtomB{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");

  // Tile along K mode first before tiling over MN. PIPE mode last as usual.
  // This maximizes TMA boxes due to better smem-K vectorization, reducing total issued TMAs.
  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomA{},
      append(MmaShapeA_MK{}, Int<DispatchPolicy::Stages>{}),
      Step<_2,_1,_3>{}));
  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(MmaShapeB_NK{}, Int<DispatchPolicy::Stages>{}),
      Step<_2,_1,_3>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
  static_assert(cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");

  static constexpr bool is_im2col_A = detail::is_im2col_load<GmemTiledCopyA>::value;
  static constexpr bool is_im2col_B = detail::is_im2col_load<GmemTiledCopyB>::value;
  static constexpr bool is_strided_dgrad = ConvOp == conv::Operator::kDgrad && not is_im2col_A && not is_im2col_B;

  static constexpr int TileShapeMNKLRank = rank(TileShapeMNKL_{});
  // If rank > 3, TileL exists and it is GroupsPerTile. The kernel is grouped conv now.
  static constexpr bool is_grouped_wgrad = ConvOp == conv::Operator::kWgrad && TileShapeMNKLRank > 3;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Only one thread issues the TMA and updates the barriers in a 2SM MMA, adjust bytes accordingly
  static constexpr uint32_t TmaTransactionBytes =
    size(AtomThrShapeMNK{}) * (size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * size<2>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof(ElementA))) +
    size(AtomThrShapeMNK{}) * (size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * size<2>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof(ElementB)));

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A{nullptr};
    ElementB const* ptr_B{nullptr};
  };

private:

  // Note that for fprop and non-strided dgrad kernel, the tma load mode is im2col for tensor A and tiled for
  // tensor B while for wgrad kernel, the tma load mode is tiled for tensor A and im2col for tensor
  // B since operand A, B is swapped.
  // For strided dgrad A and B are both tma tiled and not im2col

  template <class TensorA, class ClusterShapeVMNK>
  static constexpr auto
  get_tma_load_a_instance(
    TensorA const& tensor_a,
    ProblemShape const& problem_shape,
    ClusterShapeVMNK const& cluster_shape_vmnk) {

    if constexpr (is_im2col_A) {
      // compute the upper and lower corners based on the conv padding
      auto lower_corner_whd = detail::compute_lower_corner_whd(problem_shape);
      auto upper_corner_whd = detail::compute_upper_corner_whd(problem_shape);
      auto lower_srt = detail::compute_lower_srt(problem_shape);

      // gbasis strides for dgrad kernel need to be negated
      cute::array<int32_t, NumSpatialDimensions> stride_srt{};
      for (int i = 0; i < NumSpatialDimensions; ++i) {
        stride_srt[i] = ConvOp == conv::Operator::kDgrad ?
            -problem_shape.dilation[NumSpatialDimensions-1-i] :
            problem_shape.dilation[NumSpatialDimensions-1-i];
      }

      return make_im2col_tma_atom_A_sm100(
          GmemTiledCopyA{},
          tensor_a,
          SmemLayoutA{}(_,_,_,cute::Int<0>{}),
          TileShape{},
          TiledMma{},
          cluster_shape_vmnk,
          shape(lower_corner_whd),
          shape(upper_corner_whd),
          cute::reverse(shape(problem_shape.lower_padding)),
          cute::reverse(shape(problem_shape.upper_padding)),
          cute::reverse(shape(problem_shape.traversal_stride)),
          shape(lower_srt),
          shape(stride_srt));
    }
    // TMA tiled mode for tensor A in wgrad and strided dgrad
    else {
      return make_tma_atom_A_sm100<TmaInternalElementA>(
          GmemTiledCopyA{},
          tensor_a,
          SmemLayoutA{}(_,_,_,cute::Int<0>{}),
          TileShape{},
          TiledMma{},
          cluster_shape_vmnk);
    }
  }

  template <class TensorB, class ClusterShapeVMNK>
  static constexpr auto
  get_tma_load_b_instance(
    TensorB const& tensor_b,
    ProblemShape const& problem_shape,
    ClusterShapeVMNK const& cluster_shape_vmnk) {

    if constexpr (is_im2col_B) {
      // compute the upper and lower corners based on the conv padding
      auto lower_corner_whd = detail::compute_lower_corner_whd(problem_shape);
      auto upper_corner_whd = detail::compute_upper_corner_whd(problem_shape);
      auto lower_srt = detail::compute_lower_srt(problem_shape);

      return make_im2col_tma_atom_B_sm100(
          GmemTiledCopyB{},
          tensor_b,
          SmemLayoutB{}(_,_,_,cute::Int<0>{}),
          TileShape{},
          TiledMma{},
          cluster_shape_vmnk,
          shape(lower_corner_whd),
          shape(upper_corner_whd),
          cute::reverse(shape(problem_shape.lower_padding)),
          cute::reverse(shape(problem_shape.upper_padding)),
          cute::reverse(shape(problem_shape.traversal_stride)),
          shape(lower_srt),
          cute::reverse(shape(problem_shape.dilation)));
    }
    else {
      return make_tma_atom_B_sm100<TmaInternalElementB>(
          GmemTiledCopyB{},
          tensor_b,
          SmemLayoutB{}(_,_,_,cute::Int<0>{}),
          TileShape{},
          TiledMma{},
          cluster_shape_vmnk);
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

  // Device-side kernel params
  //
  // Arguments has the untransformed problem shape from the user.
  // Params will have the transformed problem shape.
  struct Params {
    using _Submode = decltype(take<0,NumTensorDimensions-1>(typename ProblemShape::TensorExtent{}));

    using ClusterLayout_VMNK = decltype(tiled_divide(make_layout(conditional_return<IsDynamicCluster>(make_shape(uint32_t(0), uint32_t(0), Int<1>{}), ClusterShape{})),
                                                     make_tile(typename TiledMma::AtomThrID{})));

    // Assumption: StrideA is congruent with Problem_MK
    // Select TMA load type according to convolution operator.
    using TensorShapeA = cute::conditional_t<ConvOp == conv::Operator::kWgrad,
        decltype(repeat_like(StrideA{}, int32_t(0))),
        decltype(make_shape(_Submode{}, int32_t(0)))>;

    using TensorShapeB = cute::conditional_t<ConvOp == conv::Operator::kWgrad,
        decltype(make_shape(int32_t(0), _Submode{})),
        decltype(repeat_like(StrideB{}, int32_t(0)))>;

    using TMA_A = decltype(get_tma_load_a_instance(
        make_tensor(
            make_gmem_ptr(recast_ptr<TmaInternalElementA>(nullptr)),
            make_layout(TensorShapeA{}, StrideA{})),
        ConvProblemShape<ConvOp, NumSpatialDimensions>{},
        ClusterLayout_VMNK{}));

    using TMA_B = decltype(get_tma_load_b_instance(
        make_tensor(
            make_gmem_ptr(recast_ptr<TmaInternalElementB>(nullptr)),
            make_layout(TensorShapeB{}, StrideB{})),
        ConvProblemShape<ConvOp, NumSpatialDimensions>{},
        ClusterLayout_VMNK{}));

    // Members
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_A tma_load_a_fallback;
    TMA_B tma_load_b_fallback;
    dim3 cluster_shape_fallback;
  };

  //
  // Constructor
  //
  CUTLASS_DEVICE
  CollectiveConv(Params const& params, ClusterShape cluster_shape, uint32_t block_rank_in_cluster)
    : cluster_shape_(cluster_shape)
    , block_rank_in_cluster_(block_rank_in_cluster) {
    if constexpr (IsDynamicCluster) {
      const bool is_fallback_cluster = (cute::size<0>(cluster_shape_) == params.cluster_shape_fallback.x &&
                                        cute::size<1>(cluster_shape_) == params.cluster_shape_fallback.y);
      observed_tma_load_a_ = is_fallback_cluster ? &params.tma_load_a_fallback : &params.tma_load_a;
      observed_tma_load_b_ = is_fallback_cluster ? &params.tma_load_b_fallback : &params.tma_load_b;
    }
    else {
      observed_tma_load_a_ = &params.tma_load_a;
      observed_tma_load_b_ = &params.tma_load_b;
    }
  }

  //
  // Methods
  //

  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{}) {
    (void) workspace;

    // from the flat problem shape arrays of ConvProblemShape<N>, create a rank-3 MNK problem shape tuple
    // tma desc creation depends on the original untransformed domain.

    // A extents.
    auto shape_A_orig = problem_shape.get_shape_A();
    // B extents.
    auto shape_B_orig = problem_shape.get_shape_B();

    // Fill inferred cute strides from flat stride arrays
    auto dA = make_cute_packed_stride(StrideA{}, problem_shape.stride_A, ConvOp);
    auto dB = make_cute_packed_stride(StrideB{}, problem_shape.stride_B, ConvOp);

    auto ptr_A = recast_ptr<TmaInternalElementA>(args.ptr_A);
    auto ptr_B = recast_ptr<TmaInternalElementB>(args.ptr_B);

    Tensor tensor_a = make_tensor(make_gmem_ptr(ptr_A), make_layout(shape_A_orig, dA));
    Tensor tensor_b = make_tensor(make_gmem_ptr(ptr_B), make_layout(shape_B_orig, dB));

    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);
    // Cluster layout for TMA construction
    auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));
    auto cluster_shape_fallback = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape_fallback);

    // Cluster layout for TMA construction
    auto cluster_layout_vmnk_fallback = tiled_divide(make_layout(cluster_shape_fallback), make_tile(typename TiledMma::AtomThrID{}));

    auto tma_load_a = get_tma_load_a_instance(tensor_a, problem_shape, cluster_layout_vmnk);
    auto tma_load_b = get_tma_load_b_instance(tensor_b, problem_shape, cluster_layout_vmnk);
    auto tma_load_a_fallback = get_tma_load_a_instance(tensor_a, problem_shape, cluster_layout_vmnk_fallback);
    auto tma_load_b_fallback = get_tma_load_b_instance(tensor_b, problem_shape, cluster_layout_vmnk_fallback);

    static_assert(size(typename decltype(tma_load_a)::ThrID{}) == size(AtomThrShapeMNK{}));
    static_assert(size(typename decltype(tma_load_b)::ThrID{}) == size(AtomThrShapeMNK{}));

    return {
      tma_load_a,
      tma_load_b,
      tma_load_a_fallback,
      tma_load_b_fallback,
      hw_info.cluster_shape_fallback
    };
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      Arguments const& args) {
    // Activation and Filter channel mode extents much match
    bool implementable = true;
    // channel mode is major
    {
      const bool check = problem_shape.stride_A[NumTensorDimensions-1] == 1;
#if (! defined(__CUDA_ARCH__)) && (CUTLASS_DEBUG_TRACE_LEVEL > 0)
      if (not check) {
        const auto offending_stride =
          problem_shape.stride_A[NumTensorDimensions-1];
        std::ostringstream os;
        os << "CollectiveConv::can_implement: "
          "problem_shape.stride_A[NumTensorDimensions-1 = "
          << (NumTensorDimensions-1) << "] = "
          << offending_stride << " != 1";
        CUTLASS_TRACE_HOST( os.str() );
      }
#endif
      implementable &= check;
    }

    {
      const bool check = problem_shape.stride_B[NumTensorDimensions-1] == 1;
#if (! defined(__CUDA_ARCH__)) && (CUTLASS_DEBUG_TRACE_LEVEL > 0)
      if (not check) {
        const auto offending_stride =
          problem_shape.stride_B[NumTensorDimensions-1];
        std::ostringstream os;
        os << "CollectiveConv::can_implement: "
          "problem_shape.stride_B[NumTensorDimensions-1 = "
          << (NumTensorDimensions-1) << "] = "
          << offending_stride << " != 1\n";
        CUTLASS_TRACE_HOST( os.str() );
      }
#endif
      implementable &= check;
    }

    {
      const auto & traversal_stride  = problem_shape.traversal_stride;
      for (auto stride: traversal_stride) {
       implementable &= (stride >= 1 && stride <= 8);
      }
    }

    if constexpr (ConvOp == conv::Operator::kDgrad && not is_strided_dgrad) {
      const auto & traversal_stride  = problem_shape.traversal_stride;
      for (auto stride: traversal_stride) {
        implementable &= (stride == 1);
      }
    }

    constexpr int tma_alignment_bits = 128;
    // A extents.
    auto shape_A_orig = problem_shape.get_shape_A();
    // B extents.
    auto shape_B_orig = problem_shape.get_shape_B();

    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    {
      const bool check = cutlass::detail::check_alignment<min_tma_aligned_elements_A>(shape_A_orig, StrideA{});
      if (not check) {
        CUTLASS_TRACE_HOST("A shape and/or strides have alignment issue.");
      }
      implementable &= check;
    }

    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    {
      const bool check = cutlass::detail::check_alignment<min_tma_aligned_elements_B>(shape_B_orig, StrideB{});
      if (not check) {
        CUTLASS_TRACE_HOST("B shape and/or strides have alignment issue.");
      }
      implementable &= check;
    }

    if (not implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
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
      // Check valid filter offsets for TMA_LOAD_IM2COL, unsigned int ranging from [0, offset_limit]
      constexpr int32_t offset_limit = (1 << (16 / NumSpatialDimensions)) - 1;
      auto flt_data = (ConvOp == conv::Operator::kWgrad) ? problem_shape.shape_C : problem_shape.shape_B;
      for (int i = 0; i < problem_shape.RankS; ++i) {
        // flt_data array contains [K, T, R, S, C], so pure filter [T, R, S] starts from the second position in the array
        implementable = implementable && ((flt_data[i+1] - 1) * problem_shape.dilation[i] >= 0)
                                      && ((flt_data[i+1] - 1) * problem_shape.dilation[i] <= offset_limit);
      }

      if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: tensor coordinate offset values don't meet requirements for TMA LOAD IM2COL.\n");
        return false;
      }
    }

    // Wgrad kernels don't support non-packed output strides, non-packed tensor A stride (linearized)
    if constexpr (ConvOp == conv::Operator::kWgrad) {

      const auto & input_shape  = problem_shape.shape_A;
      const auto & input_stride  = problem_shape.stride_A;

      implementable &= input_stride[ProblemShape::RankT - 1] == 1;
      int64_t input_shape_size = 1;
      for (int i = ProblemShape::RankT - 2; i >= 0; --i) {
        input_shape_size *= input_shape[i + 1];
        implementable &= input_stride[i] == input_shape_size;
      }

      const auto & output_shape  = problem_shape.shape_C;
      const auto & output_stride  = problem_shape.stride_C;

      implementable &= output_stride[ProblemShape::RankT - 1] == 1;
      int64_t output_shape_size = 1;
      for (int i = ProblemShape::RankT - 2; i >= 0; --i) {
        output_shape_size *= output_shape[i + 1];
        implementable &= output_stride[i] == output_shape_size;
      }

      if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Wgrad kernels don't support non-packed output strides.\n");
        return false;
      }
    }

    // Conv kernels only support cross correlation mode currently.
    {
      implementable &= problem_shape.mode == cutlass::conv::Mode::kCrossCorrelation;

      if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Conv kernels only support cross correlation mode currently.\n");
        return false;
      }
    }

    // When groups > 1, it should be a Grouped Conv.
    if (problem_shape.groups > 1) {
      implementable &= TileShapeMNKLRank > 3;

      if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Only Grouped Conv can support groups > 1.\n");
        return false;
      }
    }

    // Only support Grouped Wgrad currently.
    if constexpr (TileShapeMNKLRank > 3) {
      implementable &= ConvOp == conv::Operator::kWgrad;

      if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Grouped Conv Only support Grouped Wgrad currently.\n");
        return false;
      }
    }

    // Grouped Wgrad channel check.
    if constexpr (is_grouped_wgrad) {

      int input_K = size<0>(problem_shape.get_shape_A());
      int input_C = size<0>(problem_shape.get_shape_B());

      implementable &= input_K == input_C;

      if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Grouped Conv's input K and input C do not match.\n");
        return false;
      }

      int output_K = size<0>(problem_shape.get_shape_C());
      int output_C = size<1,0>(problem_shape.get_shape_C());

      implementable &= input_K == output_K;
      implementable &= input_C == output_C * problem_shape.groups;

      if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Grouped Wgrad's input and output K,C and groups do not match\n");
        return false;
      }

      constexpr int Tile_N = size<1>(TileShape{});
      constexpr int GroupsPerTile = size<3>(TileShapeMNKL_{});

      implementable &= Tile_N / GroupsPerTile == input_C / problem_shape.groups;

      if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Grouped Wgrad's Tile_N, GroupsPerTile and input_C, groups do not match.\n");
        return false;
      }
    }

    // The extents of linearized problem shape should be int32_t type(maximum is 2^31-1).
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
  CUTLASS_DEVICE void
  prefetch_tma_descriptors() {
    cute::prefetch_tma_descriptor(observed_tma_load_a_->get_tma_descriptor());
    cute::prefetch_tma_descriptor(observed_tma_load_b_->get_tma_descriptor());
  }

  /// Construct A Single Stage's Accumulator Shape
  CUTLASS_DEVICE static auto
  partition_accumulator_shape() {
    auto acc_shape = partition_shape_C(TiledMma{}, take<0,2>(TileShape{}));  // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N)

    return acc_shape;
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class GTensorA, class GTensorB,
    class GTensorPartitionedA, class GTensorPartitionedB,
    class STensorA, class STensorB,
    class TileCoordMNKL,
    class KTileIterator
  >
  CUTLASS_DEVICE auto
  load(
      Params const& params,
      MainloopPipeline pipeline,
      MainloopPipelineState mainloop_pipe_producer_state,
      cute::tuple<GTensorA, GTensorB,
                  GTensorPartitionedA, GTensorPartitionedB,
                  STensorA, STensorB,
                  uint16_t, uint16_t> const& load_inputs,
      TileCoordMNKL const& cta_coord_mnkl,
      KTileIterator k_tile_iter, int k_tile_count) {

    auto [unused_gA, unused_gB,
          tAgA_mk, tBgB_nk, tAsA, tBsB,
          mcast_mask_a, mcast_mask_b] = load_inputs;

    // slice out the work coord from partitioned tensors
    Tensor tAgA = tAgA_mk(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _);
    auto tensor_b_coord = get<1>(cta_coord_mnkl);
    if constexpr (is_grouped_wgrad) {
      // in grouped wgrad, tensor A = NZPQK, tensor B = NDHWC, tensor C = KTRSc, where C = G*c, c = channel_per_group = 8,16,32.
      // CTA Tiling follows output tensor KTRSc. So cta_size_m = K/CTA_TILE_M. cta_size_n = T*R*S*ceil(c/CTA_TILE_N) = T*R*S*1 = T*R*S.
      // tensor_a_coord = K_idx = cta_coord_m.
      // tensor_b_coord = TRS_idx * C/CTA_TILE_N + C_idx = cta_coord_n * get<1,0>(shape(tBgB_nk) + cta_coord_m,
      // because K == C and CTA_TILE_M == CTA_TILE_N => C_idx = K_idx = cta_coord_m.
      tensor_b_coord = get<0>(cta_coord_mnkl) + get<1>(cta_coord_mnkl) * get<1,0>(shape(tBgB_nk));
    }
    Tensor tBgB = tBgB_nk(_, tensor_b_coord, _);

    auto barrier_token = pipeline.producer_try_acquire(mainloop_pipe_producer_state);

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // LOCK mainloop_pipe_producer_state for _writing_
      pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);

      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(mainloop_pipe_producer_state);

      int write_stage = mainloop_pipe_producer_state.index();
      ++mainloop_pipe_producer_state;
      barrier_token = pipeline.producer_try_acquire(mainloop_pipe_producer_state);

      if constexpr (is_strided_dgrad) {
        // construct gemm-k tile coord for gB
        auto [conv_k, flt_coord, out_coord] = *k_tile_iter;
        auto gemm_k_tile = prepend(flt_coord, conv_k); // (k,s,r,t)

        // gA doesn't have a gemm-k (k,s,r,t) iterator mode because it's not an im2col tensor
        auto offset_kqpzn = append(prepend(out_coord, _0{}),_0{}); // (k,q,p,z,n)
        auto tAgA_offset = make_tensor(tAgA.data() + offset_kqpzn, tAgA.layout()); // (TMA, k)

        if (cute::elect_one_sync()) {
          copy(observed_tma_load_a_->with(*tma_barrier, mcast_mask_a), tAgA_offset(_,conv_k), tAsA(_,write_stage));
          copy(observed_tma_load_b_->with(*tma_barrier, mcast_mask_b), tBgB(_,gemm_k_tile)  , tBsB(_,write_stage));
        }
      }
      else {
        if (cute::elect_one_sync()) {
          copy(observed_tma_load_a_->with(*tma_barrier, mcast_mask_a), tAgA(_,*k_tile_iter), tAsA(_,write_stage));
          copy(observed_tma_load_b_->with(*tma_barrier, mcast_mask_b), tBgB(_,*k_tile_iter), tBsB(_,write_stage));
        }
      }

      --k_tile_count;
      ++k_tile_iter;
  }

    return cute::make_tuple(mainloop_pipe_producer_state, k_tile_iter);
  }

  /// Set up the data needed by this collective for load.
  /// Return tuple element contain
  /// gA_mk - The tiled tma tensor for input A
  /// gB_nk - The tiled tma tensor for input B
  /// tAsA - partitioned smem tensor for A
  /// tBsB - partitioned smem tensor for B
  /// mcast_mask_a - tma multicast mask for A
  /// mcast_mask_b - tma multicast mask for B
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      Params const& params,
      TensorStorage& shared_tensors) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // Represent the full tensors -- get these from TMA
    auto K_A = conditional_return<is_strided_dgrad>(get<0>(K), K);
    Tensor mA_mk = observed_tma_load_a_->get_tma_tensor(make_shape(M, K_A));
    Tensor mB_nk = observed_tma_load_b_->get_tma_tensor(make_shape(N, K));

    // Tile the tensors and defer the slice
    Tensor gA_mk = local_tile(mA_mk, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});         // (BLK_M, BLK_K, m, k)
    Tensor gB_nk = local_tile(mB_nk, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});         // (BLK_N, BLK_K, n, k)

    // Partition for this CTA
    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));

    Tensor tCgA_mk = cta_mma.partition_A(gA_mk);          // (MMA, MMA_M, MMA_K, m, k)
    Tensor tCgB_nk = cta_mma.partition_B(gB_nk);          // (MMA, MMA_N, MMA_K, n, k)

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});  // (MMA,MMA_N,MMA_K,PIPE)

    // Define the CTA-in-cluster Layout and Coord
    Layout cta_layout_mnk  = make_layout(cluster_shape_);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster_);

    // Project the cta_layout for tma_a along the n-modes
    auto [tAgA_mk, tAsA] = tma_partition(*observed_tma_load_a_,
                                    get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                    group_modes<0,3>(sA), group_modes<0,3>(tCgA_mk));

    // Project the cta_layout for tma_b along the m-modes
    auto [tBgB_nk, tBsB] = tma_partition(*observed_tma_load_b_,
                                    get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
                                    group_modes<0,3>(sB), group_modes<0,3>(tCgB_nk));

    // TMA Multicast Masks
    uint16_t mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

    return cute::make_tuple(
        gA_mk, gB_nk,                        // for scheduler
        tAgA_mk, tBgB_nk, tAsA, tBsB,        // for input tensor values
        mcast_mask_a, mcast_mask_b);         // multicast masks
  }

  /// Perform a Producer Epilogue to prevent early exit of ctas in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, MainloopPipelineState mainloop_pipe_producer_state) {
    // Issue the epilogue waits
    /* This helps avoid early exit of ctas in Cluster
      * Waits for all stages to either be released (all
      * Consumer UNLOCKs), or if the stage was never used
      * then would just be acquired since the phase was
      * still inverted from make_producer_start_state
      */
    pipeline.producer_tail(mainloop_pipe_producer_state);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgEngine, class FrgLayout,
    class FragmentA, class FragmentB
  >
  CUTLASS_DEVICE auto
  mma(MainloopPipeline pipeline,
      MainloopPipelineState mainloop_pipe_consumer_state,
      cute::Tensor<FrgEngine, FrgLayout>& accumulators,
      cute::tuple<TiledMma, FragmentA, FragmentB> const& mma_inputs,
      int k_tile_count)
  {
    static_assert(is_tmem<FrgEngine>::value, "Accumulator must be tmem resident.");
    static_assert(rank(FrgLayout{}) == 3, "Accumulator must be MMA-partitioned: (MMA, MMA_M, MMA_N)");

    auto [tiled_mma, tCrA, tCrB] = mma_inputs;

    uint32_t skip_wait = k_tile_count <= 0;
    auto barrier_token = pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

    //
    // PIPELINED MAIN LOOP
    //
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // WAIT on mainloop_pipe_consumer_state until its data are available (phase bit flips from mainloop_pipe_consumer_state.phase() value)
      pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);

      // Compute on k_tile
      int read_stage = mainloop_pipe_consumer_state.index();
      // Save current mainlop pipeline read state
      auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;

      // Advance mainloop_pipe
      ++mainloop_pipe_consumer_state;
      --k_tile_count;
      skip_wait = k_tile_count <= 0;
      // Peek at next iteration
      barrier_token = pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

      // Unroll the K mode manually so we can set scale C to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accumulators);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      pipeline.consumer_release(curr_mainloop_pipe_consumer_state);
    }

    return mainloop_pipe_consumer_state;
  }

  CUTLASS_DEVICE auto
  mma_init(TensorStorage& shared_tensors) const {
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    TiledMma tiled_mma;

    // Allocate "fragments/descriptors" for A and B matrices
    Tensor tCrA = tiled_mma.make_fragment_A(sA);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = tiled_mma.make_fragment_B(sB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sA));                                     // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sB));                                     // PIPE
    return cute::make_tuple(tiled_mma, tCrA, tCrB);
  }

private:

  typename Params::TMA_A const* observed_tma_load_a_ = nullptr;
  typename Params::TMA_B const* observed_tma_load_b_ = nullptr;

  ClusterShape cluster_shape_;
  uint32_t block_rank_in_cluster_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
