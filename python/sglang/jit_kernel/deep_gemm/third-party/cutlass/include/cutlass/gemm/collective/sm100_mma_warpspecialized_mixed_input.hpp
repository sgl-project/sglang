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
#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cutlass/detail/cluster.hpp"
#include "cutlass/detail/collective/mixed_input_utils.hpp"
#include "cutlass/detail/sm100_mixed_dtype_blockwise_layout.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/mma_sm100.hpp"
#include "cutlass/trace.h"
#include "cutlass/kernel_hardware_info.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop for Mixed Input Kernels
template <
  int Load2TransformPipelineStageCount_,
  int Transform2MmaPipelineStageCount_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape,
  class TileShape_,
  class ElementAOptionalTuple_,
  class StridePairA_,
  class ElementBOptionalTuple_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomsA_,
  class CopyAtomsA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomsB_,
  class CopyAtomsB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm100TmaUmmaWarpSpecializedMixedInput<
      Load2TransformPipelineStageCount_,
      Transform2MmaPipelineStageCount_,
      SchedulerPipelineStageCount_,
      AccumulatorPipelineStageCount_,
      ClusterShape>,
    TileShape_,
    ElementAOptionalTuple_,
    StridePairA_,
    ElementBOptionalTuple_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomsA_,
    CopyAtomsA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomsB_,
    CopyAtomsB_,
    TransformB_>
{
public:
  //
  // Type Aliases
  //

  using ConversionMode = cutlass::detail::ConversionMode;
  // Determine MMA type: MMA_1SM vs MMA_2SM
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma_::ThrLayoutVMNK{})), _1, _1>;
  using DispatchPolicy = MainloopSm100TmaUmmaWarpSpecializedMixedInput<
                            Load2TransformPipelineStageCount_,
                            Transform2MmaPipelineStageCount_,
                            SchedulerPipelineStageCount_,
                            AccumulatorPipelineStageCount_,
                            ClusterShape>;
  using TileShape = TileShape_;
  using TiledMma = TiledMma_;
  using KernelSchedule = typename DispatchPolicy::Schedule;
  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));
  using ElementAOptionalTuple = ElementAOptionalTuple_;
  using ElementBOptionalTuple = ElementBOptionalTuple_;

private:

  template<class T> friend struct detail::MixedInputUtils;
  using CollectiveType = CollectiveMma<DispatchPolicy, TileShape_, 
                                       ElementAOptionalTuple, StridePairA_, 
                                       ElementBOptionalTuple, StrideB_,
                                       TiledMma_, 
                                       GmemTiledCopyA_, SmemLayoutAtomsA_, CopyAtomsA_,
                                       TransformA_,
                                       GmemTiledCopyB_, SmemLayoutAtomsB_, CopyAtomsB_,
                                       TransformB_>;
  using Utils = detail::MixedInputUtils<CollectiveType>;

  using ElementScaleA = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple_>;
  using ElementScaleB = detail::deduce_mixed_width_dtype_t<1, ElementBOptionalTuple>;
  using ElementZeroA = detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>;
  using ElementZeroB = detail::deduce_mixed_width_dtype_t<2, ElementBOptionalTuple>;

public:
  static_assert(cute::is_tuple<ElementAOptionalTuple>::value ^ cute::is_tuple<ElementBOptionalTuple>::value, 
    "Either A OR B must be a tuple. It must take the from {ElementOperand, [ElementScale],"
    "[ElementZero]}. Inputs in [] are optional.");
  
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;
  static constexpr bool IsATransformed = cute::is_tuple<ElementAOptionalTuple>::value;
  using ElementScale = cute::conditional_t<IsATransformed, ElementScaleA, ElementScaleB>;
  using ElementZero = cute::conditional_t<IsATransformed, ElementZeroA, ElementZeroB>;
  // For cases where we can't have a void type, we can use this to allow the code to compile when the scale / zero is void.
  using NonVoidElementScale = cute::conditional_t<cute::is_void_v<ElementScale>, float, ElementScale>;
  using NonVoidElementZero = cute::conditional_t<cute::is_void_v<ElementZero>, float, ElementZero>;

  using StrideA = cute::remove_cvref_t<decltype(get<0>(StridePairA_{}))>;
  using LayoutScale = cute::remove_cvref_t<decltype(get<1>(StridePairA_{}))>;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;

  static_assert((IsATransformed && cutlass::gemm::detail::is_k_major<StrideA>()) || 
                (!IsATransformed && cutlass::gemm::detail::is_k_major<StrideB>()),
                "The transformed type must be K-major.");

  static_assert(( IsATransformed && (sizeof(ElementB) == 2)) ||
                (!IsATransformed && (sizeof(ElementA) == 2)) ||
                (cutlass::gemm::detail::is_k_major<StrideA>() && 
                 cutlass::gemm::detail::is_k_major<StrideB>()), 
                "The unscaled element must be 2 bytes OR both inputs must be K-major");

  // Define A and B block shapes for reduced size TMA_LOADs
  using CtaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
  using CtaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));

  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementBMma = typename TiledMma::ValTypeB;

  using ElementAccumulator = typename TiledMma::ValTypeC;

  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using GmemTiledCopyScale = GmemTiledCopyA_;

  using SmemLayoutAtomsA = SmemLayoutAtomsA_;
  using SmemLayoutAtomsB = SmemLayoutAtomsB_;
  using CopyAtomsA = CopyAtomsA_;
  using CopyAtomsB = CopyAtomsB_;
  using SmemCopyAtomScale = Copy_Atom<cute::AutoVectorizingCopy, NonVoidElementScale>;

  using SmemLayoutAtomA = typename SmemLayoutAtomsA::InputLayoutAtom;
  using SmemLayoutAtomACompute = typename SmemLayoutAtomsA::ComputeLayoutAtom;
  using SmemLayoutAtomB = typename SmemLayoutAtomsB::InputLayoutAtom;
  using SmemLayoutAtomBCompute = typename SmemLayoutAtomsB::ComputeLayoutAtom;

  using InputCopyAtomA = typename CopyAtomsA::InputCopyAtom;
  using ComputeCopyAtomA = typename CopyAtomsA::ComputeCopyAtom;
  using InputCopyAtomB = typename CopyAtomsB::InputCopyAtom;
  using ComputeCopyAtomB = typename CopyAtomsB::ComputeCopyAtom;

  // We must ensure the type to be scaled goes to RF
  static constexpr bool SwapAB = !IsATransformed;
  using InternalSmemLayoutAtomA = cute::conditional_t<!SwapAB, SmemLayoutAtomA, SmemLayoutAtomB>;
  using InternalSmemLayoutAtomB = cute::conditional_t<!SwapAB, SmemLayoutAtomB, SmemLayoutAtomA>;
  using InternalSmemLayoutAtomACompute = cute::conditional_t<!SwapAB, SmemLayoutAtomACompute, SmemLayoutAtomBCompute>;
  using InternalSmemLayoutAtomBCompute = cute::conditional_t<!SwapAB, SmemLayoutAtomBCompute, SmemLayoutAtomACompute>;

  using InternalInputCopyAtomA   = cute::conditional_t<!SwapAB, InputCopyAtomA, InputCopyAtomB>;
  using InternalInputCopyAtomB   = cute::conditional_t<!SwapAB, InputCopyAtomB, InputCopyAtomA>;
  using InternalComputeCopyAtomA   = cute::conditional_t<!SwapAB, ComputeCopyAtomA, ComputeCopyAtomB>;
  using InternalComputeCopyAtomB   = cute::conditional_t<!SwapAB, ComputeCopyAtomB, ComputeCopyAtomA>;

  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using ConvertedElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using ConvertedElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;
  using RealSwappedElementA = cute::conditional_t<!SwapAB, ElementA, ElementB>;
  using RealSwappedElementB = cute::conditional_t<!SwapAB, ElementB, ElementA>;
  using SwappedElementA = cute::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
  using SwappedElementB = cute::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;
  using SwappedStrideA = cute::conditional_t<!SwapAB, StrideA, StrideB>;
  using SwappedStrideB = cute::conditional_t<!SwapAB, StrideB, StrideA>;
  using InternalSwappedStrideA = cute::conditional_t<!SwapAB, InternalStrideA, InternalStrideB>;
  using InternalSwappedStrideB = cute::conditional_t<!SwapAB, InternalStrideB, InternalStrideA>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using InternalTransformA  = cute::conditional_t<!SwapAB, TransformA, TransformB>;
  using InternalTransformB  = cute::conditional_t<!SwapAB, TransformB, TransformA>;

  static constexpr int IsSubbyteA = cute::sizeof_bits_v<SwappedElementA> < 8;
  using TmaElementA = cute::conditional_t<IsSubbyteA, uint8_t, SwappedElementA>;
  using TmaElementScale = uint_bit_t<sizeof_bits_v<NonVoidElementScale> >; // in case we have array. translating to uint to satisfy tma descriptor's specialization

  using ArchTag = typename DispatchPolicy::ArchTag;
  static_assert(cute::is_same_v<ElementAMma, cutlass::bfloat16_t> || cute::is_same_v<ElementAMma, cutlass::half_t> || cute::is_same_v<ElementAMma, cutlass::float_e4m3_t>, 
         "Compute type A should be cutlass::bfloat16_t or cutlass::half_t or cutlass::float_e4m3_t");

  using Load2TransformPipeline = cutlass::PipelineTmaTransformAsync<
                             DispatchPolicy::Load2TransformPipelineStageCount,
                             AtomThrShapeMNK>;
  using Load2TransformPipelineState = typename Load2TransformPipeline::PipelineState;

  using Load2MmaPipeline = cutlass::PipelineTmaUmmaAsync<
                             DispatchPolicy::Load2TransformPipelineStageCount,
                             ClusterShape,
                             AtomThrShapeMNK>;
  using Load2MmaPipelineState = typename Load2MmaPipeline::PipelineState;

  using Transform2MmaPipeline = cutlass::PipelineUmmaConsumerAsync<
                              DispatchPolicy::Transform2MmaPipelineStageCount,
                              AtomThrShapeMNK>;
  using Transform2MmaPipelineState = typename Transform2MmaPipeline::PipelineState;

  using Mma2AccumPipeline =  cutlass::PipelineUmmaAsync<
                              DispatchPolicy::Schedule::AccumulatorPipelineStageCount,
                              AtomThrShapeMNK>;
  using Mma2AccumPipelineState = typename Mma2AccumPipeline::PipelineState;


  static constexpr int ScaleGranularityMN = size<0,0>(LayoutScale{});
  static constexpr int ScaleGranularityK = size<1,0>(LayoutScale{});
  using ScaleConfig = cutlass::detail::Sm100MixedInputBlockwiseScaleConfig<
      ScaleGranularityMN, 
      ScaleGranularityK>; 
 
  using ScaleTileShape = cute::conditional_t<!SwapAB, 
          decltype(make_shape(size<0>(TileShape{}), size<2>(TileShape{}))), 
          decltype(make_shape(size<1>(TileShape{}), size<2>(TileShape{})))>;

  static constexpr int ScaleTileShape_MN = get<0>(ScaleTileShape{});

  static constexpr int ScaleK = get<1>(ScaleTileShape{}) / ScaleGranularityK;

  using SmemLayoutAtomScale = decltype(ScaleConfig::smem_atom_layout_scale(ScaleTileShape{})); 

  static_assert(cute::rank(InternalSmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(InternalSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(InternalSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(InternalSmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(InternalSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(InternalSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomScale{}) == 2, "SmemLayoutAtomScale must be rank 2");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must equal the tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must evenly divide tile k shape.");

  // Thread Counts
  static constexpr uint32_t NumTransformationThreads = 128;
  static constexpr uint32_t NumAccumThreads = 128; //Maintains compatibility with input_transform kernel

  // Get the Algorithm parameters
  constexpr static int AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
  constexpr static int StagesPerTile = size<2>(CtaShapeA_MK{});

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert(((size<0,0>(CtaShapeA_MK{}) * size<1>(CtaShapeA_MK{})) % size<0>(SmemLayoutAtomACompute{})) == 0, "SmemLayoutAtomCompute must evenly divide tile shape.");
  static_assert(((size<0,1>(CtaShapeA_MK{}) * size<2>(CtaShapeA_MK{})) % size<1>(SmemLayoutAtomACompute{})) == 0, "SmemLayoutAtomCompute must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert(((size<0,0>(CtaShapeB_NK{}) * size<1>(CtaShapeB_NK{})) % size<0>(SmemLayoutAtomBCompute{})) == 0, "SmemLayoutAtomCompute must evenly divide tile shape.");
  static_assert(((size<0,1>(CtaShapeB_NK{}) * size<2>(CtaShapeB_NK{})) % size<1>(SmemLayoutAtomBCompute{})) == 0, "SmemLayoutAtomCompute must evenly divide tile shape.");

  // Tile along K mode first before tiling over MN. PIPE mode last as usual.
  // This maximizes TMA boxes due to better smem-K vectorization, reducing total issued TMAs.
  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomA{},
      append(CtaShapeA_MK{}, Int<DispatchPolicy::Load2TransformPipelineStageCount>{}),
             (cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{})));

  using SmemLayoutACompute = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomACompute{},
      append(CtaShapeA_MK{}, Int<DispatchPolicy::Transform2MmaPipelineStageCount>{}),
             (cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{})));

  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(CtaShapeB_NK{}, Int<DispatchPolicy::Load2TransformPipelineStageCount>{}),
             (cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{})));

  using SmemLayoutScale = decltype(make_layout(
    append(shape(SmemLayoutAtomScale{}), Int<DispatchPolicy::Load2TransformPipelineStageCount>{}),
    append(stride(SmemLayoutAtomScale{}), size(filter_zeros(SmemLayoutAtomScale{})))
  ));

  static_assert(DispatchPolicy::Load2TransformPipelineStageCount >= 2 && DispatchPolicy::Load2TransformPipelineStageCount >= 2,
                "Specialization requires Stages set to value 2 or more.");
  static_assert((cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value ||
                 cute::is_base_of<cute::UMMA::tmem_frg_base,      typename TiledMma::FrgTypeA>::value  ) &&
                 cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                 "MMA atom must A operand from SMEM or TMEM and B operand from SMEM for this mainloop.");
  static_assert((cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>),
                 "GmemTiledCopyA - invalid TMA copy atom specified.");

private:
  static constexpr ConversionMode 
  get_conversion_mode() {
    if constexpr (cute::is_void_v<ElementScale>) {
      return ConversionMode::DirectConvert;
    } 
    else if constexpr (cute::is_void_v<ElementZero>) {
      return ConversionMode::ConvertAndScale;
    }
    else {
      return ConversionMode::ConvertAndScaleWithZero;
    }
  }

public:
  static constexpr ConversionMode KernelConversionMode = get_conversion_mode();
  static constexpr bool ModeHasScales = KernelConversionMode == ConversionMode::ConvertAndScale ||
                                        KernelConversionMode == ConversionMode::ConvertAndScaleWithZero;
  static constexpr bool UseScaleLookupTable = KernelConversionMode == ConversionMode::ConvertAndScale &&
                                              cutlass::detail::is_Array_v<ElementScale>;
  static constexpr size_t SmemAlignmentA = cutlass::detail::alignment_for_swizzle(SmemLayoutA{}); 

  static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(SmemLayoutB{});

  // Just pick the max alignment of A and B since it is required to be at least 128B
  static constexpr size_t SmemAlignmentScale = cute::max(SmemAlignmentA, SmemAlignmentB);

  static_assert(SmemAlignmentA >= 128 and SmemAlignmentB >= 128, "Require at least 128B alignment");

  struct PipelineStorage {
    using Load2TransformPipelineStorage = typename Load2TransformPipeline::SharedStorage;
    alignas(16) Load2TransformPipelineStorage load2transform_pipeline;
    using Load2MmaPipelineStorage = typename Load2MmaPipeline::SharedStorage;
    alignas(16) Load2MmaPipelineStorage load2mma_pipeline;
    using Transform2MmaPipelineStorage = typename Transform2MmaPipeline::SharedStorage;
    alignas(16) Transform2MmaPipelineStorage transform2mma_pipeline;
    using Mma2AccumPipelineStorage = typename Mma2AccumPipeline::SharedStorage;
    alignas(16) Mma2AccumPipelineStorage mma2accum_pipeline;
  };

  struct SharedStorage {
    static constexpr int scale_elements = Utils::elements_per_smem_scale();
    static constexpr int zero_elements = Utils::elements_per_smem_zero();
    struct TensorStorage : cute::aligned_struct<128, _0> {

      struct TensorStorageUntransformed {
        alignas(512) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> smem_A;
        alignas(1024) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> smem_B;
        cute::ArrayEngine<NonVoidElementScale, scale_elements> smem_scale;
        cute::ArrayEngine<NonVoidElementZero, zero_elements> smem_zero;
      };

      struct TensorStorageTransformedAinSmem {
        // We require alignas(1024) here because the smem_ACompute may not be aligned to 1024 by default.
        // We need 1024B alignment of smem_ACompute because we are using Swizzle<3,4,3> here.
        // The Swizzle<3,4,3> aligns with 1024B. If we don't align the data, the compiler cannot deduce
        // the base pointer of the data.
        // This alignment allows us to perform the function swizzle(layout(i) * base_ptr).
        alignas(1024) cute::ArrayEngine<ElementAMma, cute::cosize_v<SmemLayoutACompute>> smem_ACompute;
      };

      union TensorStorageTransformedAinTmem {
        cute::ArrayEngine<ElementAMma, 1> smem_ACompute;  // No smem_ACompute
      };

      using TensorStorageTransformed = cute::conditional_t<
                                      cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value,
                                      TensorStorageTransformedAinSmem,
                                      TensorStorageTransformedAinTmem>;

      TensorStorageUntransformed input;
      TensorStorageTransformed compute;
    } tensors;

    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;

  // Different from other GEMM kernels, both CTAs should be aware of loads. Both CTAs will work on
  // loaded input A and B matrices to convert the data type
  static constexpr uint32_t TmaTransactionBytes_A = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementA>) + Utils::compute_tma_transaction_bytes_extra_transform();
  static constexpr uint32_t TmaTransactionBytes_B = cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>);
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytes_A + TmaTransactionBytes_B;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A{nullptr};
    StrideA dA{};
    ElementB const* ptr_B{nullptr};
    StrideB dB{};
    ElementScale const* ptr_S{nullptr};
    LayoutScale layout_S{};
    ElementZero const* ptr_Z{nullptr};
  };

  struct TMAScaleParams {
    using ClusterLayout_VMNK = decltype(tiled_divide(make_layout(conditional_return<IsDynamicCluster>(make_shape(uint32_t(0), uint32_t(0), Int<1>{}), ClusterShape{})),
                              make_tile(typename TiledMma::AtomThrID{})));

    using TMA_Scale = decltype(make_tma_atom(
        GmemTiledCopyScale{},
        make_tensor(static_cast<NonVoidElementScale const*>(nullptr), LayoutScale{}),
        SmemLayoutScale{}(_,_,cute::Int<0>{}),
        ScaleTileShape{},
        size<2>(ClusterLayout_VMNK{}))
    );

    TMA_Scale tma_load_scale;
    TMA_Scale tma_load_zero;
    
  };

  struct EmptyScaleParams {};

  // Device side kernel params
  struct Params : public cute::conditional_t<ModeHasScales, TMAScaleParams, EmptyScaleParams>  {

    using ClusterLayout_VMNK = decltype(tiled_divide(make_layout(conditional_return<IsDynamicCluster>(make_shape(uint32_t(0), uint32_t(0), Int<1>{}), ClusterShape{})),
                                                     make_tile(typename TiledMma::AtomThrID{})));

    using TMA_A = decltype(make_tma_atom_A_sm100<TmaElementA>(
        GmemTiledCopyA{},
        make_tensor(static_cast<ElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );

    using TMA_B = decltype(make_tma_atom_B_sm100<ElementB>(
        GmemTiledCopyB{},
        make_tensor(static_cast<ElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
    );

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_A tma_load_a_fallback;
    TMA_B tma_load_b_fallback;
    dim3 cluster_shape_fallback;

    uint32_t tma_transaction_bytes{TmaTransactionBytes};
    SwappedStrideA dA{};
    SwappedStrideB dB{};
  };

  CUTLASS_DEVICE
  CollectiveMma(Params const& params, ClusterShape cluster_shape, uint32_t block_rank_in_cluster)
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

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
    ProblemShape const& problem_shape, 
    Arguments const& args, 
    void* workspace, 
    cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{}) {
    (void) workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    Tensor tensor_a = make_tensor(args.ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(args.ptr_B, make_layout(make_shape(N,K,L), args.dB));

    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);
    // Cluster layout for TMA construction
    auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));

    auto cluster_shape_fallback = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape_fallback);
    // Cluster layout for TMA construction
    auto cluster_layout_vmnk_fallback = tiled_divide(make_layout(cluster_shape_fallback), make_tile(typename TiledMma::AtomThrID{}));

    typename Params::TMA_A tma_load_a = make_tma_atom_A_sm100<TmaElementA>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    typename Params::TMA_B tma_load_b = make_tma_atom_B_sm100<ElementB>(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    typename Params::TMA_A tma_load_a_fallback = make_tma_atom_A_sm100<TmaElementA>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

    typename Params::TMA_B tma_load_b_fallback = make_tma_atom_B_sm100<ElementB>(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

    uint32_t tma_transaction_bytes = TmaTransactionBytes;

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return { 
        {},
        tma_load_a, 
        tma_load_b, 
        tma_load_a_fallback, 
        tma_load_b_fallback, 
        hw_info.cluster_shape_fallback, 
        tma_transaction_bytes, 
        args.dA, args.dB };
    } 
    else if constexpr (ModeHasScales) {
      ElementScale const* ptr_S = args.ptr_S;
    
      Tensor tensor_scale = make_tensor(detail::get_logical_ptr(ptr_S), args.layout_S);
      typename Params::TMA_Scale tma_load_scale = make_tma_atom(
          GmemTiledCopyScale{},
          tensor_scale,
          SmemLayoutScale{}(_,_,cute::Int<0>{}),
          ScaleTileShape{},
          size<2>(cluster_layout_vmnk)
      );

      if constexpr(KernelConversionMode == ConversionMode::ConvertAndScale) {
        typename Params::TMAScaleParams scale_params{tma_load_scale, {}};
        return { 
          scale_params,
          tma_load_a, 
          tma_load_b, 
          tma_load_a_fallback, 
          tma_load_b_fallback, 
          hw_info.cluster_shape_fallback, 
          tma_transaction_bytes, 
          args.dA, args.dB };
      }
      else if constexpr(KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor tensor_zero = make_tensor(detail::get_logical_ptr(args.ptr_Z), args.layout_S);
        typename Params::TMA_Scale tma_load_zero = make_tma_atom(
            GmemTiledCopyScale{},
            tensor_zero,
            SmemLayoutScale{}(_,_,cute::Int<0>{}),
            ScaleTileShape{},
            size<2>(cluster_layout_vmnk));

        typename Params::TMAScaleParams scale_params{tma_load_scale, tma_load_zero};
        return { 
          scale_params,
          tma_load_a, 
          tma_load_b, 
          tma_load_a_fallback, 
          tma_load_b_fallback, 
          hw_info.cluster_shape_fallback, 
          tma_transaction_bytes, 
          args.dA, args.dB };
      } 
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in to_underlying_arguments.");
      }
    } 
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in to_underlying_arguments.");
    }
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {

    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    constexpr int tma_alignment_bits_A = cutlass::detail::get_input_alignment_bits<ElementA>();
    constexpr int tma_alignment_bits_B = cutlass::detail::get_input_alignment_bits<ElementB>();
    constexpr int tma_alignment_bits_S = cutlass::detail::get_input_alignment_bits<NonVoidElementScale>();

    constexpr int min_tma_aligned_elements_A = tma_alignment_bits_A / cutlass::sizeof_bits<ElementA>::value;
    bool check_aligned_A = cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits_B / cutlass::sizeof_bits<ElementB>::value;
    bool check_aligned_B = cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});

    bool check_aligned_S = true;
    bool check_aligned_Z = true;
    bool check_mode_args = true;

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      check_mode_args = check_mode_args && (args.ptr_S == nullptr);
      check_mode_args = check_mode_args && (args.ptr_Z == nullptr);
    } 
    else if constexpr (ModeHasScales) {
      constexpr int min_tma_aligned_elements_scale = tma_alignment_bits_S / cutlass::sizeof_bits<ElementScale>::value;
      check_aligned_S = cutlass::detail::check_alignment<min_tma_aligned_elements_scale>(args.layout_S);
      check_mode_args = check_mode_args && (args.ptr_S != nullptr);

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        check_mode_args = check_mode_args && (args.ptr_Z == nullptr);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        constexpr int min_tma_aligned_elements_zero = tma_alignment_bits_S / cutlass::sizeof_bits<ElementZero>::value;
        check_aligned_Z = cutlass::detail::check_alignment<min_tma_aligned_elements_zero>(args.layout_S);
        check_mode_args = check_mode_args && (args.ptr_Z != nullptr);
      } 
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in can_implement.");
      }
    }
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in can_implement.");
    }

    if (!check_mode_args) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Invalid arguments for the selected conversion mode.\n");
    }
    if (!check_aligned_A) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Tensor A meet the minimum alignment requirements for TMA.\n");
    }
    if (!check_aligned_B) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Tensor B meet the minimum alignment requirements for TMA.\n");
    }
    if (!check_aligned_S) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Tensor S (scale) meet the minimum alignment requirements for TMA.\n");
    }
    if (!check_aligned_Z) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Tensor Z (zeros) meet the minimum alignment requirements for TMA.\n");
    }

    return check_mode_args && check_aligned_A && check_aligned_B && check_aligned_S && check_aligned_Z;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE static void
  prefetch_tma_descriptors(Params const& params) {
    if constexpr (IsDynamicCluster) {
      dim3 cs = cute::cluster_shape();
      const bool is_fallback_cluster = (cs.x == params.cluster_shape_fallback.x && cs.y == params.cluster_shape_fallback.y);
      if (is_fallback_cluster) {
        cute::prefetch_tma_descriptor(params.tma_load_a_fallback.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_b_fallback.get_tma_descriptor());
      }
      else {
        cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
      }
    }
    else {
      cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
    }

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert);
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      cute::prefetch_tma_descriptor(params.tma_load_scale.get_tma_descriptor());
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      cute::prefetch_tma_descriptor(params.tma_load_scale.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_zero.get_tma_descriptor());
    }  
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in TMA prefetch.");
    }
  }

  /// Construct A Single Stage's Accumulator Shape
  CUTLASS_DEVICE auto
  partition_accumulator_shape() {
    auto acc_shape = partition_shape_C(TiledMma{}, take<0,2>(TileShape{}));  // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N)

    return acc_shape;
  }

  /// Produce the inputs to the transform threads by loading inputs from gmem -> smem
  template <
    class GTensorA, class GTensorB,
    class GTensorPartitionedA, class GTensorPartitionedB,
    class STensorA, class STensorB,
    class TileCoordMNKL,
    class KTileIterator,
    class... Ts
  >
  CUTLASS_DEVICE auto
  load_A(
      Params const& params,
      Load2TransformPipeline load2xform_pipeline,
      Load2TransformPipelineState load2xform_pipeline_state,
      cute::tuple<GTensorA, GTensorB,
                  GTensorPartitionedA, GTensorPartitionedB,
                  STensorA, STensorB,
                  uint16_t, uint16_t,
                  cute::tuple<Ts...>> const& load_inputs,
      TileCoordMNKL const& cta_coord_mnkl,
      KTileIterator k_tile_iter, int k_tile_count) {

    auto [unused_gA, unused_gB,
          tAgA_mkl, tBgB_nkl, tAsA, tBsB,
          mcast_mask_a, mcast_mask_b, extra_input_partitions] = load_inputs;

    // slice out the work coord from tiled tensors
    Tensor tAgA = tAgA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));

    uint32_t skip_wait = (k_tile_count <= 0);
    auto load2xform_pipeline_flag = load2xform_pipeline.producer_try_acquire(load2xform_pipeline_state, skip_wait);

    //Load2Mma and Load2Transform pipelines both have the same ProducerBarrierType
    using BarrierType = typename Load2TransformPipeline::ProducerBarrierType;

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {

      // LOCK mainloop_load2xform_pipeline_state for _writing_
      load2xform_pipeline.producer_acquire(load2xform_pipeline_state, load2xform_pipeline_flag);

      int tile_A_write_stage = load2xform_pipeline_state.index();

      BarrierType* load2xform_tma_barrier = load2xform_pipeline.producer_get_barrier(load2xform_pipeline_state);

      // Advance mainloop load2transform pipeline
      ++load2xform_pipeline_state;

      skip_wait = (k_tile_count <= 1);
      load2xform_pipeline_flag = load2xform_pipeline.producer_try_acquire(load2xform_pipeline_state, skip_wait);

      // TMA load for A k_tile
      copy(observed_tma_load_a_->with(*load2xform_tma_barrier, mcast_mask_a), tAgA(_,*k_tile_iter), tAsA(_,tile_A_write_stage));

      if constexpr (ModeHasScales) {
        auto tSgS_mkl = get<0>(extra_input_partitions);
        auto tSgS = tSgS_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
        auto tSsS = get<1>(extra_input_partitions);
        copy(params.tma_load_scale.with(*load2xform_tma_barrier, mcast_mask_a), tSgS(_,*k_tile_iter), tSsS(_,tile_A_write_stage));

        if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
          auto tZgZ_mkl = get<2>(extra_input_partitions);
          auto tZgZ = tZgZ_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
          auto tZsZ = get<3>(extra_input_partitions);
          copy(params.tma_load_zero.with(*load2xform_tma_barrier, mcast_mask_a), tZgZ(_,*k_tile_iter), tZsZ(_,tile_A_write_stage));
        }
      } 
      else {
        if constexpr (KernelConversionMode == ConversionMode::DirectConvert);
        else static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for TMA copy op.");
      }

      ++k_tile_iter;
    }


    return cute::make_tuple(load2xform_pipeline_state, k_tile_iter);

  }

  /// Produce the inputs to the transform threads by loading inputs from gmem -> smem
  template <
    class GTensorA, class GTensorB,
    class GTensorPartitionedA, class GTensorPartitionedB,
    class STensorA, class STensorB,
    class TileCoordMNKL,
    class KTileIterator,
    class... Ts
  >
  CUTLASS_DEVICE auto
  load_B(
      Params const& params,
      Load2MmaPipeline load2mma_pipeline,
      Load2MmaPipelineState load2mma_pipeline_state,
      cute::tuple<GTensorA, GTensorB,
                  GTensorPartitionedA, GTensorPartitionedB,
                  STensorA, STensorB,
                  uint16_t, uint16_t,
                  cute::tuple<Ts...>> const& load_inputs,
      TileCoordMNKL const& cta_coord_mnkl,
      KTileIterator k_tile_iter, int k_tile_count) {

    auto [unused_gA, unused_gB,
          tAgA_mkl, tBgB_nkl, tAsA, tBsB,
          mcast_mask_a, mcast_mask_b, extra_input_partitions] = load_inputs;

    // slice out the work coord from tiled tensors
    Tensor tBgB = tBgB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    uint32_t skip_wait = (k_tile_count <= 0);
    auto load2mma_pipeline_flag = load2mma_pipeline.producer_try_acquire(load2mma_pipeline_state, skip_wait);

    //Load2Mma and Load2Transform pipelines both have the same ProducerBarrierType
    using BarrierType = typename Load2TransformPipeline::ProducerBarrierType;

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {

      // LOCK mainloop_load2mma_pipeline_state for _writing_
      load2mma_pipeline.producer_acquire(load2mma_pipeline_state, load2mma_pipeline_flag);

      int tile_B_write_stage = load2mma_pipeline_state.index();

      BarrierType* load2mma_tma_barrier = load2mma_pipeline.producer_get_barrier(load2mma_pipeline_state);

      // Advance mainloop load2mma pipeline
      ++load2mma_pipeline_state;

      skip_wait = (k_tile_count <= 1);
      load2mma_pipeline_flag = load2mma_pipeline.producer_try_acquire(load2mma_pipeline_state, skip_wait);

      // TMA load for B k_tile
      copy(observed_tma_load_b_->with(*load2mma_tma_barrier, mcast_mask_b), tBgB(_,*k_tile_iter), tBsB(_,tile_B_write_stage));

      ++k_tile_iter;
    }

    return cute::make_tuple(load2mma_pipeline_state, k_tile_iter);

  }

  /// Set up the data needed by this collective for load.
  /// Returned tuple must contain at least two elements, with the first two elements being:
  /// gA_mkl - The tiled tensor for input A
  /// gB_nkl - The tiled tensor for input B
  // Other inputs needed for load(): partitioned AB tensors for gmem and smem, and mcast masks
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      Params const& params,
      TensorStorage& shared_storage) const {
    auto [gA_mkl, gB_nkl] = tile_input_tensors(params, problem_shape_MNKL);

    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));

    Tensor tCgA_mkl = cta_mma.partition_A(gA_mkl);          // (MMA, MMA_M, MMA_K, m, k, l)
    Tensor tCgB_nkl = cta_mma.partition_B(gB_nkl);          // (MMA, MMA_N, MMA_K, n, k, l)

    Tensor sA = make_tensor(make_smem_ptr(shared_storage.input.smem_A.begin()), SmemLayoutA{});  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.input.smem_B.begin()), SmemLayoutB{});  // (MMA,MMA_N,MMA_K,PIPE)

    // Define the CTA-in-cluster Layout and Coord
    Layout cta_layout_mnk  = make_layout(cluster_shape_);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster_);

    // Project the cta_layout for tma_a along the n-modes
    auto [tAgA_mkl, tAsA] = tma_partition(*observed_tma_load_a_,
                                      get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                      group_modes<0,3>(sA), group_modes<0,3>(tCgA_mkl));

    // Project the cta_layout for tma_b along the m-modes
    auto [tBgB_nkl, tBsB] = tma_partition(*observed_tma_load_b_,
                                      get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
                                      group_modes<0,3>(sB), group_modes<0,3>(tCgB_nkl));

    // TMA Multicast Masks
    uint16_t mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::make_tuple(
          gA_mkl, gB_nkl,                        // for scheduler
          tAgA_mkl, tBgB_nkl, tAsA, tBsB,        // for input tensor values
          mcast_mask_a, mcast_mask_b,            // multicast masks
          cute::make_tuple());           
    }
    else if constexpr (ModeHasScales) {
      // Separate out problem shape for convenience
      auto [M,N,K,L] = problem_shape_MNKL;

      Tensor mS_mkl = params.tma_load_scale.get_tma_tensor(shape(LayoutScale{}));
      Tensor gS_mkl = local_tile(mS_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});

      Tensor sS  = make_tensor(make_smem_ptr(shared_storage.input.smem_scale.begin()), SmemLayoutScale{});

      Tensor tCgS_mkl = cta_mma.partition_A(gS_mkl);          // (MMA, MMA_M, MMA_K, m, k, l)
      Tensor tCsS = cta_mma.partition_A(sS);

      // Project the cta_layout for tma_scale along the n-modes
      auto [tSgS_mkl, tSsS] = tma_partition(params.tma_load_scale,
                                      get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                      group_modes<0,3>(tCsS), group_modes<0,3>(tCgS_mkl));

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(
          gA_mkl, gB_nkl,                        // for scheduler
          tAgA_mkl, tBgB_nkl, tAsA, tBsB,        // for input tensor values
          mcast_mask_a, mcast_mask_b,            // multicast masks
          cute::make_tuple(tSgS_mkl, tSsS));
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor mZ_mkl = params.tma_load_scale.get_tma_tensor(shape(LayoutScale{}));
        Tensor gZ_mkl = local_tile(mS_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
        Tensor sZ  = make_tensor(make_smem_ptr(shared_storage.input.smem_zero.begin()), SmemLayoutScale{});

        Tensor tCgZ_mkl = cta_mma.partition_A(gZ_mkl);          // (MMA, MMA_M, MMA_K, m, k, l)

        Tensor tCsZ = cta_mma.partition_A(sZ);
        // Project the cta_layout for tma_scale along the n-modes
        auto [tZgZ_mkl, tZsZ] = tma_partition(params.tma_load_zero,
                                          get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                          group_modes<0,3>(tCsZ), group_modes<0,3>(tCgZ_mkl));
        return cute::make_tuple(
          gA_mkl, gB_nkl,                        // for scheduler
          tAgA_mkl, tBgB_nkl, tAsA, tBsB,        // for input tensor values
          mcast_mask_a, mcast_mask_b,            // multicast masks
          cute::make_tuple(tSgS_mkl, tSsS, tZgZ_mkl, tZsZ));
      }
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in load_init.");
      }
    }
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in load_init.");
    }

  }

  template<
    class KTileIterator, class Accumulator,
    class GTensorA, class DstCopyA, class SrcTensorA, class DstTensorA,
    class... Ts
  >
  CUTLASS_DEVICE auto
  transform(
      Load2TransformPipeline load2transform_pipeline,
      Load2TransformPipelineState load2transform_pipeline_consumer_state,
      Transform2MmaPipeline transform2mma_pipeline,
      Transform2MmaPipelineState transform2mma_pipeline_producer_state,
      Accumulator accumulators,
      cute::tuple<GTensorA, DstCopyA, SrcTensorA, DstTensorA,
                  cute::tuple<Ts...>> input_operands,
      KTileIterator k_tile_iter, int k_tile_count) {

    static_assert(cute::is_same_v<ElementAMma, ElementBMma>, "ElementAMma and ElementBMma types should be the same.");
    cutlass::arch::NamedBarrier transform_bar(NumTransformationThreads, cutlass::arch::ReservedNamedBarriers::TransformBarrier);

    // tAsA : (Copy,#Copy),MMA_Rest,MMA_M_Rest,MMA_K_Rest, SmemStages (In SMEM)
    // tAsACompute : (Copy,#Copy),MMA_Rest,MMA_M_Rest,MMA_K_Rest, SmemStages (In SMEM or TMEM)
    auto [unused_tAgA, dst_copy_A, tAsA, tAsACompute,
          partitioned_extra_info] = input_operands;

    // Create the tensors in registers
    auto tArA = make_tensor<ElementA>(tAsA(_,_,_,_,0).shape());  //(Copy,#Copy),MMA_Rest,MMA_M_Rest,MMA_K_Rest (Register)
    auto tArACompute = make_tensor<ElementAMma>(tAsA(_,_,_,_,0).shape());
    constexpr int K_BLOCK_MAX = size<3>(tArA);

    uint32_t skip_wait = (k_tile_count <= 0);
    auto load2transform_flag = load2transform_pipeline.consumer_try_wait(load2transform_pipeline_consumer_state, skip_wait);
    auto transform2mma_flag = transform2mma_pipeline.producer_try_acquire(transform2mma_pipeline_producer_state, skip_wait);

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {

      load2transform_pipeline.consumer_wait(load2transform_pipeline_consumer_state, load2transform_flag);

      transform2mma_pipeline.producer_acquire(transform2mma_pipeline_producer_state, transform2mma_flag);

      int load2transform_consumer_index = load2transform_pipeline_consumer_state.index(); // read stage
      int transform2mma_producer_index = transform2mma_pipeline_producer_state.index(); //write stage

      auto curr_load2transform_pipeline_consumer_state = load2transform_pipeline_consumer_state;

      // Copy the input A matrix from SMEM
      copy(AutoVectorizingCopy{}, tAsA(_,_,_,_,load2transform_consumer_index), tArA);
      // Copy scale/zero vector from SMEM
      Utils::copy_scale_zeros_for_transform(partitioned_extra_info, load2transform_consumer_index);

      // Loads from SMEM are done. Signal the mainloop load as early as possible
      transform_bar.sync();
      load2transform_pipeline.consumer_release(curr_load2transform_pipeline_consumer_state);

      auto curr_transform2mma_pipeline_producer_state = transform2mma_pipeline_producer_state;

      // Dequantize A with scale/zero in RF
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; k_block ++){
        Utils::dequantize_A_kblock_for_transform(tArA, tArACompute, partitioned_extra_info, k_block);
      }

      // Dequantized A is stored into either Smem or Tmem
      copy(dst_copy_A, tArACompute, tAsACompute(_,_,_,_,transform2mma_producer_index));

      // fence for SMEM writes
      cutlass::arch::fence_view_async_shared();
      if constexpr (is_tmem<decltype(tAsACompute)>::value) {
        // fence for TMEM writes if A operand is coming from TMEM
        cutlass::arch::fence_view_async_tmem_store();
      }

      // Let the MMA know we are done transforming
      transform2mma_pipeline.producer_commit(curr_transform2mma_pipeline_producer_state);
      // Next pipeline stage
      ++load2transform_pipeline_consumer_state;
      ++transform2mma_pipeline_producer_state;

      skip_wait = (k_tile_count <= 1);
      // Peek the next pipeline stage's barriers
      load2transform_flag = load2transform_pipeline.consumer_try_wait(load2transform_pipeline_consumer_state, skip_wait);
      transform2mma_flag = transform2mma_pipeline.producer_try_acquire(transform2mma_pipeline_producer_state, skip_wait);
    }
    return cute::make_tuple(load2transform_pipeline_consumer_state, transform2mma_pipeline_producer_state);
  }

  template<class ProblemShape_MNKL, class Accumulator>
  CUTLASS_DEVICE auto
  transform_init(
      Params const& params,
      ProblemShape_MNKL const& problem_shape_MNKL,
      Accumulator accumulators,
      TensorStorage& shared_storage) {

    auto [gA_mkl, gB_nkl] = tile_input_tensors(params, problem_shape_MNKL);

    Tensor sA_orig = make_tensor(make_smem_ptr(shared_storage.input.smem_A.begin()), SmemLayoutA{});
    Tensor sA = as_position_independent_swizzle_tensor(sA_orig);
    Tensor sACompute = make_tensor(make_smem_ptr(shared_storage.compute.smem_ACompute.begin()), SmemLayoutACompute{});

    Tensor sS = make_tensor(make_smem_ptr(shared_storage.input.smem_scale.begin()), SmemLayoutScale{}); 
    Tensor sZ = make_tensor(make_smem_ptr(shared_storage.input.smem_zero.begin()), SmemLayoutScale{}); 

    // Map input, compute, and fragment tensors to
    //   Copy strategies and partitioned tensors. These will become the input
    //   operands of the transform function. Depending on MMA atom type, the
    //   operands can reside in SMEM or TMEM
    auto setup_copy_ops = [&] (
        auto tensor_input,
        auto input_copy_atom,
        auto tensor_compute,
        auto make_fragment,
        auto compute_copy_atom) constexpr {
      auto fragment_compute = make_fragment(tensor_compute);
      if constexpr (cute::is_tmem<cute::remove_cvref_t<decltype(fragment_compute)>>::value) {
        // For M=128 with 2CTA MMA atoms, the TMEM tensor for A has a duplicated allocation.
        // Instead of allocation a 64x16 TMEM tensor, we have a 128x16 allocation
        // See: TmemAllocMode::Duplicated.
        Tensor tensor_input2x = [&] () constexpr {
        if constexpr (decltype(size<0,0>(fragment_compute) == Int<128>{} && size<0,0>(tensor_input) == Int<64>{})::value) {
          return make_tensor(tensor_input.data(),
                             logical_product(tensor_input.layout(),
                                             make_tile(make_tile(Layout<_2,_0>{},_),_,_,_))); 
          }
          else {
            return tensor_input;
          }
        }();

        fragment_compute.data() = accumulators.data().get() + cutlass::detail::find_tmem_tensor_col_offset(accumulators);
        // If operand comes from TMEM, create the TMEM_STORE based copy
        auto r2t_tiled_copy = make_tmem_copy(compute_copy_atom, fragment_compute(_,_,_,0));
        auto thr_r2t_tiled_copy = r2t_tiled_copy.get_slice(threadIdx.x % NumTransformationThreads);
        auto partitioned_tensor_input = thr_r2t_tiled_copy.partition_S(tensor_input2x); //(TMEM_STORE, TMEM_STORE_M, TMEM_STORE_N)
        auto partitioned_tensor_compute = thr_r2t_tiled_copy.partition_D(fragment_compute); //(TMEM_STORE, TMEM_STORE_M, TMEM_STORE_N)

        // Source copy is based on the source operand of TMEM_STORE copy.
        auto smem2reg_tiled_copy = make_tiled_copy_S(Copy_Atom<DefaultCopy, ElementA>{}, r2t_tiled_copy);
        return cute::make_tuple(smem2reg_tiled_copy, r2t_tiled_copy, partitioned_tensor_input, partitioned_tensor_compute);
      }
      else {
        auto tensor_compute_ind_sw = as_position_independent_swizzle_tensor(tensor_compute);
        auto r2s_tiled_copy = make_cotiled_copy(compute_copy_atom, Layout<Shape <_128,_8>, Stride<  _8,_1>>{},
                                                     tensor_compute(_,_,_,0).layout());

        auto smem2reg_tiled_copy = make_tiled_copy_S(input_copy_atom, r2s_tiled_copy);
        auto thr_r2s_tiled_copy = r2s_tiled_copy.get_slice(threadIdx.x % NumTransformationThreads);
        auto partitioned_tensor_input = thr_r2s_tiled_copy.partition_S(tensor_input); //(SMEM_STORE, SMEM_STORE_M, SMEM_STORE_N)

        auto partitioned_tensor_compute = thr_r2s_tiled_copy.partition_D(tensor_compute_ind_sw);//(SMEM_STORE, SMEM_STORE_M, SMEM_STORE_N)


        return cute::make_tuple(smem2reg_tiled_copy, AutoVectorizingCopy{}, partitioned_tensor_input, partitioned_tensor_compute);
      }
    };

    auto [src_copy_A, dst_copy_A, tAsA, tAsACompute] =
        setup_copy_ops(sA, InputCopyAtomA{}, sACompute, [&](auto &arg) {return TiledMma::make_fragment_A(arg);}, ComputeCopyAtomA{});

    // Partition of thread -> shared and thread -> RF
    auto fragment_compute = TiledMma::make_fragment_A(sACompute);
    fragment_compute.data() = accumulators.data().get() + cutlass::detail::find_tmem_tensor_col_offset(accumulators);
    auto r2t_tiled_copy = make_tmem_copy(ComputeCopyAtomA{}, fragment_compute(_,_,_,0));
    auto src_copy_scale = make_tiled_copy_S(Copy_Atom<DefaultCopy, ElementScale>{}, r2t_tiled_copy);

    auto partitioned_extra_info = Utils::partition_extra_transform_info(TiledMma{}, src_copy_scale, shared_storage);

    return cute::make_tuple(gA_mkl, dst_copy_A, tAsA, tAsACompute,
                            partitioned_extra_info);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgEngine, class FrgLayout,
    class TensorA, class TensorB
  >
  CUTLASS_DEVICE auto
  mma(
      Load2MmaPipeline load2mma_pipeline,
      Load2MmaPipelineState load2mma_pipeline_consumer_state,
      Transform2MmaPipeline transform2mma_pipeline,
      Transform2MmaPipelineState transform2mma_pipeline_consumer_state,
      Mma2AccumPipeline mma2accum_pipeline,
      Mma2AccumPipelineState mma2accum_pipeline_producer_state,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      cute::tuple<TensorA, TensorB> const& input_operands,
      int k_tile_count
  ) {
    TiledMma tiled_mma;

    auto curr_load2mma_pipeline_consumer_state = load2mma_pipeline_consumer_state;
    auto next_load2mma_pipeline_consumer_state = load2mma_pipeline_consumer_state;

    auto curr_transform2mma_pipeline_consumer_state = transform2mma_pipeline_consumer_state;
    auto next_transform2mma_pipeline_consumer_state = transform2mma_pipeline_consumer_state;

    uint32_t skip_wait = (k_tile_count <= 0);
    auto transform2mma_flag = transform2mma_pipeline.consumer_try_wait(next_transform2mma_pipeline_consumer_state, skip_wait);
    auto load2mma_flag = load2mma_pipeline.consumer_try_wait(next_load2mma_pipeline_consumer_state, skip_wait);
    ++next_transform2mma_pipeline_consumer_state;
    ++next_load2mma_pipeline_consumer_state;


    // tCrA : (MMA), MMA_M, MMA_K, SmemStage  (In SMEM or TMEM)
    //      We use SMEM stages to match #buffers in Load <-> Convert
    // tCrB : (MMA), MMA_N, MMA_K, SmemStages (In SMEM)
    auto const [tCrA, tCrB] = input_operands;

    mma2accum_pipeline.producer_acquire(mma2accum_pipeline_producer_state);

    int mma2accum_pipeline_producer_state_index = mma2accum_pipeline_producer_state.index();
    auto tCtC = accumulators(_,_,_,mma2accum_pipeline_producer_state_index);
    auto curr_mma2accum_pipeline_producer_state = mma2accum_pipeline_producer_state;
    ++mma2accum_pipeline_producer_state;

    //
    // PIPELINED MAIN LOOP
    //
    // Clear the accumulator
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {

      load2mma_pipeline.consumer_wait(curr_load2mma_pipeline_consumer_state, load2mma_flag);
      transform2mma_pipeline.consumer_wait(curr_transform2mma_pipeline_consumer_state, transform2mma_flag);

      int load2mma_pipeline_consumer_state_index = curr_load2mma_pipeline_consumer_state.index(); //read_stage
      int transform2mma_pipeline_consumer_state_index = curr_transform2mma_pipeline_consumer_state.index(); //read_stage

      auto tCrA0 = tCrA(_,_,_,transform2mma_pipeline_consumer_state_index);
      auto tCrB0 = tCrB(_,_,_,load2mma_pipeline_consumer_state_index);

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); k_block ++) {
        cute::gemm(tiled_mma, tCrA0(_,_,k_block), tCrB0(_,_,k_block), tCtC);               // A[0]*B[0]
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }

      load2mma_pipeline.consumer_release(curr_load2mma_pipeline_consumer_state);
      transform2mma_pipeline.consumer_release(curr_transform2mma_pipeline_consumer_state);

      skip_wait = (k_tile_count <= 1);
      load2mma_flag = load2mma_pipeline.consumer_try_wait(next_load2mma_pipeline_consumer_state, skip_wait);
      transform2mma_flag = transform2mma_pipeline.consumer_try_wait(next_transform2mma_pipeline_consumer_state, skip_wait);

      curr_load2mma_pipeline_consumer_state = next_load2mma_pipeline_consumer_state;
      curr_transform2mma_pipeline_consumer_state = next_transform2mma_pipeline_consumer_state;

      ++next_load2mma_pipeline_consumer_state;
      ++next_transform2mma_pipeline_consumer_state;
    }

    mma2accum_pipeline.producer_commit(curr_mma2accum_pipeline_producer_state);

    return cute::make_tuple(curr_load2mma_pipeline_consumer_state, curr_transform2mma_pipeline_consumer_state, mma2accum_pipeline_producer_state);
  }

  template<class FrgEngine, class FrgLayout>
  CUTLASS_DEVICE auto
  mma_init(cute::Tensor<FrgEngine, FrgLayout> const& accumulators, TensorStorage& shared_storage) const {
    TiledMma tiled_mma;

    auto get_tCrA = [&] () constexpr {
      if constexpr (cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value) {
        Tensor sACompute = make_tensor(make_smem_ptr(shared_storage.compute.smem_ACompute.begin()), SmemLayoutACompute{});
        return tiled_mma.make_fragment_A(sACompute);
      }
      else {
        auto tCrA = tiled_mma.make_fragment_A(shape(SmemLayoutACompute{}));
        tCrA.data() = accumulators.data().get() + cutlass::detail::find_tmem_tensor_col_offset(accumulators);
        return tCrA;
      }
    };

    Tensor tCrA = get_tCrA();
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.input.smem_B.begin()), SmemLayoutB{});
    Tensor tCrB = tiled_mma.make_fragment_B(sB);
    return cute::make_tuple(tCrA, tCrB);
  }

  template<class FrgEngine, class FrgLayout, class TmemCopyAtom, class EpilogueTile>
  CUTLASS_DEVICE auto
  accum_init(cute::Tensor<FrgEngine, FrgLayout> const& accumulators, TmemCopyAtom tmem_cp_atom, EpilogueTile epilogue_tile) {
    return accumulators;
  }

private:
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE
  constexpr auto
  tile_input_tensors(Params const& params, ProblemShape_MNKL const& problem_shape_MNKL) const {
    using X = cute::Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = observed_tma_load_a_->get_tma_tensor(make_shape(M,K,L));
    Tensor mB_nkl = observed_tma_load_b_->get_tma_tensor(make_shape(N,K,L));

    // Tile the tensors and defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});

    return cute::make_tuple(gA_mkl, gB_nkl);
  }

  typename Params::TMA_A const* observed_tma_load_a_ = nullptr;
  typename Params::TMA_B const* observed_tma_load_b_ = nullptr;

  ClusterShape cluster_shape_;
  uint32_t block_rank_in_cluster_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
