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
/* \file
   \brief Defines operations for all GEMM operation kinds in CUTLASS Library.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/detail/collective.hpp"
#include "cutlass/array.h"
#include "cutlass/array_subbyte.h"
#include "cutlass/library/library.h"
#include "library_internal.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/mixed_dtype_utils.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cute/tensor.hpp"
#include <unordered_map>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::library {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class GemmOperation3xBase : public Operation {
public:
  using Operator = Operator_;
  using OperatorArguments = typename Operator::Arguments;
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementD = typename Operator::ElementD;
  using LayoutD = typename Operator::LayoutD;
  // assuming all tensors use same type for StrideIndex
  using StrideIndex = typename Operator::LayoutA::Index;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

protected:
  GemmDescription description_;

public:

  /// Constructor
  GemmOperation3xBase(char const *name = "unknown_gemm", GemmKind gemm_kind_ = GemmKind::kGemm) {

    description_.name = name;
    description_.provider = Provider::kCUTLASS;
    description_.kind = OperationKind::kGemm;
    description_.gemm_kind = gemm_kind_;

    description_.tile_description.threadblock_shape = make_Coord(
      Operator::ThreadblockShape::kM,
      Operator::ThreadblockShape::kN,
      Operator::ThreadblockShape::kK);

    if constexpr (Operator::ArchTag::kMinComputeCapability >= 90) {
      description_.tile_description.cluster_shape = make_Coord(
        Operator::ClusterShape::kM,
        Operator::ClusterShape::kN,
        Operator::ClusterShape::kK);
    }

    description_.tile_description.threadblock_stages = Operator::kStages;

    description_.tile_description.warp_count = make_Coord(
      Operator::WarpCount::kM,
      Operator::WarpCount::kN,
      Operator::WarpCount::kK);

    description_.tile_description.math_instruction.instruction_shape = make_Coord(
      Operator::InstructionShape::kM,
      Operator::InstructionShape::kN,
      Operator::InstructionShape::kK);

    description_.tile_description.math_instruction.element_accumulator =
      NumericTypeMap<ElementAccumulator>::kId;

    description_.tile_description.math_instruction.opcode_class =
      OpcodeClassMap<typename Operator::OperatorClass>::kId;

    description_.tile_description.math_instruction.math_operation =
      MathOperationMap<typename Operator::MathOperator>::kId;

    description_.tile_description.minimum_compute_capability =
      ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMin;

    description_.tile_description.maximum_compute_capability =
      ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMax;

    description_.A = make_TensorDescription<ElementA, LayoutA>(Operator::kAlignmentA);
    description_.B = make_TensorDescription<ElementB, LayoutB>(Operator::kAlignmentB);
    description_.C = make_TensorDescription<ElementC, LayoutC>(Operator::kAlignmentC);
    description_.D = make_TensorDescription<ElementD, LayoutD>(Operator::kAlignmentD);
    description_.element_epilogue = NumericTypeMap<ElementCompute>::kId;

    description_.split_k_mode = SplitKMode::kNone;
    description_.transform_A = ComplexTransformMap<Operator::kTransformA>::kId;
    description_.transform_B = ComplexTransformMap<Operator::kTransformB>::kId;
  }

  /// Returns the description of the GEMM operation
  virtual OperationDescription const & description() const {
    return description_;
  }

  /// Returns the description of the GEMM operation
  GemmDescription const& get_gemm_description() const {
    return description_;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class GemmUniversal3xOperation : public GemmOperation3xBase<Operator_> {
public:

  using Operator = Operator_;
  using OperatorArguments = typename Operator::Arguments;
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementD = typename Operator::ElementD;
  using LayoutD = typename Operator::LayoutD;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  using CollectiveMainloop = typename Operator::CollectiveMainloop;
  using CollectiveEpilogue = typename Operator::CollectiveEpilogue;
  using ThreadEpilogueOp = typename CollectiveEpilogue::ThreadEpilogueOp;

  static constexpr bool IsRuntimeDataTypeA = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementA>();

  static constexpr bool IsRuntimeDataTypeB = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementB>();

  static_assert((IsRuntimeDataTypeA && IsRuntimeDataTypeB) ||
                (!IsRuntimeDataTypeA && !IsRuntimeDataTypeB),
                "ElementA and ElementB in a GEMM kernel should be both runtime or both static.");

  static constexpr bool IsRuntimeDataType = IsRuntimeDataTypeA && IsRuntimeDataTypeB;
  

public:

  /// Constructor
  GemmUniversal3xOperation(char const *name = "unknown_gemm"):
    GemmOperation3xBase<Operator_>(name, GemmKind::kUniversal) {
    if constexpr (Operator::ArchTag::kMinComputeCapability == 90) {
      dim3 cluster_dims(
        cute::size<0>(typename Operator::GemmKernel::ClusterShape{}),
        cute::size<1>(typename Operator::GemmKernel::ClusterShape{}),
        cute::size<2>(typename Operator::GemmKernel::ClusterShape{}));
      uint32_t threads_per_block = Operator::GemmKernel::MaxThreadsPerBlock;
      void const* kernel_ptr = (void*)(device_kernel<typename Operator::GemmKernel>);
      max_active_clusters = cutlass::KernelHardwareInfo::query_device_max_active_clusters(
        cluster_dims,
        threads_per_block,
        kernel_ptr);
    }
  }

private:
  int max_active_clusters{};

protected:

  /// Constructs the arguments structure given the configuration and arguments
  static Status construct_arguments_(
      OperatorArguments &operator_args, GemmUniversalConfiguration const *configuration) {
    // NOTE: GemmUniversalConfiguration does not contain problem shapes or batch strides
    // Do nothing here and construct kernel arguments in update_arguments_ instead
    // We also cannot construct TMA descriptors without all the arguments available

    operator_args.mode = configuration->mode;
    return Status::kSuccess;
  }

  template<class FusionArgs, class = void>
  struct UpdateFusionArgs {
    static Status update_(FusionArgs const& fusion_args, GemmUniversalArguments const &arguments) {
      // If a custom EVT is instantiated then it is the users's responsibility
      // to ensure alpha and beta are updated appropriately
      return Status::kSuccess;
    }
  };

  template<class FusionArgs>
  struct UpdateFusionArgs<FusionArgs, cute::void_t<decltype(FusionArgs{}.alpha)>> {
    static Status update_(FusionArgs& fusion_args, GemmUniversalArguments const &arguments) {
      if (arguments.pointer_mode == ScalarPointerMode::kHost) {
        fusion_args.alpha = *static_cast<ElementCompute const *>(arguments.alpha);
        fusion_args.beta = *static_cast<ElementCompute const *>(arguments.beta);
        fusion_args.alpha_ptr = nullptr;
        fusion_args.beta_ptr = nullptr;

        return Status::kSuccess;
      }
      else if (arguments.pointer_mode == ScalarPointerMode::kDevice) {
        fusion_args.alpha = 0;
        fusion_args.beta = 0;
        fusion_args.alpha_ptr = static_cast<ElementCompute const *>(arguments.alpha);
        fusion_args.beta_ptr = static_cast<ElementCompute const *>(arguments.beta);

        return Status::kSuccess;
      }
      else {
        return Status::kErrorInvalidProblem;
      }
    }
  };

  template<template<int, class, class> class Policy, int Stages, class ClusterShape, class KernelSchedule>
  static constexpr bool is_sm90_mixed_dtype_mainloop_(Policy<Stages, ClusterShape, KernelSchedule> policy) {
    return (cute::is_same_v<Policy<Stages, ClusterShape, KernelSchedule>,
                            cutlass::gemm::MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput<Stages, ClusterShape, KernelSchedule>>);
  }

  template <class DispatchPolicy>
  static constexpr bool is_sm90_mixed_dtype_mainloop_(DispatchPolicy) {
    return false;
  }

  template <
    typename ElementWide,
    typename ElementNarrow,
    typename ElementScaleMainloop,
    class ActualStrideAB,
    Sm90MixedInputWiderOperand wider_operand,
    bool is_n4w8,
    typename ElementScale,
    typename ElementZero,
    class Layout_SZ>
  static void dequantize_encode_(
      OperatorArguments &operator_args,
      GemmUniversalArguments const *arguments,
      cudaStream_t stream,
      const int &problem_mn,
      const int &problem_k,
      const int &options_l,
      const int &options_g,
      ElementScale *ptr_S,
      ElementZero *ptr_Z,
      const size_t &SZ_size,
      Layout_SZ layout_SZ
      ) {

    auto shape_AB  = cute::make_shape(problem_mn, problem_k, options_l);
    auto stride_AB = cutlass::make_cute_packed_stride(ActualStrideAB{}, shape_AB);
    auto layout_AB = cute::make_layout(shape_AB, stride_AB);
    auto *ptr_dequantized_AB = static_cast<ElementWide *>(arguments->dequantized_AB);
    const ElementNarrow *ptr_AB = nullptr;
    if constexpr(wider_operand == Sm90MixedInputWiderOperand::A) {
      ptr_AB = static_cast<const ElementNarrow *>(arguments->B);
    }
    else {
      ptr_AB = static_cast<const ElementNarrow *>(arguments->A);
    }
    dequantize(ptr_dequantized_AB, ptr_AB, layout_AB, ptr_S, ptr_Z, layout_SZ, options_g, stream);
    if constexpr(is_n4w8) {
      size_t AB_size = cute::size(layout_AB);
      cutlass::int4b_t *encoded_AB = static_cast<cutlass::int4b_t *>(arguments->encoded_AB);
      unified_encode_int4b(ptr_AB, encoded_AB, AB_size);
      if constexpr(wider_operand == Sm90MixedInputWiderOperand::A) {
        operator_args.mainloop.ptr_B = static_cast<ElementNarrow const *>(encoded_AB);
      }
      else {
        operator_args.mainloop.ptr_A = static_cast<ElementNarrow const *>(encoded_AB);
      }
      ElementScaleMainloop *ptr_packed_Scale = static_cast<ElementScaleMainloop *>(arguments->packed_Scale);
      pack_scale_fp8(ptr_S, ptr_packed_Scale, SZ_size);
    }
  }

  template <
    typename ElementAB,
    class ActualStrideAB,
    class LayoutAB_Reordered,
    class LayoutAtomQuant,
    Sm90MixedInputWiderOperand wider_operand>
  static void handle_shuffle_tensor_(
      OperatorArguments &operator_args,
      GemmUniversalArguments const *arguments,
      const int &problem_mn,
      const int &problem_k,
      const int &options_l) {

    auto shape_AB  = cute::make_shape(problem_mn, problem_k, options_l);
    auto stride_AB = cutlass::make_cute_packed_stride(ActualStrideAB{}, shape_AB);
    auto layout_AB = cute::make_layout(shape_AB, stride_AB);
    LayoutAB_Reordered layout_AB_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_AB);
    if constexpr(wider_operand == Sm90MixedInputWiderOperand::A) {
      operator_args.mainloop.dB = layout_AB_reordered;
    }
    else {
      operator_args.mainloop.dA = layout_AB_reordered;
    }
    if (arguments->generate_dequantized_AB) {
      size_t AB_size = cute::size(layout_AB);
      ElementAB *AB_reordered = cutlass::device_memory::allocate<ElementAB>(AB_size);
      const ElementAB *AB_src = nullptr;
      if constexpr(wider_operand == Sm90MixedInputWiderOperand::A) {
        AB_src = static_cast<const ElementAB *>(operator_args.mainloop.ptr_B);
      }
      else {
        AB_src = static_cast<const ElementAB *>(operator_args.mainloop.ptr_A);
      }
      reorder_tensor(AB_src, layout_AB, AB_reordered, layout_AB_reordered);
      ElementAB *AB_dst = static_cast<ElementAB *>(arguments->encoded_AB);
      cutlass::device_memory::copy_device_to_device(AB_dst, AB_reordered, AB_size);
      cutlass::device_memory::free(AB_reordered);
      if constexpr(wider_operand == Sm90MixedInputWiderOperand::A) {
        operator_args.mainloop.ptr_B = AB_dst;
      }
      else {
        operator_args.mainloop.ptr_A = AB_dst;
      }
    }
  }

  /// Constructs the arguments structure given the configuration and arguments
  Status update_arguments_(
    OperatorArguments& operator_args,
    GemmUniversalArguments const* arguments,
    cudaStream_t stream = nullptr) const {
    Status status = Status::kSuccess;

    status = UpdateFusionArgs<decltype(operator_args.epilogue.thread)>::update_(
      operator_args.epilogue.thread, *arguments);
    if (status != Status::kSuccess) {
      return status;
    }

    // TODO: type erase Arguments structure in 3.0 GEMM
    operator_args.problem_shape = cute::make_shape(
      arguments->problem_size.m(),
      arguments->problem_size.n(),
      arguments->problem_size.k(),
      arguments->batch_count);

    // update arguments

    if constexpr (IsRuntimeDataType) {
      using ArrayElementA = typename Operator::GemmKernel::CollectiveMainloop::ArrayElementA;
      using ArrayElementB = typename Operator::GemmKernel::CollectiveMainloop::ArrayElementB;
      operator_args.mainloop.ptr_A = static_cast<ArrayElementA const *>(arguments->A);
      operator_args.mainloop.ptr_B = static_cast<ArrayElementB const *>(arguments->B);

      std::unordered_map<RuntimeDatatype, cute::UMMA::MXF8F6F4Format> mapping = {
          {RuntimeDatatype::kE4M3, cute::UMMA::MXF8F6F4Format::E4M3},
          {RuntimeDatatype::kE5M2, cute::UMMA::MXF8F6F4Format::E5M2},
          {RuntimeDatatype::kE3M2, cute::UMMA::MXF8F6F4Format::E3M2},
          {RuntimeDatatype::kE2M1, cute::UMMA::MXF8F6F4Format::E2M1}
      };

      auto iter_runtime_a = mapping.find(arguments->runtime_input_datatype_a);
      auto iter_runtime_b = mapping.find(arguments->runtime_input_datatype_b);

      if (iter_runtime_a != mapping.end()) {
          operator_args.mainloop.runtime_data_type_a = iter_runtime_a->second;
      } else {
        assert("invalid runtime argument for datatype A!");
      }

      if (iter_runtime_b != mapping.end()) {
          operator_args.mainloop.runtime_data_type_b = iter_runtime_b->second;
      } else {
        assert("invalid runtime argument for datatype B!");
      }

    }
    else {
      operator_args.mainloop.ptr_A = static_cast<ElementA const *>(arguments->A);
      operator_args.mainloop.ptr_B = static_cast<ElementB const *>(arguments->B);
    }
    operator_args.epilogue.ptr_C = static_cast<ElementC const *>(arguments->C);
    operator_args.epilogue.ptr_D = static_cast<ElementD       *>(arguments->D);

    // Stride{A,B} is a Layout if and only if:
    // (1) This is a mixed dtype kernel, and
    // (2) This mixed dtype kernel is using shuffling, and
    // (3) sizeof(narrow_type) == 4 or 8 bits, and
    // (4) sizeof(wide_type) == 16 bits.
    // If A/B has the narrow data type, Stride{A/B} will be a Layout
    constexpr bool is_StrideA_Layout = cute::is_layout<typename CollectiveMainloop::StrideA>::value;
    constexpr bool is_StrideB_Layout = cute::is_layout<typename CollectiveMainloop::StrideB>::value;
    static_assert(!(is_StrideA_Layout && is_StrideB_Layout), "Incorrect kernel configuration: StrideA and StrideB are both cute::Layout");
    if constexpr(!is_StrideA_Layout) {
      operator_args.mainloop.dA = cute::make_int_tuple_from<typename Operator::GemmKernel::StrideA>(
        arguments->lda, arguments->batch_stride_A);
    }
    if constexpr(!is_StrideB_Layout) {
      operator_args.mainloop.dB = cute::make_int_tuple_from<typename Operator::GemmKernel::StrideB>(
        arguments->ldb, arguments->batch_stride_B);
    }
    operator_args.epilogue.dC = cute::make_int_tuple_from<typename Operator::GemmKernel::StrideC>(
        arguments->ldc, arguments->batch_stride_C);
    operator_args.epilogue.dD = operator_args.epilogue.dC;

    using MainloopPolicy = typename CollectiveMainloop::DispatchPolicy;
    if constexpr(is_sm90_mixed_dtype_mainloop_(MainloopPolicy{})) {
      const int problem_m = arguments->problem_size.m();
      const int problem_n = arguments->problem_size.n();
      const int problem_k = arguments->problem_size.k();
      const int options_l = arguments->batch_count;

      constexpr Sm90MixedInputWiderOperand wider_operand =
        (cutlass::sizeof_bits<ElementA>::value > cutlass::sizeof_bits<ElementB>::value) ?
        Sm90MixedInputWiderOperand::A : Sm90MixedInputWiderOperand::B;
      using ElementWide = std::conditional_t<wider_operand == Sm90MixedInputWiderOperand::A, ElementA, ElementB>;
      using ElementNarrow = std::conditional_t<wider_operand == Sm90MixedInputWiderOperand::A, ElementB, ElementA>;

      constexpr bool has_scale = !std::is_same_v<typename CollectiveMainloop::ElementScale, void>;
      constexpr bool has_zero  = !std::is_same_v<typename CollectiveMainloop::ElementZero,  void>;

      const int options_g = problem_k;
      const int scale_k = (problem_k + options_g - 1) / options_g;

      constexpr bool is_A4B8 = (
        cutlass::is_same_v<ElementA, cutlass::int4b_t> &&
        (cutlass::is_same_v<ElementB, cutlass::float_e4m3_t> ||
         cutlass::is_same_v<ElementB, cutlass::float_e5m2_t>));
      constexpr bool is_A8B4 = (
        cutlass::is_same_v<ElementB, cutlass::int4b_t> &&
        (cutlass::is_same_v<ElementA, cutlass::float_e4m3_t> ||
         cutlass::is_same_v<ElementA, cutlass::float_e5m2_t>));
      constexpr bool is_int4_x_fp8 = is_A4B8 || is_A8B4;

      // If this is a convert-only kernel, we still need to generate dequantized A or B for verification,
      // and in this case ElementScale is the same as ElementWide
      // In int4 * fp8, ElementScale is a cutlass::Array, need to take out it's real element
      using DummyElementScaleMainloop = std::conditional_t<
        is_int4_x_fp8,
        typename cutlass::Array<ElementWide, 8>,
        ElementWide
      >;
      using ElementScaleMainloop = std::conditional_t<
        has_scale,
        typename CollectiveMainloop::ElementScale,
        DummyElementScaleMainloop
      >;
      using ElementScale = std::conditional_t<
        has_scale,
        typename UnderlyingElement<typename CollectiveMainloop::ElementScale>::type,
        ElementWide
      >;
      using StrideScale = typename CollectiveMainloop::StrideScale;
      // In ScaleOnly mode, we have allocated the same size of memory for arguments->Z and arguments->S
      using ElementZero = std::conditional_t<
        has_zero,
        typename CollectiveMainloop::ElementZero,
        ElementScale
      >;
      const int SZ_1st_dim = (wider_operand == Sm90MixedInputWiderOperand::A) ? problem_n : problem_m;
      const size_t SZ_size = static_cast<size_t>(SZ_1st_dim * scale_k * options_l);
      auto shape_SZ = cute::make_shape(SZ_1st_dim, scale_k, options_l);
      ElementScale *ptr_S = static_cast<ElementScale *>(arguments->Scale);
      ElementZero  *ptr_Z = static_cast<ElementZero  *>(arguments->Zero);

      // 1. If arguments is initialized in profiler, S and Z needs to be allocated and filled
      if (arguments->generate_scale_and_zero) {
        float scale_min = 1.0f, scale_max = 1.0f;
        if constexpr(has_scale) {
          const float elt_max_f = float(cutlass::platform::numeric_limits<ElementScale>::max());
          // Need to fix max_dequant_val and min_dequant_val?
          const float max_dequant_val = elt_max_f * 0.25f;
          const float min_dequant_val = 0.5f;
          scale_max = max_dequant_val / elt_max_f;
          scale_min = min_dequant_val / elt_max_f;
        }
        uint64_t seed = 2023;
        cutlass::reference::device::BlockFillRandomUniform(
          ptr_S, SZ_size, seed, ElementScale(scale_max), ElementScale(scale_min));

        // In ScaleOnly mode, set Z as zero for generating dequantized A or B
        const float zero_max = has_zero ?  2.0f : 0.0f;
        const float zero_min = has_zero ? -2.0f : 0.0f;
        cutlass::reference::device::BlockFillRandomUniform(
          ptr_Z, SZ_size, seed, ElementZero(zero_max), ElementZero(zero_min));
      }  // End of "if (arguments->generate_scale_and_zero)"

      // 2. Generate the dequantized A or B for verification
      if (arguments->generate_dequantized_AB) {
        StrideScale stride_SZ = cutlass::make_cute_packed_stride(StrideScale{}, shape_SZ);
        auto layout_SZ = cute::make_layout(shape_SZ, stride_SZ);
        if constexpr(wider_operand == Sm90MixedInputWiderOperand::A) {
          if constexpr(is_StrideB_Layout) {
            // The generator only generates row-major A and col-major B at the moment
            // Need a way to read out the actual layout of B later
            using ActualLayoutB = cutlass::layout::ColumnMajor;
            using ActualStrideB = cutlass::detail::TagToStrideB_t<ActualLayoutB>;
            dequantize_encode_<ElementWide, ElementNarrow, ElementScaleMainloop, ActualStrideB, wider_operand, is_A8B4>(
              operator_args, arguments, stream, problem_m, problem_k, options_l, options_g, ptr_S, ptr_Z, SZ_size, layout_SZ);
          }
          else {
            using ActualStrideB = typename CollectiveMainloop::StrideB;
            dequantize_encode_<ElementWide, ElementNarrow, ElementScaleMainloop, ActualStrideB, wider_operand, is_A8B4>(
              operator_args, arguments, stream, problem_m, problem_k, options_l, options_g, ptr_S, ptr_Z, SZ_size, layout_SZ);
          }
        }
        else {
          if constexpr(is_StrideA_Layout) {
            // The generator only generates row-major A and col-major B at the moment
            // Need a way to read out the actual layout of A later
            using ActualLayoutA = cutlass::layout::RowMajor;
            using ActualStrideA = cutlass::detail::TagToStrideA_t<ActualLayoutA>;
            dequantize_encode_<ElementWide, ElementNarrow, ElementScaleMainloop, ActualStrideA, wider_operand, is_A4B8>(
              operator_args, arguments, stream, problem_m, problem_k, options_l, options_g, ptr_S, ptr_Z, SZ_size, layout_SZ);
          }
          else {
            using ActualStrideA = typename CollectiveMainloop::StrideA;
            dequantize_encode_<ElementWide, ElementNarrow, ElementScaleMainloop, ActualStrideA, wider_operand, is_A4B8>(
              operator_args, arguments, stream, problem_m, problem_k, options_l, options_g, ptr_S, ptr_Z, SZ_size, layout_SZ);
          }
        }  // End of "if constexpr(wider_operand == Sm90MixedInputWiderOperand::A)"
      }  // End of "if (arguments->generate_dequantized_AB)"

      // 3. Put Scale and Zero in mainloop
      if constexpr(has_scale) {
        if constexpr(is_int4_x_fp8) {
          operator_args.mainloop.ptr_S = static_cast<ElementScaleMainloop const*>(arguments->packed_Scale);
        }
        else {
          operator_args.mainloop.ptr_S = static_cast<ElementScale const*>(arguments->Scale);
        }
        operator_args.mainloop.dS = cutlass::make_cute_packed_stride(StrideScale{}, shape_SZ);
        operator_args.mainloop.group_size = options_g;
        if constexpr(has_zero) {
          operator_args.mainloop.ptr_Z = static_cast<ElementZero const*>(arguments->Zero);
        }
      }  // End of "if constexpr(has_scale)"

      // Handle the shuffling
      using ValueShuffle = std::conditional_t<
        cutlass::sizeof_bits<ElementNarrow>::value == 4,
        cute::Layout<cute::Shape<cute::_2,cute::_4>, cute::Stride<cute::_4,cute::_1>>,
        cute::Layout<cute::Shape<cute::_2,cute::_2>, cute::Stride<cute::_2,cute::_1>>
      >;
      constexpr int NumShuffleAtoms = 1;
      using MmaAtomShape = cute::Layout<cute::Shape<cute::_1,cute::Int<NumShuffleAtoms>>>;
      using LayoutAtomQuant = decltype(compute_memory_reordering_atom<ElementWide, MmaAtomShape, ValueShuffle>());
      // The generator only generates row-major A and col-major B at the moment
      // Need a way to read out the actual layout and stride of A/B later
      if constexpr(wider_operand == Sm90MixedInputWiderOperand::A && is_StrideB_Layout) {
        using ActualLayoutB = cutlass::layout::ColumnMajor;
        using ActualStrideB = cutlass::detail::TagToStrideB_t<ActualLayoutB>;
        using LayoutB_Reordered = typename CollectiveMainloop::StrideB;
        handle_shuffle_tensor_<ElementB, ActualStrideB, LayoutB_Reordered, LayoutAtomQuant, wider_operand>(
          operator_args, arguments, problem_n, problem_k, options_l);
      }
      if constexpr(wider_operand == Sm90MixedInputWiderOperand::B && is_StrideA_Layout) {
        using ActualLayoutA = cutlass::layout::RowMajor;
        using ActualStrideA = cutlass::detail::TagToStrideA_t<ActualLayoutA>;
        using LayoutA_Reordered = typename CollectiveMainloop::StrideA;
        handle_shuffle_tensor_<ElementA, ActualStrideA, LayoutA_Reordered, LayoutAtomQuant, wider_operand>(
          operator_args, arguments, problem_m, problem_k, options_l);
      }
    } // End of "if constexpr(is_sm90_mixed_dtype_mainloop_(MainloopPolicy{}))"

    /* Query device SM count and max active clusters to pass onto the kernel as an argument, where needed */
    operator_args.hw_info.sm_count = arguments->sm_count;
    if constexpr (Operator::ArchTag::kMinComputeCapability == 90) {
      operator_args.hw_info.max_active_clusters = max_active_clusters;
    }
    if constexpr (!std::is_const_v<decltype(operator_args.scheduler.max_swizzle_size)>) {
      operator_args.scheduler.max_swizzle_size = arguments->swizzle_size;
    }

    if constexpr (!std::is_const_v<decltype(operator_args.scheduler.raster_order)>) {
      using Enum_t = decltype(operator_args.scheduler.raster_order);
      switch (arguments->raster_order) {
        case RasterOrder::kAlongN:
          operator_args.scheduler.raster_order = Enum_t::AlongN;
          break;
        case RasterOrder::kAlongM:
          operator_args.scheduler.raster_order = Enum_t::AlongM;
          break;
        default:
          operator_args.scheduler.raster_order = Enum_t::Heuristic;
      }
    }

    if constexpr (std::is_same_v<typename Operator::GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>) {
      operator_args.scheduler.splits = arguments->split_k_slices;
    }

    if constexpr (Operator::ArchTag::kMinComputeCapability >= 100) {
      operator_args.hw_info.cluster_shape = dim3(
        arguments->cluster_shape.m(),
        arguments->cluster_shape.n(),
        arguments->cluster_shape.k());
      operator_args.hw_info.cluster_shape_fallback = dim3(
        arguments->cluster_shape_fallback.m(),
        arguments->cluster_shape_fallback.n(),
        arguments->cluster_shape_fallback.k());
    }
    return status;
  }

public:

  /// Returns success if the operation can proceed
  Status can_implement(
      [[maybe_unused]] void const *configuration_ptr, void const *arguments_ptr) const override {
    GemmUniversalArguments const *arguments =
      static_cast<GemmUniversalArguments const *>(arguments_ptr);
    OperatorArguments args;

    auto status = update_arguments_(args, arguments);
    if (status != Status::kSuccess) {
      return status;
    }

    Status can_impl = Operator::can_implement(args);

    //return Operator::can_implement(args);
    return can_impl;
  }

  /// Gets the host-side workspace
  uint64_t get_host_workspace_size(void const *configuration) const override {
    return sizeof(Operator);
  }

  /// Gets the device-side workspace
  uint64_t get_device_workspace_size(
      void const *configuration_ptr,void const *arguments_ptr) const override {

    OperatorArguments args;
    auto status = update_arguments_(
      args, static_cast<GemmUniversalArguments const *>(arguments_ptr));
    if (status != Status::kSuccess) {
      return 0;
    }

    uint64_t size = Operator::get_workspace_size(args);
    return size;
  }

  /// Initializes the workspace
  Status initialize(
      void const *configuration_ptr,
      void *host_workspace,
      void *device_workspace,
      cudaStream_t stream = nullptr) const override {
    Operator *op = new (host_workspace) Operator;
    return Status::kSuccess;
  }

  /// Runs the kernel
  Status run(
      void const *arguments_ptr,
      void *host_workspace,
      void *device_workspace = nullptr,
      cudaStream_t stream = nullptr) const override {

    OperatorArguments args;
    Status status = update_arguments_(args, static_cast<GemmUniversalArguments const *>(arguments_ptr), stream);
    if (status != Status::kSuccess) {
      return status;
    }

    Operator *op = static_cast<Operator *>(host_workspace);
    // We need to call initialize() since we have to rebuild TMA desc for every new set of args
    status = op->run(args, device_workspace, stream, nullptr, 
                     static_cast<GemmUniversalArguments const *>(arguments_ptr)->use_pdl);
    return status;
  }
};
///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::library

///////////////////////////////////////////////////////////////////////////////////////////////////
