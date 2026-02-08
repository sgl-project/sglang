/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief Defines reference operations for GEMM operation kinds in CUTLASS Library
*/

#pragma once

#include <iostream>
#include <sstream>
#include <cstring>

#include "cutlass/cutlass.h"

#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"
#include "cutlass/library/util.h"
#include "library_internal.h"

#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  Provider Provider_,
  typename ElementA_,
  typename LayoutA_,
  cutlass::ComplexTransform TransformA,
  typename ElementB_,
  typename LayoutB_,
  cutlass::ComplexTransform TransformB,
  typename ElementC_,
  typename LayoutC_,
  typename ElementCompute_,
  typename ElementAccumulator_ = ElementCompute_,
  typename ElementD_ = ElementC_,
  typename ConvertOp_ = NumericConverter<ElementD_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
class GemmReferenceOperation : public Operation {
public:
  static Provider const kProvider = Provider_;

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA, LayoutA>;
  static cutlass::ComplexTransform const kTransformA = TransformA;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB, LayoutB>;
  static cutlass::ComplexTransform const kTransformB = TransformB;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using ElementD = ElementD_;
  using TensorRefC = TensorRef<ElementC, LayoutC>;
  using TensorRefD = TensorRef<ElementD, LayoutC>;
  using ElementCompute = ElementCompute_;
  using ElementAccumulator = ElementAccumulator_;
  using ConvertOp = ConvertOp_;
  using InnerProductOp = InnerProductOp_;

protected:

  /// Storage for the name string
  std::string name_;

  ///
  GemmDescription description_;

public:

  /// Constructor
  GemmReferenceOperation() {
    
    // Basic information
    description_.provider = kProvider;
    description_.kind = OperationKind::kGemm;
    description_.gemm_kind = GemmKind::kUniversal;

    // Tensor description
    description_.A = make_TensorDescription<ElementA, LayoutA>();
    description_.transform_A = ComplexTransformMap<kTransformA>::kId;
    description_.B = make_TensorDescription<ElementB, LayoutB>();
    description_.transform_B = ComplexTransformMap<kTransformB>::kId;
    description_.C = make_TensorDescription<ElementC, LayoutC>();
    description_.D = make_TensorDescription<ElementD, LayoutC>();
    
    // Epilogue compute and accumulator type description
    description_.element_epilogue = NumericTypeMap<ElementCompute>::kId;

    description_.tile_description.math_instruction.element_accumulator =
      NumericTypeMap<ElementAccumulator>::kId;

    // Compute capability for gemm reference
    description_.tile_description.minimum_compute_capability = 
      (kProvider == Provider::kReferenceDevice ? 50 : 0);

    description_.tile_description.maximum_compute_capability = 1024;

    // Procedural name
    std::stringstream ss;

    ss << "gemm"  
      << "_reference_" << to_string(description_.provider)
      << "_" << to_string(description_.A.element) << to_string(description_.A.layout)
      << "_" << to_string(description_.B.element) << to_string(description_.B.layout)
      << "_" << to_string(description_.C.element) << to_string(description_.C.layout)
      << "_" << to_string(description_.tile_description.math_instruction.element_accumulator);

    name_ = ss.str();

    description_.name = name_.c_str();

    // Epilogue compute and accumulator type description
    description_.element_epilogue = NumericTypeMap<ElementCompute>::kId;

    description_.tile_description.math_instruction.element_accumulator =
      NumericTypeMap<ElementAccumulator>::kId;
  }

  /// Returns the description of the GEMM operation
  virtual OperationDescription const & description() const {
    return description_;
  }

  virtual Status can_implement(
    void const *configuration,
    void const *arguments) const {

    return Status::kSuccess;
  }

  virtual uint64_t get_host_workspace_size(
    void const *configuration) const {

    return sizeof(GemmUniversalConfiguration);
  }

  virtual uint64_t get_device_workspace_size(
    void const *configuration,
    void const *arguments = nullptr) const {

    return 0;
  }

  virtual Status initialize(
    void const *configuration,
    void *host_workspace,
    void *device_workspace = nullptr,
    cudaStream_t stream = nullptr) const {

    std::memcpy(host_workspace, configuration, get_host_workspace_size(configuration));

    return Status::kSuccess;
  }

  virtual Status run(
    void const *arguments,
    void *host_workspace,
    void *device_workspace = nullptr,
    cudaStream_t stream = nullptr) const {

    GemmUniversalConfiguration const &config = *static_cast<GemmUniversalConfiguration const *>(host_workspace);
    GemmUniversalArguments const &args = *static_cast<GemmUniversalArguments const *>(arguments);

    TensorRefA ref_A{static_cast<ElementA *>(const_cast<void *>(args.A)), LayoutA(int(config.lda))};
    TensorRefB ref_B{static_cast<ElementB *>(const_cast<void *>(args.B)), LayoutB(int(config.ldb))};
    TensorRefC ref_C{static_cast<ElementC *>(const_cast<void *>(args.C)), LayoutC(int(config.ldc))};
    TensorRefD ref_D{static_cast<ElementD *>(args.D), LayoutC(int(config.ldd))};

    if (kProvider == Provider::kReferenceHost) {

      cutlass::reference::host::GemmComplex<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementCompute,
        ElementAccumulator,
        ElementD,
        ConvertOp,
        InnerProductOp
      >(
        config.problem_size,
        *static_cast<ElementCompute const *>(args.alpha),
        ref_A,
        kTransformA,
        ref_B,
        kTransformB,
        *static_cast<ElementCompute const *>(args.beta),
        ref_C,
        ref_D,
        ElementAccumulator(),
        ((config.mode == library::GemmUniversalMode::kBatched) ? config.batch_count : 1),
        args.batch_stride_A,
        args.batch_stride_B,
        args.batch_stride_C,
        args.batch_stride_D
      );

      return Status::kSuccess;
    }
    else if (kProvider == Provider::kReferenceDevice) {

      cutlass::reference::device::GemmComplex<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementCompute,
        ElementAccumulator,
        ElementD,
        ConvertOp,
        InnerProductOp
      >(
        config.problem_size,
        *static_cast<ElementCompute const *>(args.alpha),
        ref_A,
        kTransformA,
        ref_B,
        kTransformB,
        *static_cast<ElementCompute const *>(args.beta),
        ref_C,
        ref_D,
        ElementAccumulator(),
        ((config.mode == library::GemmUniversalMode::kBatched) ? config.batch_count : 1),
        args.batch_stride_A,
        args.batch_stride_B,
        args.batch_stride_C,
        args.batch_stride_D
      );

      return Status::kSuccess;
    }
    
    return Status::kErrorNotSupported;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA_,
  typename LayoutA_,
  cutlass::ComplexTransform TransformA,
  typename ElementB_,
  typename LayoutB_,
  cutlass::ComplexTransform TransformB,
  typename ElementC_,
  typename LayoutC_,
  typename ElementCompute_,
  typename ElementAccumulator_ = ElementCompute_,
  typename ElementD_ = ElementC_,
  typename ConvertOp_ = NumericConverter<ElementD_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
void make_gemm(Manifest &manifest) {
#if !defined(CUTLASS_PROFILER_DISABLE_REFERENCE)
  manifest.append(new GemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_, LayoutA_, TransformA,
    ElementB_, LayoutB_, TransformB,
    ElementC_, LayoutC_,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >);

  manifest.append(new GemmReferenceOperation<
    Provider::kReferenceDevice,
    ElementA_, LayoutA_, TransformA,
    ElementB_, LayoutB_, TransformB,
    ElementC_, LayoutC_,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >);
#endif
}

/// Helper to create NN, NT, TN, and TT GEMM layouts.
template <
  typename ElementA_, cutlass::ComplexTransform TransformA,
  typename ElementB_, cutlass::ComplexTransform TransformB,
  typename ElementC_,
  typename ElementCompute_,
  typename ElementAccumulator_ = ElementCompute_,
  typename ElementD_ = ElementC_,
  typename ConvertOp_ = NumericConverter<ElementD_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
void make_gemm_canonical_layouts(Manifest &manifest) {

  // M Major outputs
  make_gemm<
    ElementA_, cutlass::layout::ColumnMajor, TransformA,
    ElementB_, cutlass::layout::ColumnMajor, TransformB,
    ElementC_, cutlass::layout::ColumnMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);

  make_gemm<
    ElementA_, cutlass::layout::ColumnMajor, TransformA,
    ElementB_, cutlass::layout::RowMajor, TransformB,
    ElementC_, cutlass::layout::ColumnMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);

  make_gemm<
    ElementA_, cutlass::layout::RowMajor, TransformA,
    ElementB_, cutlass::layout::ColumnMajor, TransformB,
    ElementC_, cutlass::layout::ColumnMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);
  
  make_gemm<
    ElementA_, cutlass::layout::RowMajor, TransformA,
    ElementB_, cutlass::layout::RowMajor, TransformB,
    ElementC_, cutlass::layout::ColumnMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);

  // N Major outputs
  make_gemm<
    ElementA_, cutlass::layout::ColumnMajor, TransformA,
    ElementB_, cutlass::layout::ColumnMajor, TransformB,
    ElementC_, cutlass::layout::RowMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);

  make_gemm<
    ElementA_, cutlass::layout::ColumnMajor, TransformA,
    ElementB_, cutlass::layout::RowMajor, TransformB,
    ElementC_, cutlass::layout::RowMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);

  make_gemm<
    ElementA_, cutlass::layout::RowMajor, TransformA,
    ElementB_, cutlass::layout::ColumnMajor, TransformB,
    ElementC_, cutlass::layout::RowMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);

  make_gemm<
    ElementA_, cutlass::layout::RowMajor, TransformA,
    ElementB_, cutlass::layout::RowMajor, TransformB,
    ElementC_, cutlass::layout::RowMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);
}


/// Helper to create TN and interleaved layouts GEMM layouts.
template <
  int InterleaveK,
  typename ElementA_,
  typename ElementB_,
  typename ElementC_,
  typename ElementCompute_,
  typename ElementAccumulator_ = ElementCompute_,
  typename ElementD_ = ElementC_,
  typename ConvertOp_ = NumericConverter<ElementC_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
void make_gemm_interleaved_layouts(Manifest &manifest) {
  
  make_gemm<
    ElementA_, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone,
    ElementB_, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone,
    ElementC_, cutlass::layout::ColumnMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);

}

/// Helper to real-valued GEMM with canonical layouts
template <
  typename ElementA_,
  typename ElementB_,
  typename ElementC_,
  typename ElementCompute_,
  typename ElementAccumulator_ = ElementCompute_,
  typename ElementD_ = ElementC_,
  typename ConvertOp_ = NumericConverter<ElementD_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
void make_gemm_real_canonical_layouts(Manifest &manifest) {
  make_gemm_canonical_layouts<
    ElementA_, cutlass::ComplexTransform::kNone,
    ElementB_, cutlass::ComplexTransform::kNone,
    ElementC_,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);  
}

// Helper to create all complex transformation permutations
template <
  typename ElementA_,
  typename ElementB_,
  typename ElementC_,
  typename ElementCompute_,
  typename ElementAccumulator_ = ElementCompute_,
  typename ElementD_ = ElementC_,
  typename ConvertOp_ = NumericConverter<ElementD_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
void make_gemm_complex_canonical_layouts(Manifest &manifest) {

  make_gemm_canonical_layouts<
    ElementA_, cutlass::ComplexTransform::kNone,
    ElementB_, cutlass::ComplexTransform::kNone,
    ElementC_,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);
  
  make_gemm_canonical_layouts<
    ElementA_, cutlass::ComplexTransform::kConjugate,
    ElementB_, cutlass::ComplexTransform::kConjugate,
    ElementC_,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);

  make_gemm_canonical_layouts<
    ElementA_, cutlass::ComplexTransform::kNone,
    ElementB_, cutlass::ComplexTransform::kConjugate,
    ElementC_,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);
  
  make_gemm_canonical_layouts<
    ElementA_, cutlass::ComplexTransform::kConjugate,
    ElementB_, cutlass::ComplexTransform::kNone,
    ElementC_,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

