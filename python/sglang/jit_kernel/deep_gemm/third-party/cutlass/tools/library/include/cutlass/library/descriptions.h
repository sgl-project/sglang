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

#include <cutlass/library/types.h>
#include <cutlass/blas3_types.h>
#include <cutlass/gemm_coord.h>

#include <optional>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

struct MathInstructionDescription {

  /// Shape of the target math instruction
  cutlass::gemm::GemmCoord instruction_shape;

  /// Describes the data type of the internal accumulator
  NumericTypeID element_accumulator;

  /// Classification of math instruction
  OpcodeClassID opcode_class;

  /// Type of math operation performed
  MathOperationID math_operation;

  //
  // Methods
  //

  MathInstructionDescription(
    cutlass::gemm::GemmCoord instruction_shape = cutlass::gemm::GemmCoord(),
    NumericTypeID element_accumulator = NumericTypeID::kInvalid,
    OpcodeClassID opcode_class = OpcodeClassID::kInvalid,
    MathOperationID math_operation = MathOperationID::kMultiplyAdd
  ):
    instruction_shape(instruction_shape), 
    element_accumulator(element_accumulator), 
    opcode_class(opcode_class),
    math_operation(math_operation) {}

  // Equality operator
  inline
  bool operator==(MathInstructionDescription const& rhs) const{
    return (
      (instruction_shape == rhs.instruction_shape) &&
      (element_accumulator == rhs.element_accumulator) &&
      (opcode_class == rhs.opcode_class) &&
      (math_operation == rhs.math_operation));
  }

  // Inequality operator
  inline
  bool operator!=(MathInstructionDescription const& rhs) const {
    return !(*this == rhs);
  }

};

/// Structure describing the tiled structure of a GEMM-like computation
struct TileDescription {

  /// Describes the shape of a threadblock (in elements)
  cutlass::gemm::GemmCoord threadblock_shape;

  /// Describes the number of pipeline stages in the threadblock-scoped mainloop
  int threadblock_stages;

  /// Number of warps in each logical dimension
  cutlass::gemm::GemmCoord warp_count;

  /// Core math instruction
  MathInstructionDescription math_instruction;

  /// Minimum compute capability (e.g. 70, 75) of a device eligible to run the operation.
  int minimum_compute_capability;

  /// Minimum compute capability (e.g. 70, 75) of a device eligible to run the operation.
  int maximum_compute_capability;

  /// Describes the shape of a cluster (in blocks)
  cutlass::gemm::GemmCoord cluster_shape;

  //
  // Methods
  //

  TileDescription(
    cutlass::gemm::GemmCoord threadblock_shape = cutlass::gemm::GemmCoord(),
    int threadblock_stages = 0,
    cutlass::gemm::GemmCoord warp_count = cutlass::gemm::GemmCoord(),
    MathInstructionDescription math_instruction = MathInstructionDescription(),
    int minimum_compute_capability = 0,
    int maximum_compute_capability = 0,
    cutlass::gemm::GemmCoord cluster_shape = cutlass::gemm::GemmCoord(1,1,1)
  ):
    threadblock_shape(threadblock_shape), 
    threadblock_stages(threadblock_stages), 
    warp_count(warp_count),
    math_instruction(math_instruction),
    minimum_compute_capability(minimum_compute_capability),
    maximum_compute_capability(maximum_compute_capability),
    cluster_shape(cluster_shape) { }

  // Equality operator
  inline
  bool operator==(TileDescription const& rhs) const{
    return (
      (threadblock_shape == rhs.threadblock_shape) &&
      (threadblock_stages == rhs.threadblock_stages) &&
      (warp_count == rhs.warp_count) &&
      (math_instruction == rhs.math_instruction) &&
      (minimum_compute_capability == rhs.minimum_compute_capability) &&
      (maximum_compute_capability == rhs.maximum_compute_capability));
  }

  // Inequality operator
  inline
  bool operator!=(TileDescription const& rhs) const {
    return !(*this == rhs);
  }
};

/// High-level description of an operation
struct OperationDescription {

  /// Unique identifier describing the operation
  char const * name;

  /// Operation provider
  Provider provider;

  /// Kind of operation
  OperationKind kind;

  /// Describes the tiled structure of a GEMM-like computation
  TileDescription tile_description;

  //
  // Methods
  //
  OperationDescription(
    char const * name = "unknown",
    Provider provider = Provider::kInvalid,
    OperationKind kind = OperationKind::kInvalid, 
    TileDescription const&  tile_description = TileDescription()
  ):
    name(name), provider(provider), kind(kind), tile_description(tile_description) { }
};

/// Structure describing the properties of a tensor
struct TensorDescription {

  /// Numeric type of an individual element
  NumericTypeID element;

  /// Enumerant identifying the layout function for the tensor
  LayoutTypeID layout;

  /// Alignment restriction on pointers, strides, and extents
  int alignment;

  /// log2() of the maximum extent of each dimension
  int log_extent_range;

  /// log2() of the maximum value each relevant stride may have
  int log_stride_range;
  
  //
  // Methods
  //

  TensorDescription(
    NumericTypeID element = NumericTypeID::kInvalid,
    LayoutTypeID layout = LayoutTypeID::kInvalid,
    int alignment = 1,
    int log_extent_range = 24,
    int log_stride_range = 24
  ):
    element(element), 
    layout(layout), 
    alignment(alignment), 
    log_extent_range(log_extent_range), 
    log_stride_range(log_stride_range)  { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Description of all GEMM computations
struct GemmDescription : public OperationDescription {

  /// Indicates the kind of GEMM performed
  GemmKind gemm_kind;
  
  /// Describes the A operand
  TensorDescription A;

  /// Describes the B operand
  TensorDescription B;

  /// Describes the source matrix
  TensorDescription C;

  /// Describes the destination matrix
  TensorDescription D;

  /// Describes the sparse meta matrices
  TensorDescription E;

  /// Describes the data type of the scalars passed to the epilogue
  NumericTypeID element_epilogue;

  /// Describes the structure of parallel reductions
  SplitKMode split_k_mode;

  /// Transformation on A operand
  ComplexTransform transform_A;

  /// Transformation on B operand
  ComplexTransform transform_B;

  //
  // Methods
  //

  GemmDescription(
    GemmKind gemm_kind = GemmKind::kGemm,
    TensorDescription const& A = TensorDescription(),
    TensorDescription const& B = TensorDescription(),
    TensorDescription const& C = TensorDescription(),
    TensorDescription const& D = TensorDescription(),
    NumericTypeID element_epilogue = NumericTypeID::kInvalid,
    SplitKMode split_k_mode = SplitKMode::kNone,
    ComplexTransform transform_A = ComplexTransform::kNone,
    ComplexTransform transform_B = ComplexTransform::kNone
  ):
    gemm_kind(gemm_kind),
    A(A),
    B(B),
    C(C),
    D(D),
    element_epilogue(element_epilogue),
    split_k_mode(split_k_mode),
    transform_A(transform_A),
    transform_B(transform_B) {} 

  GemmDescription(
    OperationDescription op_desc,
    GemmKind gemm_kind,
    TensorDescription const& A,
    TensorDescription const& B,
    TensorDescription const& C,
    TensorDescription const& D,
    NumericTypeID element_epilogue,
    SplitKMode split_k_mode,
    ComplexTransform transform_A,
    ComplexTransform transform_B
  ):
    OperationDescription(op_desc),
    gemm_kind(gemm_kind),
    A(A),
    B(B),
    C(C),
    D(D),
    element_epilogue(element_epilogue),
    split_k_mode(split_k_mode),
    transform_A(transform_A),
    transform_B(transform_B) {}
};

struct BlockScaleDescription {
  /// Describes the SFA operand
  TensorDescription SFA;

  /// Describes the SFB operand
  TensorDescription SFB;

  /// Describes the SFD operand
  TensorDescription SFD;

  /// Describes the input ScaleFactor VectorSize
  int SFMVecSize;
  int SFNVecSize;
  int SFKVecSize;

  /// Describes the Output ScaleFactor VectorSize
  int EpilogueSFVecSize;

  /// Describes the underlying kind of scaling: 
  /// Tensor Core supported (BlockScaled) or manual scaling (Blockwise)
  OperationKind kind;
};

struct GroupedGemmDescription : public OperationDescription {
  GemmDescription gemm;
  std::optional<BlockScaleDescription> block_scales;
};

/// Description of all GEMM computations
struct BlockScaledGemmDescription : public OperationDescription {

  /// Indicates the kind of GEMM performed
  GemmKind gemm_kind;

  /// Describes the A operand
  TensorDescription A;

  /// Describes the B operand
  TensorDescription B;

  /// Describes the source matrix
  TensorDescription C;

  /// Describes the destination matrix
  TensorDescription D;

  /// Describes the SFA operand
  TensorDescription SFA;

  /// Describes the SFB operand
  TensorDescription SFB;

  /// Describes the SFD operand 
  TensorDescription SFD; 

  /// Describes the data type of the scalars passed to the epilogue
  NumericTypeID element_epilogue;

  /// Describes the structure of parallel reductions
  SplitKMode split_k_mode;

  /// Transformation on A operand
  ComplexTransform transform_A;

  /// Transformation on B operand
  ComplexTransform transform_B;

  /// Describes the input ScaleFactor VectorSize 
  int SFVecSize;

  /// Describes the Output ScaleFactor VectorSize 
  int EpilogueSFVecSize;

  //
  // Methods
  //

  BlockScaledGemmDescription(
    GemmKind gemm_kind = GemmKind::kGemm,
    TensorDescription const& A = TensorDescription(),
    TensorDescription const& B = TensorDescription(),
    TensorDescription const& C = TensorDescription(),
    TensorDescription const& D = TensorDescription(),
    NumericTypeID element_epilogue = NumericTypeID::kInvalid,
    SplitKMode split_k_mode = SplitKMode::kNone,
    ComplexTransform transform_A = ComplexTransform::kNone,
    ComplexTransform transform_B = ComplexTransform::kNone
  ):
    gemm_kind(gemm_kind),
    A(A),
    B(B),
    C(C),
    D(D),
    element_epilogue(element_epilogue),
    split_k_mode(split_k_mode),
    transform_A(transform_A),
    transform_B(transform_B) {} 

  BlockScaledGemmDescription(
    OperationDescription op_desc,
    GemmKind gemm_kind,
    TensorDescription const& A,
    TensorDescription const& B,
    TensorDescription const& C,
    TensorDescription const& D,
    NumericTypeID element_epilogue,
    SplitKMode split_k_mode,
    ComplexTransform transform_A,
    ComplexTransform transform_B
  ):
    OperationDescription(op_desc),
    gemm_kind(gemm_kind),
    A(A),
    B(B),
    C(C),
    D(D),
    element_epilogue(element_epilogue),
    split_k_mode(split_k_mode),
    transform_A(transform_A),
    transform_B(transform_B) {}
};

/// Description of all GEMM computations
struct BlockwiseGemmDescription : public OperationDescription {

  /// Indicates the kind of GEMM performed
  GemmKind gemm_kind;

  /// Describes the A operand
  TensorDescription A;

  /// Describes the B operand
  TensorDescription B;

  /// Describes the source matrix
  TensorDescription C;

  /// Describes the destination matrix
  TensorDescription D;

  /// Describes the SFA operand
  TensorDescription SFA;

  /// Describes the SFB operand
  TensorDescription SFB;

  /// Describes the data type of the scalars passed to the epilogue
  NumericTypeID element_epilogue;

  /// Describes the structure of parallel reductions
  SplitKMode split_k_mode;

  /// Transformation on A operand
  ComplexTransform transform_A;

  /// Transformation on B operand
  ComplexTransform transform_B;

  /// Describes the input ScaleFactor VectorSize 
  int SFMVecSize;
  int SFNVecSize;
  int SFKVecSize;

  //
  // Methods
  //

  BlockwiseGemmDescription(
    GemmKind gemm_kind = GemmKind::kGemm,
    TensorDescription const& A = TensorDescription(),
    TensorDescription const& B = TensorDescription(),
    TensorDescription const& C = TensorDescription(),
    TensorDescription const& D = TensorDescription(),
    NumericTypeID element_epilogue = NumericTypeID::kInvalid,
    SplitKMode split_k_mode = SplitKMode::kNone,
    ComplexTransform transform_A = ComplexTransform::kNone,
    ComplexTransform transform_B = ComplexTransform::kNone
  ):
    gemm_kind(gemm_kind),
    A(A),
    B(B),
    C(C),
    D(D),
    element_epilogue(element_epilogue),
    split_k_mode(split_k_mode),
    transform_A(transform_A),
    transform_B(transform_B) {} 

  BlockwiseGemmDescription(
    OperationDescription op_desc,
    GemmKind gemm_kind,
    TensorDescription const& A,
    TensorDescription const& B,
    TensorDescription const& C,
    TensorDescription const& D,
    NumericTypeID element_epilogue,
    SplitKMode split_k_mode,
    ComplexTransform transform_A,
    ComplexTransform transform_B
  ):
    OperationDescription(op_desc),
    gemm_kind(gemm_kind),
    A(A),
    B(B),
    C(C),
    D(D),
    element_epilogue(element_epilogue),
    split_k_mode(split_k_mode),
    transform_A(transform_A),
    transform_B(transform_B) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Description for structured sparse GEMMs.
struct SparseGemmDescription : public GemmDescription {

  /// Description structure for structured sparse GEMM
  SparseGemmDescription(
    GemmKind gemm_kind = GemmKind::kGemm,
    TensorDescription const& A = TensorDescription(),
    TensorDescription const& B = TensorDescription(),
    TensorDescription const& C = TensorDescription(),
    TensorDescription const& D = TensorDescription(),
    TensorDescription const& E = TensorDescription(),
    NumericTypeID element_epilogue = NumericTypeID::kInvalid,
    SplitKMode split_k_mode = SplitKMode::kNone,
    ComplexTransform transform_A = ComplexTransform::kNone,
    ComplexTransform transform_B = ComplexTransform::kNone
  ):
    GemmDescription(gemm_kind, A, B, C, D, element_epilogue, split_k_mode, transform_A, transform_B)
     {this->E = E;}
};

/// Description of all Reduction operations
struct ReductionDescription : public OperationDescription {

  /// Describes the data type of workspace
  NumericTypeID element_workspace;

  /// Describes the data type of final output
  NumericTypeID element_output;

  /// Describes the data type of the scalars passed to the epilogue
  NumericTypeID element_epilogue;
};

/// Description of all Rank K update computations (SYRK, HERK, SYR2K, HER2K)
struct RankKDescription : public OperationDescription {

  /// Indicates which device template is used (universal or regular)
  RankKKind rank_k_kind;

  /// Number of rank update (rank k or rank 2k)
  int num_ranks;
  
  /// Describes the A operand
  TensorDescription A;

  /// Describes the B operand (used only for SYR2K and HER2K)
  TensorDescription B;

  /// Describes the source and destination matrices
  TensorDescription C;

  /// Describes the fill mode for matrix C
  FillMode fill_mode;

  /// Describes the blas mode (symmetric/hermitian)
  BlasMode blas_mode;

  /// Describes the data type of the scalars passed to the epilogue
  NumericTypeID element_epilogue;

  /// Describes the structure of parallel reductions
  SplitKMode split_k_mode;

  /// Transformation on A operand
  ComplexTransform transform_A;

  /// Transformation on B operand
  ComplexTransform transform_B;

  //
  // Methods
  //

  RankKDescription(
    RankKKind rank_k_kind = RankKKind::kUniversal,
    int num_ranks = 1,
    TensorDescription const& A = TensorDescription(),
    TensorDescription const& B = TensorDescription(),
    TensorDescription const& C = TensorDescription(),
    FillMode fill_mode = FillMode::kInvalid,
    BlasMode blas_mode = BlasMode::kInvalid,
    NumericTypeID element_epilogue = NumericTypeID::kInvalid,
    SplitKMode split_k_mode = SplitKMode::kNone,
    ComplexTransform transform_A = ComplexTransform::kNone,
    ComplexTransform transform_B = ComplexTransform::kNone
  ):
    rank_k_kind(rank_k_kind),
    num_ranks(num_ranks),
    A(A),
    B(B),
    C(C),
    fill_mode(fill_mode),
    blas_mode(blas_mode),
    element_epilogue(element_epilogue),
    split_k_mode(split_k_mode),
    transform_A(transform_A),
    transform_B(transform_B) {} 
};
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Description of all TRMM computations
struct TrmmDescription : public OperationDescription {

  /// Indicates the kind of TRMM performed
  TrmmKind trmm_kind;
  
  /// Describes the A operand
  TensorDescription A;

  /// Describes the side mode for matrix A
  SideMode side_mode;

  /// Describes the fill mode for matrix A
  FillMode fill_mode;

  /// Describes the diag type for matrix A
  DiagType diag_type;

  /// Describes the B operand
  TensorDescription B;

  /// Describes the source and destination matrices
  TensorDescription D;

  /// Describes the data type of the scalars passed to the epilogue
  NumericTypeID element_epilogue;

  /// Describes the structure of parallel reductions
  SplitKMode split_k_mode;

  /// Transformation on A operand
  ComplexTransform transform_A;

  //
  // Methods
  //

  TrmmDescription(
    TrmmKind trmm_kind = TrmmKind::kUniversal,
    TensorDescription const& A = TensorDescription(),
    SideMode side_mode = SideMode::kInvalid,
    FillMode fill_mode = FillMode::kInvalid,
    DiagType diag_type = DiagType::kInvalid,
    TensorDescription const& B = TensorDescription(),
    TensorDescription const& D = TensorDescription(),
    NumericTypeID element_epilogue = NumericTypeID::kInvalid,
    SplitKMode split_k_mode = SplitKMode::kNone,
    ComplexTransform transform_A = ComplexTransform::kNone
  ):
    trmm_kind(trmm_kind),
    A(A),
    side_mode(side_mode),
    fill_mode(fill_mode),
    diag_type(diag_type),
    B(B),
    D(D),
    element_epilogue(element_epilogue),
    split_k_mode(split_k_mode),
    transform_A(transform_A) {} 
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Description of all SYMM/HEMM update computations
struct SymmDescription : public OperationDescription {

  /// Indicates which device template is used (universal or regular)
  SymmKind symm_kind;
  
  /// Describes the A operand
  TensorDescription A;

  /// Describes the B operand 
  TensorDescription B;

  /// Describes the source and destination matrices
  TensorDescription C;

  /// Describes the side mode for matrix A
  SideMode side_mode;

  /// Describes the fill mode for matrix A
  FillMode fill_mode;

  /// Describes the blas mode (symmetric/hermitian)
  BlasMode blas_mode;

  /// Describes the data type of the scalars passed to the epilogue
  NumericTypeID element_epilogue;

  /// Describes the structure of parallel reductions
  SplitKMode split_k_mode;

  /// Transformation on A operand
  ComplexTransform transform_A;

  /// Transformation on B operand
  ComplexTransform transform_B;

  //
  // Methods
  //

  SymmDescription(
    SymmKind symm_kind = SymmKind::kUniversal,
    TensorDescription const& A = TensorDescription(),
    TensorDescription const& B = TensorDescription(),
    TensorDescription const& C = TensorDescription(),
    SideMode side_mode = SideMode::kInvalid,
    FillMode fill_mode = FillMode::kInvalid,
    BlasMode blas_mode = BlasMode::kInvalid,
    NumericTypeID element_epilogue = NumericTypeID::kInvalid,
    SplitKMode split_k_mode = SplitKMode::kNone,
    ComplexTransform transform_A = ComplexTransform::kNone,
    ComplexTransform transform_B = ComplexTransform::kNone
  ):
    symm_kind(symm_kind),
    A(A),
    B(B),
    C(C),
    side_mode(side_mode),
    fill_mode(fill_mode),
    blas_mode(blas_mode),
    element_epilogue(element_epilogue),
    split_k_mode(split_k_mode),
    transform_A(transform_A),
    transform_B(transform_B) {} 
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Description of all Conv2d operations
struct ConvDescription : public OperationDescription {
  /// Describes the convolution dimension support (2D or 3D)
  int conv_dim;
  
  /// Describes the kind of convolution
  ConvKind conv_kind;

  /// Describes the type of iterator algorithm (analytic or precomputed)
  IteratorAlgorithmID iterator_algorithm;

  /// Describes the A operand
  TensorDescription A;

  /// Describes the B operand
  TensorDescription B;

  /// Describes the C operand
  TensorDescription C;

  /// Describes the data type of the scalars passed to the epilogue
  NumericTypeID element_epilogue;

  //
  // Methods
  //
  // Returns Activation TensorDescription
  TensorDescription activation() const {
    switch(conv_kind) {
      case library::ConvKind::kFprop : return A;
      case library::ConvKind::kDgrad : return C;
      case library::ConvKind::kWgrad : return B;
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }

  // Returns Filter TensorDescription
  TensorDescription filter() const {
    switch(conv_kind) {
      case library::ConvKind::kFprop : return B;
      case library::ConvKind::kDgrad : return B;
      case library::ConvKind::kWgrad : return C;
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }

  // Returns Output TensorDescription
  TensorDescription output() const {
    switch(conv_kind) {
      case library::ConvKind::kFprop : return C;
      case library::ConvKind::kDgrad : return A;
      case library::ConvKind::kWgrad : return A;
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
