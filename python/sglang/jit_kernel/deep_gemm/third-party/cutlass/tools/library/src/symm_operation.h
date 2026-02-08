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
   \brief Defines operations for all Symm operation kinds (Symm, Hemm) 
    in CUTLASS Library.

  
*/

#pragma once
#include <iostream>
#include "cutlass/cutlass.h"

#include "cutlass/gemm/device/symm.h"
#include "cutlass/gemm/kernel/default_symm_universal.h"

#include "cutlass/library/library.h"
#include "library_internal.h"
#include "cutlass/core_io.h"
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class SymmOperationBase : public Operation {
public:
  using Operator = Operator_;
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;
  static BlasMode const kBlasMode = Operator::kBlasMode;
  static SideMode const kSideModeA = Operator::kSideModeA;
  static FillMode const kFillModeA = Operator::kFillModeA;

  using OperatorArguments = typename Operator::Arguments;

protected:

  /// 
  SymmDescription description_;

public:

  /// Constructor
  SymmOperationBase(char const *name = "unknown_symm") {

    description_.name = name;
    description_.provider = Provider::kCUTLASS;
    description_.symm_kind = SymmKind::kUniversal;
    description_.side_mode = kSideModeA;    
    description_.fill_mode = kFillModeA;    
    description_.blas_mode = kBlasMode;

    description_.kind = OperationKind::kSymm;

    description_.tile_description.threadblock_shape = make_Coord(
      Operator::ThreadblockShape::kM,
      Operator::ThreadblockShape::kN,
      Operator::ThreadblockShape::kK);

    description_.tile_description.threadblock_stages = Operator::kStages;

    description_.tile_description.warp_count = make_Coord(
      Operator::SymmKernel::WarpCount::kM,
      Operator::SymmKernel::WarpCount::kN,
      Operator::SymmKernel::WarpCount::kK);
    
    description_.tile_description.math_instruction.instruction_shape = make_Coord(
      Operator::InstructionShape::kM,
      Operator::InstructionShape::kN,
      Operator::InstructionShape::kK);

    description_.tile_description.math_instruction.element_accumulator = 
      NumericTypeMap<ElementAccumulator>::kId;

    description_.tile_description.math_instruction.opcode_class = 
      OpcodeClassMap<typename Operator::OperatorClass>::kId;

    description_.tile_description.math_instruction.math_operation =
      MathOperationMap<typename Operator::Operator>::kId;

    description_.tile_description.minimum_compute_capability = 
      ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMin;

    description_.tile_description.maximum_compute_capability = 
      ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMax;
    
    description_.A = make_TensorDescription<ElementA, LayoutA>(Operator::kAlignmentA);
    description_.B = make_TensorDescription<ElementB, LayoutB>(Operator::kAlignmentB);
    description_.C = make_TensorDescription<ElementC, LayoutC>(Operator::kAlignmentC);
    description_.element_epilogue = NumericTypeMap<ElementCompute>::kId;

    description_.split_k_mode = SplitKMode::kNone;
  }
  
  /// Returns the description of the SYMM operation
  virtual OperationDescription const & description() const {
    return description_;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class SymmOperation : public SymmOperationBase<Operator_> {
public:

  using Operator = Operator_;
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;

  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  static BlasMode const kBlasMode = Operator::kBlasMode;
  static SideMode const kSideModeA = Operator::kSideModeA;
  static FillMode const kFillModeA = Operator::kFillModeA;

  using OperatorArguments = typename Operator::Arguments;

public:

  /// Constructor
  SymmOperation(char const *name = "unknown_symm"): 
    SymmOperationBase<Operator_>(name) {

    this->description_.symm_kind = SymmKind::kUniversal;
  }

protected:

  /// Constructs the arguments structure given the configuration and arguments
  static Status construct_arguments_(
    OperatorArguments &operator_args,
    SymmConfiguration const *configuration) {

    //operator_args.mode = configuration->mode;

    operator_args.problem_size = configuration->problem_size;
    operator_args.batch_count = configuration->batch_count;

    operator_args.lda = int(configuration->lda);
    operator_args.ldb = int(configuration->ldb);
    operator_args.ldc = int(configuration->ldc);
    operator_args.ldd = int(configuration->ldd);
    
    return Status::kSuccess;
  }

  /// Constructs the arguments structure given the configuration and arguments
  static Status update_arguments_(
    OperatorArguments &operator_args,
    SymmArguments const *arguments) {
    
    if (arguments->pointer_mode == ScalarPointerMode::kHost) {
      typename Operator::EpilogueOutputOp::Params params(
        *static_cast<ElementCompute const *>(arguments->alpha),
        *static_cast<ElementCompute const *>(arguments->beta)
      );
      operator_args.epilogue = params;
    }
    else if (arguments->pointer_mode == ScalarPointerMode::kDevice){
      typename Operator::EpilogueOutputOp::Params params(
        static_cast<ElementCompute const *>(arguments->alpha),
        static_cast<ElementCompute const *>(arguments->beta)
      );
      operator_args.epilogue = params; 
    }
    else {
      return Status::kErrorInvalidProblem;
    }

    // update arguments
    operator_args.ptr_A = arguments->A;
    operator_args.ptr_B = arguments->B;
    operator_args.ptr_C = arguments->C;
    operator_args.ptr_D = arguments->D;

    operator_args.batch_stride_A = arguments->batch_stride_A;
    operator_args.batch_stride_B = arguments->batch_stride_B;
    operator_args.batch_stride_C = arguments->batch_stride_C;
    operator_args.batch_stride_D = arguments->batch_stride_D;
    
    if (arguments->use_pdl) {
      return Status::kErrorNotSupported; 
    }

    return Status::kSuccess;
  }

public:

  /// Returns success if the operation can proceed
  virtual Status can_implement(
    void const *configuration_ptr, 
    void const *arguments_ptr) const {
    
    SymmConfiguration const *configuration = 
      static_cast<SymmConfiguration const *>(configuration_ptr);

    SymmArguments const *arguments = 
      static_cast<SymmArguments const *>(arguments_ptr);

    OperatorArguments args;

    Status status = construct_arguments_(args, configuration);

    if (status != Status::kSuccess) {
      return status;
    }

    status = update_arguments_(args, arguments);

    if (status != Status::kSuccess) {
      return status;
    }

    return Operator::can_implement(args);
  }
  
  /// Gets the host-side workspace
  virtual uint64_t get_host_workspace_size(
    void const *configuration) const {

    return sizeof(Operator);
  }
  
  /// Gets the device-side workspace
  virtual uint64_t get_device_workspace_size(
    void const *configuration_ptr,
    void const *arguments_ptr = nullptr) const {

    OperatorArguments args;

    Status status = construct_arguments_(
      args, 
      static_cast<SymmConfiguration const *>(configuration_ptr));

    if (status != Status::kSuccess) {
      return 0;
    }

    uint64_t size = Operator::get_workspace_size(args);

    return size;
  }
  
  /// Initializes the workspace
  virtual Status initialize(
    void const *configuration_ptr, 
    void *host_workspace, 
    void *device_workspace, 
    cudaStream_t stream = nullptr) const {

    OperatorArguments args;

    Status status = construct_arguments_(
      args, 
      static_cast<SymmConfiguration const *>(configuration_ptr));

    if (status != Status::kSuccess) {
      return status;
    }

    Operator *op = new (host_workspace) Operator;
    
    //std::cout << "initialize() library::SymmOperation" << std::endl;
    //print_operator_args(args);
    status = op->initialize(args, device_workspace, stream);
    
    return status;
  }

  /// Runs the kernel
  virtual Status run(
    void const *arguments_ptr,
    void *host_workspace, 
    void *device_workspace = nullptr, 
    cudaStream_t stream = nullptr) const {
    OperatorArguments args;
    
    Status status = update_arguments_(
      args, 
      static_cast<SymmArguments const *>(arguments_ptr));

    if (status != Status::kSuccess) {
      return status;
    }
    
    Operator *op = static_cast<Operator *>(host_workspace);

    bool need_swapped_matrices = (kSideModeA == SideMode::kLeft && 
                                    std::is_same<typename Operator::LayoutC, layout::ColumnMajor>::value) ||
                                 (kSideModeA == SideMode::kRight &&
                                    std::is_same<typename Operator::LayoutC, layout::RowMajor>::value);
    if (need_swapped_matrices) {
      status = op->update(args.swapped_matrices(), device_workspace);
    } else {
      status = op->update(args, device_workspace);
    } 

    if (status != Status::kSuccess) {
      return status;
    }
    
    //std::cout << "run() library::SymmOperation" << std::endl;
    //print_operator_args(args);
    status = op->run(stream);
    
    return status;
  }

  /// Call print_operator_args  from the Conv2dOperation::initialize()
  // to dump arguments passed on to cutlass operator for debugging
  void print_operator_args(OperatorArguments &operator_args) const {
    std::cout << "SymmOperation::OperatorArguments" << std::endl
              << "  problem_size:" << std::endl 
              << operator_args.problem_size << std::endl
              << "  epilogue (alpha, beta): "
              << operator_args.epilogue.alpha << ", " 
              << operator_args.epilogue.beta << std::endl
              << "  ref_A (ptr, {stride}): " 
              << operator_args.ptr_A << ", {"
              << operator_args.lda << "}" << std::endl
              << "  ref_B (ptr, {stride}): " 
              << operator_args.ptr_B << ", {"
              << operator_args.ldb << "}" << std::endl
              << "  ref_C (ptr, {stride}): "
              << operator_args.ptr_C << ", {"
              << operator_args.ldc << "}" << std::endl
              << "  ref_D (ptr, {stride}): "
              << operator_args.ptr_D << ", {"
              << operator_args.ldd << "}" << std::endl;
  } 
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
