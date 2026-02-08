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
   \brief Defines operations for reduction operation in CUTLASS Library.
*/

#pragma once
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/reduction/thread/reduction_operators.h"
#include "cutlass/reduction/device/reduce_split_k.h"

#include "cutlass/library/library.h"
#include "library_internal.h"
#include "cutlass/core_io.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class ReductionOperation : public Operation {
public:
  using Operator = Operator_;
  
  using ElementWorkspace = typename Operator::ElementWorkspace;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementOutput = typename Operator::ElementOutput;
  
  using ElementCompute = typename Operator::OutputOp::ElementCompute;

  using OperatorArguments = typename Operator::Arguments;

protected:

  /// 
  ReductionDescription description_;

public:

  /// Constructor
  ReductionOperation(char const *name = "unknown_reduction") {

    description_.name = name;
    description_.provider = Provider::kCUTLASS;
    description_.kind = OperationKind::kReduction;

    description_.tile_description.threadblock_shape = make_Coord(Operator::Shape::kRow, Operator::Shape::kColumn, 1);
    
    description_.tile_description.math_instruction.instruction_shape = make_Coord(1, 1, 1);
    description_.tile_description.math_instruction.element_accumulator = NumericTypeMap<ElementAccumulator>::kId;
    description_.tile_description.math_instruction.opcode_class = OpcodeClassID::kSimt;
    description_.tile_description.math_instruction.math_operation = MathOperationID::kAdd;

    description_.tile_description.minimum_compute_capability = 50;
    description_.tile_description.maximum_compute_capability = 1024;

    description_.element_workspace = NumericTypeMap<ElementWorkspace>::kId;
    description_.element_output = NumericTypeMap<ElementOutput>::kId;
    description_.element_epilogue = NumericTypeMap<ElementCompute>::kId;

  }
  
  /// Returns the description of the Reduction operation
  virtual OperationDescription const & description() const {
    return description_;
  }


protected:

  /// Constructs the arguments structure given the configuration and arguments
  static Status construct_arguments_(
    OperatorArguments &operator_args,
    ReductionConfiguration const *configuration) {

    operator_args.problem_size     = configuration->problem_size;
    operator_args.partitions       = configuration->partitions;
    operator_args.partition_stride = configuration->partition_stride;

    operator_args.workspace        = {nullptr, int(configuration->ldw)};
    operator_args.source           = {nullptr, int(configuration->lds)};
    operator_args.destination      = {nullptr, int(configuration->ldd)};

    return Status::kSuccess;
  }

  /// Constructs the arguments structure given the configuration and arguments
  static Status update_arguments_(
    OperatorArguments &operator_args,
    ReductionArguments const *arguments) {

    if (arguments->pointer_mode == ScalarPointerMode::kHost) {
      typename Operator::OutputOp::Params params(
        *static_cast<ElementCompute const *>(arguments->alpha),
        *static_cast<ElementCompute const *>(arguments->beta)
      );
      operator_args.output = params;
    }
    else if (arguments->pointer_mode == ScalarPointerMode::kDevice){
      typename Operator::OutputOp::Params params(
        static_cast<ElementCompute const *>(arguments->alpha),
        static_cast<ElementCompute const *>(arguments->beta)
      );
      operator_args.output = params; 
    }
    else {
      return Status::kErrorInvalidProblem;
    }
    
    operator_args.workspace.reset(static_cast<ElementWorkspace *>(const_cast<void *>(arguments->workspace)));
    operator_args.source.reset(static_cast<ElementOutput *>(const_cast<void *>(arguments->source)));
    operator_args.destination.reset(static_cast<ElementOutput *>(const_cast<void *>(arguments->destination)));

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

    ReductionConfiguration const *configuration = 
      static_cast<ReductionConfiguration const *>(configuration_ptr);

    ReductionArguments const *arguments = 
      static_cast<ReductionArguments const *>(arguments_ptr);

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
      static_cast<ReductionConfiguration const *>(configuration_ptr));

    if (status != Status::kSuccess) {
      return 0;
    }

    return Operator::get_workspace_size(args);
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
      static_cast<ReductionConfiguration const *>(configuration_ptr));

    if (status != Status::kSuccess) {
      return status;
    }

    Operator *op = new (host_workspace) Operator;
    //std::cout << "initialize library::Reduction" << std::endl;
    //print_operator_args(args);
    return op->initialize(args, device_workspace, stream);
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
      static_cast<ReductionArguments const *>(arguments_ptr));

    if (status != Status::kSuccess) {
      return status;
    }

    Operator *op = static_cast<Operator *>(host_workspace);

    status = op->update(args, device_workspace);

    if (status != Status::kSuccess) {
      return status;
    }

    //std::cout << "run library::Reduction" << std::endl;
    //print_operator_args(args);
    return op->run(stream);
  }

  /// Call print_operator_args  from the Reduction::initialize()
  // to dump arguments passed on to cutlass operator for debugging
  void print_operator_args(OperatorArguments &operator_args) const {
    std::cout << "Reduction::OperatorArguments" << std::endl
              << "  problem_size: " 
              << operator_args.problem_size << std::endl 
              << "  partitions: " 
              << operator_args.partitions << std::endl 
              << "  partition_stride: " 
              << operator_args.partition_stride << std::endl
              << "  epilogue (alpha, beta): "
              << operator_args.output.alpha << ", " 
              << operator_args.output.beta << std::endl
              << "  workspace (ptr, stride): "
              << operator_args.workspace.data() << ", " 
              << operator_args.workspace.stride(0) << std::endl
              << "  source (ptr, stride): " 
              << operator_args.source.data() << ", " 
              << operator_args.source.stride(0) << std::endl
              << "  destination (ptr, stride): " 
              << operator_args.destination.data() << ", " 
              << operator_args.destination.stride(0) << std::endl;
  }
};


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
