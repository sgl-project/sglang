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
  \brief Defines operations for all CONV operation kinds in CUTLASS Library.
*/

#pragma once
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv3d_fprop.h"
#include "cutlass/conv/kernel/default_conv3d_dgrad.h"
#include "cutlass/conv/kernel/default_conv3d_wgrad.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cutlass/library/library.h"
#include "library_internal.h"
#include "cutlass/util/host_tensor.h"

#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/core_io.h"
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class Conv3dOperationBase : public Operation {
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
  static cutlass::conv::IteratorAlgorithm const kIteratorAlgorithm = Operator::kIteratorAlgorithm;
  static cutlass::conv::Operator const kConvolutionalOperator = Operator::kConvolutionalOperator;

  using OperatorArguments = typename Operator::Arguments;

protected:

  /// 
  ConvDescription description_;

public:

  /// Constructor
  Conv3dOperationBase(char const *name = "unknown_conv3d") {

    description_.name = name;
    description_.provider = Provider::kCUTLASS;
    description_.kind = OperationKind::kConv3d;
    description_.conv_dim = Operator::kConvDim;
    
    description_.iterator_algorithm = IteratorAlgorithmMap<Operator::kIteratorAlgorithm>::kId;

    description_.tile_description.threadblock_shape = make_Coord(
      Operator::ThreadblockShape::kM,
      Operator::ThreadblockShape::kN,
      Operator::ThreadblockShape::kK);

    description_.tile_description.threadblock_stages = Operator::kStages;

    description_.tile_description.warp_count = make_Coord(
      Operator::UnderlyingKernel::WarpCount::kM,
      Operator::UnderlyingKernel::WarpCount::kN,
      Operator::UnderlyingKernel::WarpCount::kK);
    
    description_.tile_description.math_instruction.instruction_shape = make_Coord(
      Operator::InstructionShape::kM,
      Operator::InstructionShape::kN,
      Operator::InstructionShape::kK);

    description_.tile_description.math_instruction.element_accumulator = 
      NumericTypeMap<ElementAccumulator>::kId;

    description_.tile_description.math_instruction.opcode_class = 
      OpcodeClassMap<typename Operator::OperatorClass>::kId;

    description_.tile_description.minimum_compute_capability = 
      ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMin;

    description_.tile_description.maximum_compute_capability = 
      ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMax;
    
    description_.A = make_TensorDescription<ElementA, LayoutA>();
    description_.B = make_TensorDescription<ElementB, LayoutB>();
    description_.C = make_TensorDescription<ElementC, LayoutC>();
    description_.element_epilogue = NumericTypeMap<ElementCompute>::kId;

  }

  /// Returns the description of the GEMM operation
  virtual OperationDescription const & description() const {
    return description_;
  }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conv2d library operation class for cutlass profiler
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Operator_>
class Conv3dOperation : public Conv3dOperationBase<Operator_> {
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
  static cutlass::conv::Operator const kConvolutionalOperator = Operator::kConvolutionalOperator;

  using OperatorArguments = typename Operator::Arguments;

public:
    /// Constructor
  Conv3dOperation(char const *name = "unknown_conv3d_fprop") : Conv3dOperationBase<Operator_>(name) {
    this->description_.conv_kind = ConvKindMap<kConvolutionalOperator>::kId;
  }

protected:

  /// Constructs the arguments structure given the configuration and arguments
  static Status construct_arguments_(
    OperatorArguments &operator_args,
    Conv3dConfiguration const *configuration) {


    operator_args.problem_size     = configuration->problem_size;

    operator_args.ref_A = 
    {
      nullptr, 
      LayoutA::packed(implicit_gemm_tensor_a_extent(kConvolutionalOperator, configuration->problem_size))
    };
    
    operator_args.ref_B = 
    {
      nullptr, 
      LayoutB::packed(implicit_gemm_tensor_b_extent(kConvolutionalOperator, configuration->problem_size))
    };
    
    operator_args.ref_C = 
    {
      nullptr, 
      LayoutC::packed(implicit_gemm_tensor_c_extent(kConvolutionalOperator, configuration->problem_size))
    };
    
    operator_args.ref_D = 
    {
      nullptr, 
      LayoutC::packed(implicit_gemm_tensor_c_extent(kConvolutionalOperator, configuration->problem_size))
    };

    operator_args.split_k_mode     = configuration->split_k_mode;

    return Status::kSuccess;
  }

  /// Constructs the arguments structure given the configuration and arguments
  static Status update_arguments_(
    OperatorArguments &operator_args,
    ConvArguments const *arguments) {

    if (arguments->pointer_mode == ScalarPointerMode::kHost) {
      typename Operator::EpilogueOutputOp::Params params(
        *static_cast<ElementCompute const *>(arguments->alpha),
        *static_cast<ElementCompute const *>(arguments->beta)
      );
      operator_args.output_op = params;
    }
    else if (arguments->pointer_mode == ScalarPointerMode::kDevice){
      typename Operator::EpilogueOutputOp::Params params(
        static_cast<ElementCompute const *>(arguments->alpha),
        static_cast<ElementCompute const *>(arguments->beta)
      );
      operator_args.output_op = params; 
    }
    else {
      return Status::kErrorInvalidProblem;
    }

    operator_args.ref_A.reset(static_cast<ElementA *>(const_cast<void *>(arguments->A)));
    operator_args.ref_B.reset(static_cast<ElementB *>(const_cast<void *>(arguments->B)));
    operator_args.ref_C.reset(static_cast<ElementC *>(const_cast<void *>(arguments->C)));
    operator_args.ref_D.reset(static_cast<ElementC *>(const_cast<void *>(arguments->D)));

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

    Conv3dConfiguration const *configuration = 
      static_cast<Conv3dConfiguration const *>(configuration_ptr);

    ConvArguments const *arguments = 
      static_cast<ConvArguments const *>(arguments_ptr);

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
      static_cast<Conv3dConfiguration const *>(configuration_ptr));

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
      static_cast<Conv3dConfiguration const *>(configuration_ptr));

    if (status != Status::kSuccess) {
      return status;
    }

    Operator *op = new (host_workspace) Operator;
    //std::cout << "initialize library::Conv3dOperation" << std::endl;
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
      static_cast<ConvArguments const *>(arguments_ptr));

    if (status != Status::kSuccess) {
      return status;
    }

    Operator *op = static_cast<Operator *>(host_workspace);

    status = op->update(args, device_workspace);

    if (status != Status::kSuccess) {
      return status;
    }
    //std::cout << "run library::Conv3dOperation" << std::endl;
    //print_operator_args(args);
    return op->run(stream);
  }

  /// Call print_operator_args  from the Conv3dOperation::initialize()
  // to dump arguments passed on to cutlass operator for debugging
  void print_operator_args(OperatorArguments &operator_args) const {
    std::cout << "Conv3dOperation::OperatorArguments" << std::endl
              << "  problem_size: " 
              << operator_args.problem_size << std::endl
              << "  split_k_mode: "
              << (operator_args.split_k_mode == cutlass::conv::SplitKMode::kSerial ? "serial" : "parallel") << std::endl
              << "  epilogue (alpha, beta): "
              << operator_args.output_op.alpha << ", " 
              << operator_args.output_op.beta << std::endl
              << "  ref_A (ptr, {stride}): " 
              << operator_args.ref_A.data() << ", {"
              << operator_args.ref_A.stride(0) << ", " 
              << operator_args.ref_A.stride(1) << ", " 
              << operator_args.ref_A.stride(2) << ", " 
              << operator_args.ref_A.stride(3) << "}" << std::endl
              << "  ref_B (ptr, {stride}): " 
              << operator_args.ref_B.data() << ", {"
              << operator_args.ref_B.stride(0) << ", " 
              << operator_args.ref_B.stride(1) << ", " 
              << operator_args.ref_B.stride(2) << ", " 
              << operator_args.ref_B.stride(3) << "}" << std::endl
              << "  ref_C (ptr, {stride}): "
              << operator_args.ref_C.data() << ", {"
              << operator_args.ref_C.stride(0) << ", "
              << operator_args.ref_C.stride(1) << ", " 
              << operator_args.ref_C.stride(2) << ", " 
              << operator_args.ref_C.stride(3) << "}" << std::endl
              << "  ref_D (ptr, {stride}): "
              << operator_args.ref_D.data() << ", {"
              << operator_args.ref_D.stride(0) << ", "
              << operator_args.ref_D.stride(1) << ", " 
              << operator_args.ref_D.stride(2) << ", "
              << operator_args.ref_D.stride(3) << "}" << std::endl;
  } 
};

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
