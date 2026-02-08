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
/*! \file
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>
#include <sstream>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/core_io.h"

#include "testbed.h"


namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// List of Gemm internal paramters this testbed supports user verification
//
enum class ParameterID {

  // Threadblock-level parameters 
  kSmemASize,
  kSmemBSize,

  // Warp-level parameters
  kWarpFragmentASize,
  kWarpFragmentBSize,
  kWarpFragmentCSize,
  kInvalid
};

struct Reference {
  ParameterID parameter_id;

  union {
    int value;
    
    struct {
      int m, n, k;
    } gemm_shape;

    struct {
      int row, column;
    } matrix_shape;
  };

  std::string error_msg;

  Reference(
    ParameterID parameter_id_, 
    int value_=-1, 
    std::string const &error_msg_="") : parameter_id(parameter_id_), value(value_), error_msg(error_msg_) {} 
};


template <typename Gemm>
struct TestbedSanity {

  //
  // Type definitions (All Gemm types top down) 
  //

  // Unpacking Gemm types in the following order
  // Kernel-level > Threadblock-level > Warp-level > Instruction-level

  // kernel-level cutlass Gemm
  using GemmKernel = typename Gemm::GemmKernel;

  //
  // Threadblock-level gemm types
  // 
  using MmaThreadBlock = typename GemmKernel::Mma;

  // Threadblock-level gemm shape covering one stage
  using ThreadblockShape = typename MmaThreadBlock::Shape;

  // Shared memory size covering all stages
  using SmemShapeA = typename MmaThreadBlock::Base::SharedStorage::ShapeA;
  using SmemPaddingA = typename MmaThreadBlock::Policy::SmemPaddingA;
  using SmemShapeB = typename MmaThreadBlock::Base::SharedStorage::ShapeB;
  using SmemPaddingB = typename MmaThreadBlock::Policy::SmemPaddingB;
  

  /// Number of stages 
  static int const kStages = MmaThreadBlock::Base::kStages;

  /// Number of warp-level GEMM oeprations
  static int const  kWarpGemmIterations = MmaThreadBlock::kWarpGemmIterations;


  //
  // Warp-level gemm types
  //

  // Warp-level gemm operator
  using MmaWarp = typename MmaThreadBlock::Operator;

  // Warp-level gemm shape covering all kgroups
  using WarpShape = typename MmaWarp::Shape;

  // Warp-level framents holding operands A & B operand and destination C
  using WarpFragmentA = typename MmaWarp::FragmentA;
  using WarpFragmentB = typename MmaWarp::FragmentB;
  using WarpFragmentC = typename MmaWarp::FragmentC;

  //
  // Instruction-level gemm types
  //

  // Instruction-level gemm operator
  using MmaInstruction = typename MmaWarp::Policy::Operator;

  // Instruction shape
  using InstructionShape = typename MmaInstruction::Shape;

  // Instruction-level framents holding operands A & B operand and destination C
  using InstructionFragmentA = typename MmaInstruction::FragmentA;
  using InstructionFragmentB = typename MmaInstruction::FragmentB;
  using InstructionFragmentC = typename MmaInstruction::FragmentC;

  //
  // Testbed types
  //

  // Vector of values holding user provided reference 
  using ReferenceVector = std::vector<Reference>;

  //
  // Data members
  //
  ReferenceVector references;

  //
  // Methods
  //

  TestbedSanity(ReferenceVector const &references_ = ReferenceVector()) : references(references_){ }

  // verify all parameter in ReferenceVector 
  bool verify() {
    for(auto ref : references)
      verify_parameter(ref);
    return true;
  }

  // verify parameter of type Reference
  void verify_parameter(Reference const& ref) {
    switch(ref.parameter_id) {
      case ParameterID::kWarpFragmentASize : EXPECT_TRUE(WarpFragmentA::kElements == ref.value) << *this; break;
      case ParameterID::kWarpFragmentBSize : EXPECT_TRUE(WarpFragmentB::kElements == ref.value) << *this; break;
      case ParameterID::kWarpFragmentCSize : EXPECT_TRUE(WarpFragmentC::kElements == ref.value) << *this; break;
    }
  } 

};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                             Overload output operators for TesbedSanity<Gemm>
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Gemm>
std::ostream & operator<<(std::ostream &out, TestbedSanity<Gemm> const &test) {


  out << "Gemm internal parameters" << std::endl 
      << "  Threadblock-level parameters:" << std::endl  
      << "     ThreadblockShape = " << typename TestbedSanity<Gemm>::ThreadblockShape() << std::endl
      << "     kStages = " << TestbedSanity<Gemm>::kStages << std::endl
      << "     kWarpGemmIterations = "<< TestbedSanity<Gemm>::kWarpGemmIterations << std::endl    
      <<"  Shared memory sizes:" << std::endl
      <<"    SmemPaddingA = " << typename TestbedSanity<Gemm>::SmemPaddingA() << std::endl
      <<"    SmemPaddingB = " << typename TestbedSanity<Gemm>::SmemPaddingB() << std::endl
      <<"      SmemShapeA = " << typename TestbedSanity<Gemm>::SmemShapeA() << std::endl
      <<"      SmemShapeB = " << typename TestbedSanity<Gemm>::SmemShapeB() << std::endl
      <<"  Warp-level parameters" << std::endl
      <<"    WarpShape = " << typename TestbedSanity<Gemm>::WarpShape() << std::endl
      <<"    Fragment sizes:" << std::endl
      <<"      WarpFragmentA::kElements = " << TestbedSanity<Gemm>::WarpFragmentA::kElements << std::endl
      <<"      WarpFragmentB::kElements = " << TestbedSanity<Gemm>::WarpFragmentB::kElements << std::endl
      <<"      WarpFragmentC::kElements = " << TestbedSanity<Gemm>::WarpFragmentC::kElements << std::endl
      <<"  Instruction-level parameters" << std::endl
      <<"    InstructionShape = " << typename TestbedSanity<Gemm>::InstructionShape() << std::endl
      <<"    Fragment sizes:" << std::endl
      <<"      InstructionFragmentA::kElements = " << TestbedSanity<Gemm>::InstructionFragmentA::kElements << std::endl
      <<"      InstructionFragmentB::kElements = " << TestbedSanity<Gemm>::InstructionFragmentB::kElements << std::endl
      <<"      InstructionFragmentC::kElements = " << TestbedSanity<Gemm>::InstructionFragmentC::kElements << std::endl;

  return out;
}

} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

