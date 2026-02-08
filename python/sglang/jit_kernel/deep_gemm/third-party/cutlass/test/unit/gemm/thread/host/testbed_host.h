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
    \brief Unit tests for thread-level GEMM
*/

#pragma once

#include "cutlass/gemm/thread/mma.h"
#include "cutlass/layout/vector.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

namespace test {
namespace gemm {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Thread-level matrix multiply-accumulate
template <typename Mma>
void kernel(
  typename Mma::ElementC *D,
  typename Mma::ElementA const *A,
  typename Mma::ElementB const *B,
  typename Mma::ElementC const *C) {

  auto ptr_D = reinterpret_cast<cutlass::Array<typename Mma::ElementC, Mma::Shape::kMN> *>(D);
  auto ptr_A = reinterpret_cast<cutlass::Array<typename Mma::ElementA, Mma::Shape::kMK> const *>(A);
  auto ptr_B = reinterpret_cast<cutlass::Array<typename Mma::ElementB, Mma::Shape::kKN> const *>(B);
  auto ptr_C = reinterpret_cast<cutlass::Array<typename Mma::ElementC, Mma::Shape::kMN> const *>(C);

  Mma mma;

  auto a = *ptr_A;
  auto b = *ptr_B;
  auto c = *ptr_C;

  using Btype = typename Mma::ElementB;
  cutlass::Array<typename Mma::ElementC, Mma::Shape::kMN> d;

  mma(d, a, b, c);

  *ptr_D = d;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape,
  /// Data type of A elements
  typename ElementA,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA,
  /// Data type of B elements
  typename ElementB,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB,
  /// Element type of C matrix
  typename ElementC,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC
>
struct Testbed {

  /// Thread-level matrix multiply-accumulate operator
  using Mma = cutlass::gemm::thread::Mma<
    Shape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC
  >;

  //
  // Data members
  //

  cutlass::HostTensor<ElementA, LayoutA> tensor_A;
  cutlass::HostTensor<ElementB, LayoutB> tensor_B;
  cutlass::HostTensor<ElementC, LayoutC> tensor_C;
  cutlass::HostTensor<ElementC, LayoutC> tensor_D_computed;
  cutlass::HostTensor<ElementC, LayoutC> tensor_D_reference;

  //
  // Methods
  //

  /// Allocates workspace in device memory
  Testbed() {

    tensor_A.reset(cutlass::make_Coord(Shape::kM, Shape::kK), false);
    tensor_B.reset(cutlass::make_Coord(Shape::kK, Shape::kN), false);
    tensor_C.reset(cutlass::make_Coord(Shape::kM, Shape::kN), false);
    tensor_D_computed.reset(cutlass::make_Coord(Shape::kM, Shape::kN), false);
    tensor_D_reference.reset(cutlass::make_Coord(Shape::kM, Shape::kN), false);
  }

  /// Runs the test
  bool run() {

    //
    // initialize device memory
    //

    cutlass::reference::host::detail::RandomUniformFunc< ElementA > tfill_rand_func( 
      0,  // seed
      10, // max
      0,  // min
      0); // bits after decimal
                                                                              
    cutlass::reference::host::detail::TensorFillRandomUniformFunc< ElementA, LayoutA > tfill_rand(
      tensor_A.host_view(),
      tfill_rand_func); 

    for (auto i=0; i< Shape::kM; i++)
      for (auto j=0; j< Shape::kK; j++)
        tfill_rand(cutlass::make_Coord(i,j));

    cutlass::reference::host::BlockFillSequential(
      tensor_B.host_data(),
      tensor_B.capacity(),
      ElementB(1),
      ElementB(2)
    );

    cutlass::reference::host::TensorFill(
      tensor_C.host_view(),
      ElementC(0)
    );

    cutlass::reference::host::TensorFill(
      tensor_D_computed.host_view(),
      ElementC(0)
    );

    cutlass::reference::host::TensorFill(
      tensor_D_reference.host_view(),
      ElementC(0)
    );


    // Host side call
    kernel<Mma>(
      tensor_D_computed.host_data(),
      tensor_A.host_data(),
      tensor_B.host_data(),
      tensor_C.host_data());

    //
    // Reference implementation
    //

    cutlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                   ElementC, LayoutC, ElementC, ElementC>
        reference_gemm;

    reference_gemm(
      {Shape::kM, Shape::kN, Shape::kK},
      ElementC(1),
      tensor_A.host_ref(),
      tensor_B.host_ref(),
      ElementC(0),
      tensor_D_reference.host_ref()
    );

    //
    // Verify equivalence
    //

    // compare
    bool passed = cutlass::reference::host::TensorEquals(
      tensor_D_computed.host_view(),
      tensor_D_reference.host_view()
    );

    EXPECT_TRUE(passed)
      << "A:\n" << tensor_A.host_view() << "\n\n"
      << "B:\n" << tensor_B.host_view() << "\n\n"
      << "C:\n" << tensor_C.host_view() << "\n\n"
      << "Reference:\n" << tensor_D_reference.host_view() << "\n\n"
      << "Computed:\n" << tensor_D_computed.host_view() << std::endl;
    
    
    return passed;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace gemm
} // namespace test
