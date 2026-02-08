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
    \brief Unit tests for thread-level Reduction
*/

#pragma once

#include "cutlass/reduction/thread/reduce.h"

#include "cutlass/layout/vector.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"

namespace test {
namespace reduction {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the reduction
template <
  /// Data type of elements
  typename Element,
  /// Number of elements
  int N
>
struct Testbed_reduce_host {

  /// Thread-level reduction operator
  using Reduce = cutlass::reduction::thread::Reduce<
    cutlass::plus<Element>,
    cutlass::Array<Element, N>
  >;

  //
  // Data members
  //

  cutlass::Array<Element, N> tensor_in;
  cutlass::Array<Element, 1> reduced_tensor_computed;
  cutlass::Array<Element, 1> reduced_tensor_reference;

  //
  // Methods
  //

  /// Allocates workspace in device memory
  Testbed_reduce_host() {
    tensor_in.clear();
    reduced_tensor_computed.clear();
    reduced_tensor_reference.clear();
  }

  /// Runs the test
  bool run() {

    //
    // initialize memory
    //

    for(int i = 0; i < N; i++)
      tensor_in.at(i) = Element(i);

   
    Reduce reduce;

    cutlass::Array<Element, 1> *out_ptr = &reduced_tensor_computed;
    out_ptr[0] = reduce(tensor_in);

    //
    // Reference implementation
    //
    Element e(0);
    for (int i = 0; i < N; i++)
       e = e + Element(i);

    reduced_tensor_reference.at(0) = e;

    //
    // Verify equivalence
    //

    // compare
    bool passed = reduced_tensor_reference[0] == reduced_tensor_computed[0];

    EXPECT_TRUE(passed) 
    << "Expected = " << float(reduced_tensor_reference.at(0)) << "\n\n"
    << "Actual   = " << float(reduced_tensor_computed.at(0)) << "\n\n"
    << std::endl;
    
    return passed;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Thread-level reduction kernel
template <typename Element, int N>
__global__ void kernel_reduce(Element const *array_in, Element *result) {

  /// Thread-level reduction operator
  using Reduce = cutlass::reduction::thread::Reduce<
    cutlass::plus<Element>,
    cutlass::Array<Element, N>
  >;

  Reduce reduce;

  auto ptr_in = reinterpret_cast<cutlass::Array<Element , N> const *>(array_in);
  auto result_ptr = reinterpret_cast<cutlass::Array<Element , 1> *>(result);
  auto in = *ptr_in;
  result_ptr[0] = reduce(in);
}


/// Structure to compute the reduction
template <
  /// Data type of elements
  typename Element,
  /// Number of elements
  int N
>
struct Testbed_reduce_device {

  using Layout = cutlass::layout::PackedVectorLayout;

  //
  // Data members
  //

  cutlass::HostTensor<Element, Layout> tensor_in;
  cutlass::HostTensor<Element, Layout> reduced_tensor_computed;
  cutlass::HostTensor<Element, Layout> reduced_tensor_reference;

  //
  // Methods
  //

  /// Allocates workspace in device memory
  Testbed_reduce_device() {

    tensor_in.reset(cutlass::make_Coord(N), true);
    reduced_tensor_computed.reset(cutlass::make_Coord(1), true);
    reduced_tensor_reference.reset(cutlass::make_Coord(1), true);
  }


  /// Runs the test
  bool run() {

    //
    // initialize memory
    //

    cutlass::reference::host::TensorFill(
      tensor_in.host_view(),
      Element(1)
    );

    cutlass::reference::host::TensorFill(
      reduced_tensor_computed.host_view(),
      Element(0)
    );

    cutlass::reference::host::TensorFill(
      reduced_tensor_reference.host_view(),
      Element(N)
    );

    tensor_in.sync_device();
    reduced_tensor_computed.sync_device();
    reduced_tensor_reference.sync_device();

    /// call the kernel
    kernel_reduce<Element, N><<< dim3(1, 1), dim3(1, 1, 1) >>> (
        tensor_in.device_data(), 
        reduced_tensor_computed.device_data()
        );
    
    // verify no errors
    cudaError_t result = cudaDeviceSynchronize();

    EXPECT_EQ(result, cudaSuccess) << "CUDA ERROR: " << cudaGetErrorString(result);
    if (result != cudaSuccess) {
      return false;
    }

    // Copy back results
    reduced_tensor_computed.sync_host();

    // Verify equivalence
    bool passed = cutlass::reference::host::TensorEquals(
      reduced_tensor_computed.host_view(),
      reduced_tensor_reference.host_view()
    );

    EXPECT_TRUE(passed) 
    << "Expected = " << reduced_tensor_reference.host_view() << "\n\n"
    << "Actual   = " << reduced_tensor_computed.host_view() << "\n\n"
    << std::endl;
    
    return passed;
  }
};

} // namespace thread
} // namespace reduction
} // namespace test
