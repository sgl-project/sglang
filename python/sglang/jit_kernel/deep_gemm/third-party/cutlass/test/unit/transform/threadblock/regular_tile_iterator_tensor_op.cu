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
    \brief 
*/

#include "../../common/cutlass_unit_test.h"

#include "cutlass/cutlass.h"
#include "cutlass/core_io.h"
#include "cutlass/layout/pitch_linear.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace gemm {
namespace threadblock {

/// 
template <typename Iterator>
__global__ void kernel_gemm_threadblock_tensor_op_multiplicand_store(
  typename Iterator::TensorRef ref_output,
  typename Iterator::Element *input) {

  // Construct fragment
  typename Iterator::Fragment frag;

  frag.clear();

  // each thread loads a fragment
  using AccessType = cutlass::Array<typename Iterator::Element, Iterator::ThreadMap::kElementsPerAccess>;

  int const kElementsPerAccess = Iterator::ThreadMap::kElementsPerAccess;
  int stride = Iterator::Shape::kContiguous;

  int warp_id = (threadIdx.x / 32);
  int lane_id = (threadIdx.x % 32);

  input += (lane_id % 8) * kElementsPerAccess + (lane_id / 8) * stride;

  input += (warp_id * Iterator::Shape::kStrided / Iterator::ThreadMap::Detail::kWarpCount) * stride;

  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < Iterator::ThreadMap::Iterations::kStrided; ++s) {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < Iterator::ThreadMap::Iterations::kContiguous; ++c) {
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < Iterator::ThreadMap::kElementsPerAccess; ++v) {
        frag[v + Iterator::ThreadMap::kElementsPerAccess * (c + s * Iterator::ThreadMap::Iterations::kContiguous)] = 
          input[v + c * 64 + s * Iterator::ThreadMap::Delta::kStrided * stride];
      }
    }
  }

  // Use iterator to store results
  Iterator iter(ref_output, threadIdx.x);
  iter.store(frag);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Simple test environment
template <
  typename Shape_,
  int WarpCount
>
class MultiplicandTileIteratorTestbed {
public:

  //
  // Define iterator
  //

  using Shape = Shape_;
  using Element = cutlass::half_t;
  using Layout = cutlass::layout::TensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  static int const kAdvanceRank = 1;
  static int const kThreads = 32 * WarpCount;

  using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
    Shape,
    kThreads,
    cutlass::layout::PitchLinearShape<8, 4>,
    128 / cutlass::sizeof_bits<Element>::value
  >;

  using Iterator = cutlass::transform::threadblock::RegularTileIterator<
    Shape, Element, Layout, kAdvanceRank, ThreadMap
  >;

public:

  //
  // Members
  //

  cutlass::HostTensor<Element, Layout> destination_tensor;
  cutlass::HostTensor<Element, cutlass::layout::PitchLinear> source_tensor;
  

public:

  MultiplicandTileIteratorTestbed(): 
    destination_tensor({Shape::kContiguous, Shape::kStrided}),
    source_tensor({Shape::kContiguous, Shape::kStrided}) {

  }

  bool run() {

    cutlass::reference::host::BlockFillSequential(
      source_tensor.host_data(),
      source_tensor.capacity()
    );

    cutlass::reference::host::BlockFillSequential(
      destination_tensor.host_data(),
      destination_tensor.capacity(),
      Element(0),
      Element(0)
    );

    //
    // Launch kernel
    //

    dim3 grid(1,1);
    dim3 block(kThreads, 1);

    destination_tensor.sync_device();
    source_tensor.sync_device();

    test::gemm::threadblock::kernel_gemm_threadblock_tensor_op_multiplicand_store<Iterator><<<
      grid, block
    >>>(
      destination_tensor.device_ref(),
      source_tensor.device_data()
    );

    cudaError_t result = cudaDeviceSynchronize();
    EXPECT_EQ(result, cudaSuccess) << " - CUDA ERROR: " << cudaGetErrorString(result);

    destination_tensor.sync_host();

    //
    // Verify
    //

    // Verify that its contents match the destination
    int errors = 0;
    for (int s = 0; s < Shape::kStrided; ++s) {
      for (int c = 0; c < Shape::kContiguous; ++c) {

        if (errors >= 10) {
          break;
        }

        Element expected = source_tensor.at({c, s});
        Element got = destination_tensor.at({c, s});

        bool passed = (expected == got);
        if (!passed) {
          ++errors;
        }
      }
    }

    EXPECT_EQ(errors, 0)
      << source_tensor.host_view() << "\n\n" << destination_tensor.host_view() << std::endl;

    return !errors;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_gemm_threadblock_tensor_op_multplicand_iterator_congruous_16b, 64x8_w1) {

  test::gemm::threadblock::MultiplicandTileIteratorTestbed<
    cutlass::layout::PitchLinearShape<64, 8>, 1>().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_gemm_threadblock_tensor_op_multplicand_iterator_congruous_16b, 64x16_w1) {

  test::gemm::threadblock::MultiplicandTileIteratorTestbed<
    cutlass::layout::PitchLinearShape<64, 16>, 1>().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_gemm_threadblock_tensor_op_multplicand_iterator_congruous_16b, 64x16_w2) {

  test::gemm::threadblock::MultiplicandTileIteratorTestbed<
    cutlass::layout::PitchLinearShape<64, 16>, 2>().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_gemm_threadblock_tensor_op_multplicand_iterator_congruous_16b, 128x8_w1) {

  test::gemm::threadblock::MultiplicandTileIteratorTestbed<
    cutlass::layout::PitchLinearShape<128, 8>, 1>().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_gemm_threadblock_tensor_op_multplicand_iterator_congruous_16b, 64x32_w4) {

  test::gemm::threadblock::MultiplicandTileIteratorTestbed<
    cutlass::layout::PitchLinearShape<64, 32>, 4>().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_gemm_threadblock_tensor_op_multplicand_iterator_congruous_16b, 128x32_w1) {

  test::gemm::threadblock::MultiplicandTileIteratorTestbed<
    cutlass::layout::PitchLinearShape<128, 32>, 1>().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_gemm_threadblock_tensor_op_multplicand_iterator_congruous_16b, 128x32_w4) {

  test::gemm::threadblock::MultiplicandTileIteratorTestbed<
    cutlass::layout::PitchLinearShape<128, 32>, 4>().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_gemm_threadblock_tensor_op_multplicand_iterator_congruous_16b, 256x32_w4) {

  test::gemm::threadblock::MultiplicandTileIteratorTestbed<
    cutlass::layout::PitchLinearShape<256, 32>, 4>().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_gemm_threadblock_tensor_op_multplicand_iterator_congruous_16b, 256x32_w8) {

  test::gemm::threadblock::MultiplicandTileIteratorTestbed<
    cutlass::layout::PitchLinearShape<256, 32>, 8>().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
