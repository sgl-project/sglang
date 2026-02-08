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

#include <fstream>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/half.h"

#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"

#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TileIterator>
__global__ void kernel_store_iterator(
  typename TileIterator::Params params,
  typename TileIterator::TensorRef ref, 
  cutlass::MatrixCoord extent) {

  TileIterator iterator(params, ref.data(), extent, threadIdx.x, {0, 0});

  typename TileIterator::Fragment fragment;

  CUTLASS_PRAGMA_NO_UNROLL
  for (int iter = 0; iter < TileIterator::ThreadMap::Count::kTile; ++iter) {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < TileIterator::Fragment::kElements; ++i) {
      typename TileIterator::Element tidx(iter + 1);
      fragment[i] = tidx;
    }

    iterator.store(fragment);
    
    ++iterator;  
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

}
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename Layout>
static bool verify_footprint(cutlass::TensorView<T, Layout> view, cutlass::MatrixCoord extent) {

  for (int r = 0; r < view.extent().row(); ++r) {
    for (int c = 0; c < view.extent().column(); ++c) {

      cutlass::MatrixCoord coord{r, c};
      bool within = coord < extent;
      if (within) {
        if (view.at(coord) == T(0)) {
          return false;
        }
      }
      else {
        if (view.at(coord) != T(0)) {
          return false;
        }
      }
    }
  }

  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(PredicatedTileIterator, tensor_op_64x64x32_64x64x8) {

  using Layout = cutlass::layout::RowMajor;
  using Element = int;

  static int const kElementsPerAccess = 128 / cutlass::sizeof_bits<Element>::value;
  static int const kThreads = 32;

  //
  // The following tests were used to develop the OutputTileOptimalThreadMap
  // metaprogram. The definitions in the disabled blocks of code in this and
  // the following tests are hand-written quantities. They are expected to
  // match what is defined in the ThreadMap.
  //

  #if 1
  using ThreadMap = cutlass::epilogue::threadblock::OutputTileOptimalThreadMap <
    cutlass::epilogue::threadblock::OutputTileShape<64, 8, 1, 1, 1>,
    cutlass::epilogue::threadblock::OutputTileShape<1, 8, 1, 1, 8>,
    kThreads,
    kElementsPerAccess,
    cutlass::sizeof_bits<Element>::value
  >;

  #else
  using InternalThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<64, 64>,
    kThreads,
    kElementsPerAccess
  >;

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<
    64,   // column
    8,    // row
    1,    // group
    1,    // cluster
    1     // iterations
  >;

  using Iterations = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    4,    // row
    1,    // group
    1,    // cluster
    1     // iterations
  >;

  using Delta = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    2,    // row
    1,    // group
    1,    // cluster
    1     // iterations
  >;

  using Count = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    8,    // row
    1,    // group
    1,    // cluster
    8     // iterations
  >;

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadMap<
    InternalThreadMap,
    Shape,
    Iterations,
    Delta,
    Count
  >;
  #endif

  using PredicatedTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    ThreadMap,
    Element
  >;

  //
  // Initialize workspace
  //
  cutlass::MatrixCoord tensor_extent{64, 64};
  cutlass::MatrixCoord output_extent{62, 56};

  //
  // Configure parameters
  //

  cutlass::HostTensor<Element, Layout> host_tensor(tensor_extent);

  typename PredicatedTileIterator::Params iterator_params(host_tensor.layout());

  host_tensor.sync_device();

  //
  // Launch kernel
  //

  dim3 grid(1,1);
  dim3 block(kThreads, 1);

  test::epilogue::threadblock::kernel_store_iterator<PredicatedTileIterator><<< grid, block >>>(
    iterator_params, host_tensor.device_ref(), output_extent);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);

  //
  // Verify results
  //

  host_tensor.sync_host();
  
  bool passed = verify_footprint(host_tensor.host_view(), output_extent);
  EXPECT_TRUE(passed);

  if (!passed) {
    std::ofstream output("tensor_op_64x64x32_64x64x8.csv");
    output << host_tensor.host_view();
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(PredicatedTileIterator, tensor_op_128x64x32_64x64x8) {

  using Layout = cutlass::layout::RowMajor;
  using Element = int;

  static int const kElementsPerAccess = 128 / cutlass::sizeof_bits<Element>::value;
  static int const kThreads = 64;

  #if 1

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileOptimalThreadMap <
    cutlass::epilogue::threadblock::OutputTileShape<128, 8, 2, 1, 1>,
    cutlass::epilogue::threadblock::OutputTileShape<1, 8, 2, 1, 8>,
    kThreads,
    kElementsPerAccess,
    cutlass::sizeof_bits<Element>::value
  >;

  #else
  using InternalThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<64, 128>,
    kThreads,
    kElementsPerAccess
  >;

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<
    64,   // column
    8,    // row
    2,    // group
    1,    // cluster
    8     // tile
  >;

  using Iterations = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    2,    // row
    2,    // group
    1,    // cluster
    1     // iterations
  >;

  using Delta = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    4,    // row
    64,    // group
    1,    // cluster
    1     // tile
  >;

  using Count = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    8,    // row
    1,    // group
    1,    // cluster
    8     // iterations
  >;

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadMap<
    InternalThreadMap,
    Shape,
    Iterations,
    Delta,
    Count
  >;
  #endif

  using PredicatedTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    ThreadMap,
    Element
  >;

  //
  // Initialize workspace
  //
  cutlass::MatrixCoord tensor_extent{128, 64};
  cutlass::MatrixCoord output_extent{125, 56};

  //
  // Configure parameters
  //

  cutlass::HostTensor<Element, Layout> host_tensor(tensor_extent);

  typename PredicatedTileIterator::Params iterator_params(host_tensor.layout());

  host_tensor.sync_device();

  //
  // Launch kernel
  //

  dim3 grid(1,1);
  dim3 block(kThreads, 1);

  test::epilogue::threadblock::kernel_store_iterator<PredicatedTileIterator><<< grid, block >>>(
    iterator_params, host_tensor.device_ref(), output_extent);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);

  //
  // Verify results
  //

  host_tensor.sync_host();
  
  bool passed = verify_footprint(host_tensor.host_view(), output_extent);
  EXPECT_TRUE(passed);

  if (!passed) {
    std::ofstream output("tensor_op_128x64x32_64x64x8.csv");
    output << host_tensor.host_view();
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(PredicatedTileIterator, tensor_op_128x256x32_64x64x8) {

  using Layout = cutlass::layout::RowMajor;
  using Element = int;

  static int const kElementsPerAccess = 128 / cutlass::sizeof_bits<Element>::value;
  static int const kThreads = 256;

  #if 1

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileOptimalThreadMap <
    cutlass::epilogue::threadblock::OutputTileShape<256, 8, 2, 1, 1>,
    cutlass::epilogue::threadblock::OutputTileShape<1, 8, 2, 1, 8>,
    kThreads,
    kElementsPerAccess,
    cutlass::sizeof_bits<Element>::value
  >;

  #else
  using InternalThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<256, 128>,
    kThreads,
    kElementsPerAccess
  >;

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<
    256,   // column
    8,    // row
    2,    // group
    1,    // cluster
    8     // tile
  >;

  using Iterations = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    2,    // row
    2,    // group
    1,    // cluster
    1     // iterations
  >;

  using Delta = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    4,    // row
    64,    // group
    1,    // cluster
    1     // tile
  >;

  using Count = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    8,    // row
    1,    // group
    1,    // cluster
    8     // iterations
  >;

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadMap<
    InternalThreadMap,
    Shape,
    Iterations,
    Delta,
    Count
  >;
  #endif

  using PredicatedTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    ThreadMap,
    Element
  >;

  //
  // Initialize workspace
  //
  cutlass::MatrixCoord tensor_extent{128, 256};
  cutlass::MatrixCoord output_extent{123, 252};

  //
  // Configure parameters
  //

  cutlass::HostTensor<Element, Layout> host_tensor(tensor_extent);

  typename PredicatedTileIterator::Params iterator_params(host_tensor.layout());

  host_tensor.sync_device();

  //
  // Launch kernel
  //

  dim3 grid(1,1);
  dim3 block(kThreads, 1);

  test::epilogue::threadblock::kernel_store_iterator<PredicatedTileIterator><<< grid, block >>>(
    iterator_params, host_tensor.device_ref(), output_extent);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);

  //
  // Verify results
  //

  host_tensor.sync_host();
  
  bool passed = verify_footprint(host_tensor.host_view(), output_extent);
  EXPECT_TRUE(passed);

  if (!passed) {
    std::ofstream output("tensor_op_128x256x32_64x64x8.csv");
    output << host_tensor.host_view();
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(PredicatedTileIterator, volta_tensor_op_64x64x32_64x64x4) {

  using Layout = cutlass::layout::RowMajor;
  using Element = int;

  static int const kElementsPerAccess = 128 / cutlass::sizeof_bits<Element>::value;
  static int const kThreads = 32;

#if 1

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileOptimalThreadMap <
    cutlass::epilogue::threadblock::OutputTileShape<64, 2, 4, 1, 1>,
    cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 8>,
    kThreads,
    kElementsPerAccess,
    cutlass::sizeof_bits<Element>::value
  >;

#else
  using InternalThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<64, 8>,
    kThreads,
    kElementsPerAccess
  >;

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<
    64,   // column
    2,    // row
    4,    // group
    1,    // cluster
    8     // iterations
  >;

  using Iterations = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    1,    // row
    4,    // group
    1,    // cluster
    1     // iterations
  >;

  using Delta = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    1,    // row
    8,    // group
    1,    // cluster
    1     // iterations
  >;

  using Count = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    4,    // row
    2,    // group
    1,    // cluster
    8     // iterations
  >;

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadMap<
    InternalThreadMap,
    Shape,
    Iterations,
    Delta,
    Count
  >;
#endif

  using PredicatedTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    ThreadMap,
    Element
  >;

  //
  // Initialize workspace
  //
  cutlass::MatrixCoord tensor_extent{64, 64};
  cutlass::MatrixCoord output_extent{62, 56};

  //
  // Configure parameters
  //

  cutlass::HostTensor<Element, Layout> host_tensor(tensor_extent);

  typename PredicatedTileIterator::Params iterator_params(host_tensor.layout());

  host_tensor.sync_device();

  //
  // Launch kernel
  //

  dim3 grid(1,1);
  dim3 block(kThreads, 1);

  test::epilogue::threadblock::kernel_store_iterator<PredicatedTileIterator><<< grid, block >>>(
    iterator_params, host_tensor.device_ref(), output_extent);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);

  //
  // Verify results
  //
  host_tensor.sync_host();
  
  bool passed = verify_footprint(host_tensor.host_view(), output_extent);
  EXPECT_TRUE(passed);

  if (!passed) {
    std::ofstream output("volta_tensor_op_64x64x32_64x64x4.csv");
    output << host_tensor.host_view();
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(PredicatedTileIterator, volta_tensor_op_64x128x32_32x64x4) {

  using Layout = cutlass::layout::RowMajor;
  using Element = int;

  static int const kElementsPerAccess = 128 / cutlass::sizeof_bits<Element>::value;
  static int const kThreads = 128;

#if 1

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileOptimalThreadMap <
    cutlass::epilogue::threadblock::OutputTileShape<128, 2, 4, 1, 1>,
    cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 8>,
    kThreads,
    kElementsPerAccess,
    cutlass::sizeof_bits<Element>::value
  >;

#else
  using InternalThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<128, 8>,
    kThreads,
    kElementsPerAccess
  >;

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<
    128,   // column
    2,    // row
    2,    // group
    2,    // cluster
    8     // iterations
  >;

  using Iterations = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    1,    // row
    1,    // group
    2,    // cluster
    1     // iterations
  >;

  using Delta = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    1,    // row
    8,    // group
    32,    // cluster
    1     // iterations
  >;

  using Count = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    4,    // row
    4,    // group
    1,    // cluster
    8     // iterations
  >;

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadMap<
    InternalThreadMap,
    Shape,
    Iterations,
    Delta,
    Count
  >;
#endif

  using PredicatedTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    ThreadMap,
    Element
  >;

  //
  // Initialize workspace
  //
  cutlass::MatrixCoord tensor_extent{64, 128};
  cutlass::MatrixCoord output_extent{57, 124};

  //
  // Configure parameters
  //

  cutlass::HostTensor<Element, Layout> host_tensor(tensor_extent);

  typename PredicatedTileIterator::Params iterator_params(host_tensor.layout());

  host_tensor.sync_device();

  //
  // Launch kernel
  //

  dim3 grid(1,1);
  dim3 block(kThreads, 1);

  test::epilogue::threadblock::kernel_store_iterator<PredicatedTileIterator><<< grid, block >>>(
    iterator_params, host_tensor.device_ref(), output_extent);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);

  //
  // Verify results
  //

  host_tensor.sync_host();

  bool passed = verify_footprint(host_tensor.host_view(), output_extent);
  EXPECT_TRUE(passed);

  if (!passed) {
    std::ofstream output("volta_tensor_op_64x128x32_32x64x4.csv");
    output << host_tensor.host_view();
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(PredicatedTileIterator, volta_tensor_op_128x256x32_64x64x4) {

  using Layout = cutlass::layout::RowMajor;
  using Element = int;

  static int const kElementsPerAccess = 128 / cutlass::sizeof_bits<Element>::value;
  static int const kThreads = 256;

#if 1

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileOptimalThreadMap <
    cutlass::epilogue::threadblock::OutputTileShape<256, 2, 4, 2, 1>,
    cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 8>,
    kThreads,
    kElementsPerAccess,
    cutlass::sizeof_bits<Element>::value
  >;

#else
  using InternalThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<256, 16>,
    kThreads,
    kElementsPerAccess
  >;

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<
    256,   // column
    2,    // row
    4,    // group
    2,    // cluster
    8     // iterations
  >;

  using Iterations = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    1,    // row
    2,    // group
    2,    // cluster
    1     // iterations
  >;

  using Delta = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    1,    // row
    16,    // group
    64,    // cluster
    1     // iterations
  >;

  using Count = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    4,    // row
    2,    // group
    1,    // cluster
    8     // iterations
  >;

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadMap<
    InternalThreadMap,
    Shape,
    Iterations,
    Delta,
    Count
  >;
#endif

  using PredicatedTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    ThreadMap,
    Element
  >;

  //
  // Initialize workspace
  //
  cutlass::MatrixCoord tensor_extent{128, 256};
  cutlass::MatrixCoord output_extent{128, 256};

  //
  // Configure parameters
  //

  cutlass::HostTensor<Element, Layout> host_tensor(tensor_extent);

  typename PredicatedTileIterator::Params iterator_params(host_tensor.layout());

  host_tensor.sync_device();

  //
  // Launch kernel
  //

  dim3 grid(1,1);
  dim3 block(kThreads, 1);

  test::epilogue::threadblock::kernel_store_iterator<PredicatedTileIterator><<< grid, block >>>(
    iterator_params, host_tensor.device_ref(), output_extent);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);

  //
  // Verify results
  //

  host_tensor.sync_host();

  bool passed = verify_footprint(host_tensor.host_view(), output_extent);
  EXPECT_TRUE(passed);

  if (!passed || true) {
    std::ofstream output("volta_tensor_op_128x256x32_64x64x4.csv");
    output << host_tensor.host_view();
  }
}


TEST(PredicatedTileIterator, volta_tensor_op_256x128x32_64x64x4) {

  using Layout = cutlass::layout::RowMajor;
  using Element = int;

  static int const kElementsPerAccess = 128 / cutlass::sizeof_bits<Element>::value;
  static int const kThreads = 256;

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileOptimalThreadMap <
    cutlass::epilogue::threadblock::OutputTileShape<128, 2, 4, 4, 1>,
    cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 8>,
    kThreads,
    kElementsPerAccess,
    cutlass::sizeof_bits<Element>::value
  >;

  using PredicatedTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    ThreadMap,
    Element
  >;

  //
  // Initialize workspace
  //
  cutlass::MatrixCoord tensor_extent{ 256, 128 };
  cutlass::MatrixCoord output_extent{ 256, 128 };

  //
  // Configure parameters
  //

  cutlass::HostTensor<Element, Layout> host_tensor(tensor_extent);

  typename PredicatedTileIterator::Params iterator_params(host_tensor.layout());

  host_tensor.sync_device();

  //
  // Launch kernel
  //

  dim3 grid(1, 1);
  dim3 block(kThreads, 1);

  test::epilogue::threadblock::kernel_store_iterator<PredicatedTileIterator> <<< grid, block >>>(
    iterator_params, host_tensor.device_ref(), output_extent);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);

  //
  // Verify results
  //

  host_tensor.sync_host();

  bool passed = verify_footprint(host_tensor.host_view(), output_extent);
  EXPECT_TRUE(passed);

  if (!passed || true) {
    std::ofstream output("volta_tensor_op_256x128x32_64x64x4.csv");
    output << host_tensor.host_view();
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(PredicatedTileIterator, simt_32x64x8_32x64x1) {

  using Layout = cutlass::layout::RowMajor;
  using Element = int;

  static int const kElementsPerAccess = 32 / cutlass::sizeof_bits<Element>::value;
  static int const kThreads = 32;

#if 1

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileOptimalThreadMap <
    cutlass::epilogue::threadblock::OutputTileShape<64, 1, 4, 1, 1>,
    cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 8>,
    kThreads,
    kElementsPerAccess,
    cutlass::sizeof_bits<Element>::value
  >;

#else
  using InternalThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<64, 4>,
    kThreads,
    kElementsPerAccess
  >;

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<
    64,   // column
    1,    // row
    4,    // group
    1,    // cluster
    1     // iterations
  >;

  using Iterations = cutlass::epilogue::threadblock::OutputTileShape<
    2,    // column
    1,    // row
    4,    // group
    1,    // cluster
    1     // iterations
  >;

  using Delta = cutlass::epilogue::threadblock::OutputTileShape<
    32,    // column
    1,    // row
    4,    // group
    16,    // cluster
    1     // iterations
  >;

  using Count = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    4,    // row
    2,    // group
    1,    // cluster
    8     // iterations
  >;

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadMap<
    InternalThreadMap,
    Shape,
    Iterations,
    Delta,
    Count
  >;
#endif

  using PredicatedTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    ThreadMap,
    Element
  >;

  //
  // Initialize workspace
  //
  cutlass::MatrixCoord tensor_extent{32, 64};
  cutlass::MatrixCoord output_extent{27, 63};

  //
  // Configure parameters
  //

  cutlass::HostTensor<Element, Layout> host_tensor(tensor_extent);

  typename PredicatedTileIterator::Params iterator_params(host_tensor.layout());

  host_tensor.sync_device();

  //
  // Launch kernel
  //

  dim3 grid(1,1);
  dim3 block(kThreads, 1);

  test::epilogue::threadblock::kernel_store_iterator<PredicatedTileIterator><<< grid, block >>>(
    iterator_params, host_tensor.device_ref(), output_extent);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);

  //
  // Verify results
  //

  host_tensor.sync_host();

  bool passed = verify_footprint(host_tensor.host_view(), output_extent);
  EXPECT_TRUE(passed);

  if (!passed) {
    std::ofstream output("simt_32x64x8_32x64x1.csv");
    output << host_tensor.host_view(); 
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(PredicatedTileIterator, simt_128x128x8_32x64x1) {

  using Layout = cutlass::layout::RowMajor;
  using Element = int;

  static int const kElementsPerAccess = 32 / cutlass::sizeof_bits<Element>::value;
  static int const kThreads = 256;

#if 1

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileOptimalThreadMap <
    cutlass::epilogue::threadblock::OutputTileShape<128, 1, 4, 4, 1>,
    cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 8>,
    kThreads,
    kElementsPerAccess,
    cutlass::sizeof_bits<Element>::value
  >;

#else
  using InternalThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<128, 16>,
    kThreads,
    kElementsPerAccess
  >;

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<
    128,   // column
    1,    // row
    4,    // group
    4,    // cluster
    1     // iterations
  >;

  using Iterations = cutlass::epilogue::threadblock::OutputTileShape<
    2,    // column
    1,    // row
    2,    // group
    4,    // cluster
    1     // iterations
  >;

  using Delta = cutlass::epilogue::threadblock::OutputTileShape<
    32,    // column
    1,    // row
    8,    // group
    32,    // cluster
    1     // iterations
  >;

  using Count = cutlass::epilogue::threadblock::OutputTileShape<
    1,    // column
    4,    // row
    2,    // group
    1,    // cluster
    8     // iterations
  >;

  using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadMap<
    InternalThreadMap,
    Shape,
    Iterations,
    Delta,
    Count
  >;
#endif

  using PredicatedTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    ThreadMap,
    Element
  >;

  //
  // Initialize workspace
  //
  cutlass::MatrixCoord tensor_extent{128, 128};
  cutlass::MatrixCoord output_extent{123, 121};

  //
  // Configure parameters
  //

  cutlass::HostTensor<Element, Layout> host_tensor(tensor_extent);

  typename PredicatedTileIterator::Params iterator_params(host_tensor.layout());

  host_tensor.sync_device();

  //
  // Launch kernel
  //

  dim3 grid(1,1);
  dim3 block(kThreads, 1);

  test::epilogue::threadblock::kernel_store_iterator<PredicatedTileIterator><<< grid, block >>>(
    iterator_params, host_tensor.device_ref(), output_extent);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);

  //
  // Verify results
  //

  host_tensor.sync_host();

  bool passed = verify_footprint(host_tensor.host_view(), output_extent);
  EXPECT_TRUE(passed);

  if (!passed) {
    std::ofstream output("simt_128x128x8_32x64x1.csv");
    output << host_tensor.host_view(); 
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
