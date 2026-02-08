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

/*
  This example demonstrates how to use the PredicatedTileIterator in CUTLASS to load data from
  addressable memory, and then store it back into addressable memory.

  TileIterator is a core concept in CUTLASS that enables efficient loading and storing of data to
  and from addressable memory. The PredicatedTileIterator accepts a ThreadMap type, which defines
  the mapping of threads to a "tile" in memory. This separation of concerns enables user-defined
  thread mappings to be specified. 

  In this example, a PredicatedTileIterator is used to load elements from a tile in global memory,
  stored in column-major layout, into a fragment and then back into global memory in the same
  layout.

  This example uses CUTLASS utilities to ease the matrix operations.

*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// CUTLASS includes
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/transform/pitch_linear_thread_map.h"

//
//  CUTLASS utility includes
//

// Defines operator<<() to write TensorView objects to std::ostream
#include "cutlass/util/tensor_view_io.h"

// Defines cutlass::HostTensor<>
#include "cutlass/util/host_tensor.h"

// Defines cutlass::reference::host::TensorFill() and
// cutlass::reference::host::TensorFillBlockSequential()
#include "cutlass/util/reference/host/tensor_fill.h"

#pragma warning( disable : 4503)
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define PredicatedTileIterators to load and store a M-by-K tile, in column major layout.

template <typename Iterator>
__global__ void copy(
    typename Iterator::Params dst_params,
    typename Iterator::Element *dst_pointer,
    typename Iterator::Params src_params,
    typename Iterator::Element *src_pointer,
    cutlass::Coord<2> extent) {


    Iterator dst_iterator(dst_params, dst_pointer, extent, threadIdx.x);
    Iterator src_iterator(src_params, src_pointer, extent, threadIdx.x);

    // PredicatedTileIterator uses PitchLinear layout and therefore takes in a PitchLinearShape.
    // The contiguous dimension can be accessed via Iterator::Shape::kContiguous and the strided
    // dimension can be accessed via Iterator::Shape::kStrided
    int iterations = (extent[1] + Iterator::Shape::kStrided - 1) / Iterator::Shape::kStrided;

    typename Iterator::Fragment fragment;

    for(size_t i = 0; i < fragment.size(); ++i) {
      fragment[i] = 0;
    }

    src_iterator.load(fragment);
    dst_iterator.store(fragment);


    ++src_iterator;
    ++dst_iterator;

    for(; iterations > 1; --iterations) {

      src_iterator.load(fragment);
      dst_iterator.store(fragment);

      ++src_iterator;
      ++dst_iterator;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Initializes the source tile with sequentially increasing values and performs the copy into
// the destination tile using two PredicatedTileIterators, one to load the data from addressable
// memory into a fragment (regiser-backed array of elements owned by each thread) and another to 
// store the data from the fragment back into the addressable memory of the destination tile.

cudaError_t TestTileIterator(int M, int K) {

    // For this example, we chose a <64, 4> tile shape. The PredicatedTileIterator expects
    // PitchLinearShape and PitchLinear layout.
    using Shape = cutlass::layout::PitchLinearShape<64, 4>;
    using Layout = cutlass::layout::PitchLinear;
    using Element = int;
    int const kThreads = 32;

    // ThreadMaps define how threads are mapped to a given tile. The PitchLinearStripminedThreadMap
    // stripmines a pitch-linear tile among a given number of threads, first along the contiguous
    // dimension then along the strided dimension.
    using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads>;

    // Define the PredicatedTileIterator, using TileShape, Element, Layout, and ThreadMap types
    using Iterator = cutlass::transform::threadblock::PredicatedTileIterator<
        Shape, Element, Layout, 1, ThreadMap>;


    cutlass::Coord<2> copy_extent = cutlass::make_Coord(M, K);
    cutlass::Coord<2> alloc_extent = cutlass::make_Coord(M, K);

    // Allocate source and destination tensors
    cutlass::HostTensor<Element, Layout> src_tensor(alloc_extent);
    cutlass::HostTensor<Element, Layout> dst_tensor(alloc_extent);

    Element oob_value = Element(-1);

    // Initialize destination tensor with all -1s
    cutlass::reference::host::TensorFill(dst_tensor.host_view(), oob_value);
    // Initialize source tensor with sequentially increasing values
    cutlass::reference::host::BlockFillSequential(src_tensor.host_data(), src_tensor.capacity());

    dst_tensor.sync_device();
    src_tensor.sync_device();

    typename Iterator::Params dst_params(dst_tensor.layout());
    typename Iterator::Params src_params(src_tensor.layout());

    dim3 block(kThreads, 1);
    dim3 grid(1, 1);

    // Launch copy kernel to perform the copy
    copy<Iterator><<< grid, block >>>(
            dst_params,
            dst_tensor.device_data(),
            src_params,
            src_tensor.device_data(),
            copy_extent
    );

    cudaError_t result = cudaGetLastError();
    if(result != cudaSuccess) {
      std::cerr << "Error - kernel failed." << std::endl;
      return result;
    }

    dst_tensor.sync_host();

    // Verify results
    for(int s = 0; s < alloc_extent[1]; ++s) {
      for(int c = 0; c < alloc_extent[0]; ++c) {

          Element expected = Element(0);

          if(c < copy_extent[0] && s < copy_extent[1]) {
            expected = src_tensor.at({c, s});
          }
          else {
            expected = oob_value;
          }

          Element got = dst_tensor.at({c, s});
          bool equal = (expected == got);

          if(!equal) {
              std::cerr << "Error - source tile differs from destination tile." << std::endl;
            return cudaErrorUnknown;
          }
      }
    }

    return cudaSuccess;
}

int main(int argc, const char *arg[]) {

    cudaError_t result = TestTileIterator(57, 35);

    if(result == cudaSuccess) {
      std::cout << "Passed." << std::endl;  
    }

    // Exit
    return result == cudaSuccess ? 0 : -1;
}

