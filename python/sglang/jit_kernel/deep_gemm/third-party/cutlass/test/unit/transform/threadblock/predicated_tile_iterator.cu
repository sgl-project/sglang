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
    \brief Tests cutlass::transform::threadblock::PredicatedTileIterator 
*/

#include "../../common/cutlass_unit_test.h"

#include "cutlass/cutlass.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"

#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace transform {
namespace threadblock {
namespace kernel {

/// Copy with an iterator
template <typename Iterator>
__global__ void copy(
  typename Iterator::Params dst_params, 
  typename Iterator::Element *dst_pointer,
  typename Iterator::Params src_params,
  typename Iterator::Element *src_pointer,
  cutlass::Coord<2> extent) {

  Iterator dst_iterator(dst_params, dst_pointer, extent, threadIdx.x);
  Iterator src_iterator(src_params, src_pointer, extent, threadIdx.x);

  int iterations = (extent[1] + Iterator::Shape::kStrided - 1) / Iterator::Shape::kStrided;

  typename Iterator::Fragment frag;

  for(size_t i = 0; i < frag.size(); i++)
    frag[i] = 0;

  src_iterator.load(frag);
  dst_iterator.store(frag);

  ++dst_iterator;
  ++src_iterator;

  for (; iterations > 1; --iterations) {
    
    src_iterator.load(frag);
    dst_iterator.store(frag);

    ++dst_iterator;
    ++src_iterator;
  }
}

} // namespace kernel
} // namespace threadblock
} // namespace transform
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Transform_threadblock_PredicatedTileIterator, PitchLinear_Stripmined) {

  using Shape = cutlass::layout::PitchLinearShape<64, 4>;
  using Layout = cutlass::layout::PitchLinear;
  using Element = int;
  static int const kThreads = 32;
  
  using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads>;

  using Iterator = cutlass::transform::threadblock::PredicatedTileIterator<
    Shape, Element, Layout, 1, ThreadMap
  >;

  cutlass::Coord<2> copy_extent = cutlass::make_Coord(57, 35);
  cutlass::Coord<2> alloc_extent = cutlass::make_Coord(64, 35);
  
  cutlass::HostTensor<int, Layout> src_tensor(alloc_extent);
  cutlass::HostTensor<int, Layout> dst_tensor(alloc_extent);

  Element oob_value = Element(-1);
  cutlass::reference::host::TensorFill(dst_tensor.host_view(), oob_value);
  cutlass::reference::host::BlockFillSequential(src_tensor.host_data(), src_tensor.capacity());

  dst_tensor.sync_device();
  src_tensor.sync_device();

  typename Iterator::Params dst_params(dst_tensor.layout());
  typename Iterator::Params src_params(src_tensor.layout());

  dim3 block(kThreads, 1);
  dim3 grid(1, 1);

  test::transform::threadblock::kernel::copy<Iterator><<< grid, block >>>(
    dst_params,
    dst_tensor.device_data(),
    src_params,
    src_tensor.device_data(),
    copy_extent
  );

  cudaError_t result = cudaGetLastError();
  EXPECT_EQ(result, cudaSuccess) << " - CUDA error: " << cudaGetErrorString(result);
  
  dst_tensor.sync_host();

  for (int s = 0; s < alloc_extent[1]; ++s) {
    for (int c = 0; c < alloc_extent[0]; ++c) {

      Element expected = Element(0);

      if (c < copy_extent[0] && s < copy_extent[1]) {
        expected = src_tensor.at({c, s});
      }
      else {
        expected = oob_value;
      }

      Element got = dst_tensor.at({c, s});
      bool equal = (expected == got);

      EXPECT_EQ(expected, got)
        << "Source:\n" << src_tensor.host_view() << "\n\n"
        << "Destination:\n" << dst_tensor.host_view() << "\n";

      if (!equal) {
        return;
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
TEST(Transform_threadblock_PredicatedTileIterator, PitchLinear_Stripmined_2dtile_128x4) {

  using Shape = cutlass::layout::PitchLinearShape<128, 4>;
  using ThreadTileShape = cutlass::layout::PitchLinearShape<4, 4>;
  using Layout = cutlass::layout::PitchLinear;
  using Element = int8_t;
  static int const kThreads = 32;
  
  using ThreadMap = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<Shape, kThreads, ThreadTileShape>;

  using Iterator = cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
    Shape, Element, Layout, 1, ThreadMap, false
  >;

  cutlass::Coord<2> copy_extent = cutlass::make_Coord(128, 4);
  cutlass::Coord<2> alloc_extent = cutlass::make_Coord(128, 4);
  
  cutlass::HostTensor<int8_t, Layout> src_tensor(alloc_extent);
  cutlass::HostTensor<int8_t, Layout> dst_tensor(alloc_extent);

  Element oob_value = Element(-1);
  cutlass::reference::host::TensorFill(dst_tensor.host_view(), oob_value);
  cutlass::reference::host::BlockFillSequential(src_tensor.host_data(), src_tensor.capacity());

  dst_tensor.sync_device();
  src_tensor.sync_device();

  typename Iterator::Params dst_params(dst_tensor.layout());
  typename Iterator::Params src_params(src_tensor.layout());

  dim3 block(kThreads, 1);
  dim3 grid(1, 1);

  test::transform::threadblock::kernel::copy<Iterator><<< grid, block >>>(
    dst_params,
    dst_tensor.device_data(),
    src_params,
    src_tensor.device_data(),
    copy_extent
  );

  cudaError_t result = cudaGetLastError();
  EXPECT_EQ(result, cudaSuccess) << " - CUDA error: " << cudaGetErrorString(result);
  
  dst_tensor.sync_host();

  for (int s = 0; s < alloc_extent[1]; ++s) {
    for (int c = 0; c < alloc_extent[0]; ++c) {

      Element expected = Element(0);

      if (c < copy_extent[0] && s < copy_extent[1]) {
        expected = src_tensor.at({c, s});
      }
      else {
        expected = oob_value;
      }

      Element got = dst_tensor.at({c, s});
      bool equal = (expected == got);

      EXPECT_EQ(expected, got)
        << "Source:\n" << src_tensor.host_view() << "\n\n"
        << "Destination:\n" << dst_tensor.host_view() << "\n";

      if (!equal) {
        return;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Transform_threadblock_PredicatedTileIterator, PitchLinear_Stripmined_2dtile_128x64) {

  using Shape = cutlass::layout::PitchLinearShape<128, 64>;
  using ThreadTileShape = cutlass::layout::PitchLinearShape<4, 4>;
  using Layout = cutlass::layout::PitchLinear;
  using Element = int8_t;
  static int const kThreads = 32;
  
  using ThreadMap = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<Shape, kThreads, ThreadTileShape>;

  using Iterator = cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
    Shape, Element, Layout, 1, ThreadMap
  >;

  cutlass::Coord<2> copy_extent = cutlass::make_Coord(128, 64);
  cutlass::Coord<2> alloc_extent = cutlass::make_Coord(128, 64);
  
  cutlass::HostTensor<int8_t, Layout> src_tensor(alloc_extent);
  cutlass::HostTensor<int8_t, Layout> dst_tensor(alloc_extent);

  Element oob_value = Element(-1);
  cutlass::reference::host::TensorFill(dst_tensor.host_view(), oob_value);
  cutlass::reference::host::BlockFillSequential(src_tensor.host_data(), src_tensor.capacity());

  dst_tensor.sync_device();
  src_tensor.sync_device();

  typename Iterator::Params dst_params(dst_tensor.layout());
  typename Iterator::Params src_params(src_tensor.layout());

  dim3 block(kThreads, 1);
  dim3 grid(1, 1);

  test::transform::threadblock::kernel::copy<Iterator><<< grid, block >>>(
    dst_params,
    dst_tensor.device_data(),
    src_params,
    src_tensor.device_data(),
    copy_extent
  );

  cudaError_t result = cudaGetLastError();
  EXPECT_EQ(result, cudaSuccess) << " - CUDA error: " << cudaGetErrorString(result);
  
  dst_tensor.sync_host();

  for (int s = 0; s < alloc_extent[1]; ++s) {
    for (int c = 0; c < alloc_extent[0]; ++c) {

      Element expected = Element(0);

      if (c < copy_extent[0] && s < copy_extent[1]) {
        expected = src_tensor.at({c, s});
      }
      else {
        expected = oob_value;
      }

      Element got = dst_tensor.at({c, s});
      bool equal = (expected == got);

      EXPECT_EQ(expected, got)
        << "Source:\n" << src_tensor.host_view() << "\n\n"
        << "Destination:\n" << dst_tensor.host_view() << "\n";

      if (!equal) {
        return;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////

TEST(Transform_threadblock_PredicatedTileIterator, PitchLinear_Stripmined_2dtile_64x64) {

  using Shape = cutlass::layout::PitchLinearShape<64, 64>;
  using ThreadTileShape = cutlass::layout::PitchLinearShape<4, 4>;
  using Layout = cutlass::layout::PitchLinear;
  using Element = int8_t;
  static int const kThreads = 32;
  
  using ThreadMap = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<Shape, kThreads, ThreadTileShape>;

  using Iterator = cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
    Shape, Element, Layout, 1, ThreadMap
  >;

  cutlass::Coord<2> copy_extent = cutlass::make_Coord(64, 64);
  cutlass::Coord<2> alloc_extent = cutlass::make_Coord(64, 64);
  
  cutlass::HostTensor<int8_t, Layout> src_tensor(alloc_extent);
  cutlass::HostTensor<int8_t, Layout> dst_tensor(alloc_extent);

  Element oob_value = Element(-1);
  cutlass::reference::host::TensorFill(dst_tensor.host_view(), oob_value);
  cutlass::reference::host::BlockFillSequential(src_tensor.host_data(), src_tensor.capacity());

  dst_tensor.sync_device();
  src_tensor.sync_device();

  typename Iterator::Params dst_params(dst_tensor.layout());
  typename Iterator::Params src_params(src_tensor.layout());

  dim3 block(kThreads, 1);
  dim3 grid(1, 1);

  test::transform::threadblock::kernel::copy<Iterator><<< grid, block >>>(
    dst_params,
    dst_tensor.device_data(),
    src_params,
    src_tensor.device_data(),
    copy_extent
  );

  cudaError_t result = cudaGetLastError();
  EXPECT_EQ(result, cudaSuccess) << " - CUDA error: " << cudaGetErrorString(result);
  
  dst_tensor.sync_host();

  for (int s = 0; s < alloc_extent[1]; ++s) {
    for (int c = 0; c < alloc_extent[0]; ++c) {

      Element expected = Element(0);

      if (c < copy_extent[0] && s < copy_extent[1]) {
        expected = src_tensor.at({c, s});
      }
      else {
        expected = oob_value;
      }

      Element got = dst_tensor.at({c, s});
      bool equal = (expected == got);

      EXPECT_EQ(expected, got)
        << "Source:\n" << src_tensor.host_view() << "\n\n"
        << "Destination:\n" << dst_tensor.host_view() << "\n";

      if (!equal) {
        return;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Transform_threadblock_PredicatedTileIterator, PitchLinear_Stripmined_2dtile_64x8) {

  using Shape = cutlass::layout::PitchLinearShape<64, 8>;
  using ThreadTileShape = cutlass::layout::PitchLinearShape<4, 4>;
  using Layout = cutlass::layout::PitchLinear;
  using Element = int8_t;
  static int const kThreads = 32;
  
  using ThreadMap = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<Shape, kThreads, ThreadTileShape>;

  using Iterator = cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
    Shape, Element, Layout, 1, ThreadMap
  >;

  cutlass::Coord<2> copy_extent = cutlass::make_Coord(32, 8);
  cutlass::Coord<2> alloc_extent = cutlass::make_Coord(64, 8);
  
  cutlass::HostTensor<int8_t, Layout> src_tensor(alloc_extent);
  cutlass::HostTensor<int8_t, Layout> dst_tensor(alloc_extent);

  Element oob_value = Element(-1);
  cutlass::reference::host::TensorFill(dst_tensor.host_view(), oob_value);
  cutlass::reference::host::BlockFillSequential(src_tensor.host_data(), src_tensor.capacity());

  dst_tensor.sync_device();
  src_tensor.sync_device();

  typename Iterator::Params dst_params(dst_tensor.layout());
  typename Iterator::Params src_params(src_tensor.layout());

  dim3 block(kThreads, 1);
  dim3 grid(1, 1);

  test::transform::threadblock::kernel::copy<Iterator><<< grid, block >>>(
    dst_params,
    dst_tensor.device_data(),
    src_params,
    src_tensor.device_data(),
    copy_extent
  );

  cudaError_t result = cudaGetLastError();
  EXPECT_EQ(result, cudaSuccess) << " - CUDA error: " << cudaGetErrorString(result);
  
  dst_tensor.sync_host();

  for (int s = 0; s < alloc_extent[1]; ++s) {
    for (int c = 0; c < alloc_extent[0]; ++c) {

      Element expected = Element(0);

      if (c < copy_extent[0] && s < copy_extent[1]) {
        expected = src_tensor.at({c, s});
      }
      else {
        expected = oob_value;
      }

      Element got = dst_tensor.at({c, s});
      bool equal = (expected == got);

      EXPECT_EQ(expected, got)
        << "Source:\n" << src_tensor.host_view() << "\n\n"
        << "Destination:\n" << dst_tensor.host_view() << "\n";

      if (!equal) {
        return;
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Transform_threadblock_PredicatedTileIterator, PitchLinear_Stripmined_2dtile_64x32_transpose4x4) {

  using Shape = cutlass::layout::PitchLinearShape<64, 8>;
  using ThreadTileShape = cutlass::layout::PitchLinearShape<4, 4>;
  using Layout = cutlass::layout::PitchLinear;
  using Element = int8_t;
  static int const kThreads = 32;
  
  using ThreadMap = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<Shape, kThreads, ThreadTileShape>;

  using Iterator = cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
    Shape, Element, Layout, 1, ThreadMap, true
  >;

  cutlass::Coord<2> copy_extent = cutlass::make_Coord(64, 32);
  cutlass::Coord<2> alloc_extent = cutlass::make_Coord(64, 32);
  
  cutlass::HostTensor<int8_t, Layout> src_tensor(alloc_extent);
  cutlass::HostTensor<int8_t, Layout> dst_tensor(alloc_extent);

  Element oob_value = Element(-1);
  uint64_t seed = 7;
  cutlass::reference::host::TensorFill(dst_tensor.host_view(), oob_value);
  cutlass::reference::host::TensorFillRandomUniform(src_tensor.host_view(), seed, 8, -8, 0);

  dst_tensor.sync_device();
  src_tensor.sync_device();

  typename Iterator::Params dst_params(dst_tensor.layout());
  typename Iterator::Params src_params(src_tensor.layout());

  dim3 block(kThreads, 1);
  dim3 grid(1, 1);

  test::transform::threadblock::kernel::copy<Iterator><<< grid, block >>>(
    dst_params,
    dst_tensor.device_data(),
    src_params,
    src_tensor.device_data(),
    copy_extent
  );

  cudaError_t result = cudaGetLastError();
  EXPECT_EQ(result, cudaSuccess) << " - CUDA error: " << cudaGetErrorString(result);
  
  dst_tensor.sync_host();

  for (int s = 0; s < alloc_extent[1]/4; ++s) {
    for (int c = 0; c < alloc_extent[0]/4; ++c) {
      for (int s1 = 0; s1 < 4; s1++){
        for(int c1 = 0; c1 < 4; c1++){
          Element expected = Element(0);

          int l_c = c * 4 + c1;
          int l_s = s * 4 + s1;

          int l_tc = c * 4 + s1;
          int l_ts = s * 4 + c1;

          if (l_c < copy_extent[0] && l_s < copy_extent[1]) {
            expected = src_tensor.at({l_c, l_s});
          }
          else {
            expected = oob_value;
          }    

          Element got = dst_tensor.at({l_tc, l_ts});
          bool equal = (expected == got);

          EXPECT_EQ(expected, got)
            << "Source:\n" << src_tensor.host_view() << "\n\n"
            << "Destination:\n" << dst_tensor.host_view() << "\n";

          if (!equal) {
            return;
          }
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Transform_threadblock_PredicatedTileIterator, PitchLinear_Stripmined_2dtile_64x29_transpose4x4) {

  using Shape = cutlass::layout::PitchLinearShape<64, 8>;
  using ThreadTileShape = cutlass::layout::PitchLinearShape<4, 4>;
  using Layout = cutlass::layout::PitchLinear;
  using Element = int8_t;
  static int const kThreads = 32;
  
  using ThreadMap = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<Shape, kThreads, ThreadTileShape>;

  using Iterator = cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
    Shape, Element, Layout, 1, ThreadMap, true
  >;

  cutlass::Coord<2> copy_extent = cutlass::make_Coord(64, 29);
  cutlass::Coord<2> alloc_extent = cutlass::make_Coord(64, 29);
  
  cutlass::HostTensor<int8_t, Layout> src_tensor(alloc_extent);
  cutlass::HostTensor<int8_t, Layout> dst_tensor(alloc_extent);

  Element oob_value = Element(-1);
  uint64_t seed = 7;
  cutlass::reference::host::TensorFill(dst_tensor.host_view(), oob_value);
  cutlass::reference::host::TensorFillRandomUniform(src_tensor.host_view(), seed, 8, -8, 0);

  dst_tensor.sync_device();
  src_tensor.sync_device();

  typename Iterator::Params dst_params(dst_tensor.layout());
  typename Iterator::Params src_params(src_tensor.layout());

  dim3 block(kThreads, 1);
  dim3 grid(1, 1);

  test::transform::threadblock::kernel::copy<Iterator><<< grid, block >>>(
    dst_params,
    dst_tensor.device_data(),
    src_params,
    src_tensor.device_data(),
    copy_extent
  );

  cudaError_t result = cudaGetLastError();
  EXPECT_EQ(result, cudaSuccess) << " - CUDA error: " << cudaGetErrorString(result);
  
  dst_tensor.sync_host();

  for (int s = 0; s < alloc_extent[1]/4; ++s) {
    for (int c = 0; c < alloc_extent[0]/4; ++c) {
      for (int s1 = 0; s1 < 4; s1++){
        for(int c1 = 0; c1 < 4; c1++){
          Element expected = Element(0);

          int l_c = c * 4 + c1;
          int l_s = s * 4 + s1;

          int l_tc = c * 4 + s1;
          int l_ts = s * 4 + c1;

          if (l_c < copy_extent[0] && l_s < copy_extent[1]) {
            expected = src_tensor.at({l_c, l_s});
          }
          else {
            expected = oob_value;
          }    

          Element got = dst_tensor.at({l_tc, l_ts});
          bool equal = (expected == got);

          EXPECT_EQ(expected, got)
            << "Source:\n" << src_tensor.host_view() << "\n\n"
            << "Destination:\n" << dst_tensor.host_view() << "\n";

          if (!equal) {
            return;
          }
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Transform_threadblock_PredicatedTileIterator, PitchLinear_Stripmined_2dtile_120x4_transpose4x4) {

  using Shape = cutlass::layout::PitchLinearShape<128, 4>;
  using ThreadTileShape = cutlass::layout::PitchLinearShape<4, 4>;
  using Layout = cutlass::layout::PitchLinear;
  using Element = int8_t;
  static int const kThreads = 32;
  
  using ThreadMap = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<Shape, kThreads, ThreadTileShape>;

  using Iterator = cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
    Shape, Element, Layout, 1, ThreadMap, true
  >;

  cutlass::Coord<2> copy_extent = cutlass::make_Coord(120, 4);
  cutlass::Coord<2> alloc_extent = cutlass::make_Coord(120, 4);
  
  cutlass::HostTensor<int8_t, Layout> src_tensor(alloc_extent);
  cutlass::HostTensor<int8_t, Layout> dst_tensor(alloc_extent);

  Element oob_value = Element(-1);
  uint64_t seed = 7;
  cutlass::reference::host::TensorFill(dst_tensor.host_view(), oob_value);
  cutlass::reference::host::TensorFillRandomUniform(src_tensor.host_view(), seed, 8, -8, 0);

  dst_tensor.sync_device();
  src_tensor.sync_device();

  typename Iterator::Params dst_params(dst_tensor.layout());
  typename Iterator::Params src_params(src_tensor.layout());

  dim3 block(kThreads, 1);
  dim3 grid(1, 1);

  test::transform::threadblock::kernel::copy<Iterator><<< grid, block >>>(
    dst_params,
    dst_tensor.device_data(),
    src_params,
    src_tensor.device_data(),
    copy_extent
  );

  cudaError_t result = cudaGetLastError();
  EXPECT_EQ(result, cudaSuccess) << " - CUDA error: " << cudaGetErrorString(result);
  
  dst_tensor.sync_host();

  for (int s = 0; s < alloc_extent[1]/4; ++s) {
    for (int c = 0; c < alloc_extent[0]/4; ++c) {
      for (int s1 = 0; s1 < 4; s1++){
        for(int c1 = 0; c1 < 4; c1++){
          Element expected = Element(0);

          int l_c = c * 4 + c1;
          int l_s = s * 4 + s1;

          int l_tc = c * 4 + s1;
          int l_ts = s * 4 + c1;

          if (l_c < copy_extent[0] && l_s < copy_extent[1]) {
            expected = src_tensor.at({l_c, l_s});
          }
          else {
            expected = oob_value;
          }    

          Element got = dst_tensor.at({l_tc, l_ts});
          bool equal = (expected == got);

          EXPECT_EQ(expected, got)
            << "Source:\n" << src_tensor.host_view() << "\n\n"
            << "Destination:\n" << dst_tensor.host_view() << "\n";

          if (!equal) {
            return;
          }
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Transform_threadblock_PredicatedTileIterator, PitchLinear_Stripmined_2dtile_48x29_transpose4x4) {

  using Shape = cutlass::layout::PitchLinearShape<64, 8>;
  using ThreadTileShape = cutlass::layout::PitchLinearShape<4, 4>;
  using Layout = cutlass::layout::PitchLinear;
  using Element = int8_t;
  static int const kThreads = 32;
  
  using ThreadMap = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<Shape, kThreads, ThreadTileShape>;

  using Iterator = cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
    Shape, Element, Layout, 1, ThreadMap, true
  >;

  cutlass::Coord<2> copy_extent = cutlass::make_Coord(48, 29);
  cutlass::Coord<2> alloc_extent = cutlass::make_Coord(48, 29);
  
  cutlass::HostTensor<int8_t, Layout> src_tensor(alloc_extent);
  cutlass::HostTensor<int8_t, Layout> dst_tensor(alloc_extent);

  Element oob_value = Element(-1);
  uint64_t seed = 7;
  cutlass::reference::host::TensorFill(dst_tensor.host_view(), oob_value);
  cutlass::reference::host::TensorFillRandomUniform(src_tensor.host_view(), seed, 8, -8, 0);

  dst_tensor.sync_device();
  src_tensor.sync_device();

  typename Iterator::Params dst_params(dst_tensor.layout());
  typename Iterator::Params src_params(src_tensor.layout());

  dim3 block(kThreads, 1);
  dim3 grid(1, 1);

  test::transform::threadblock::kernel::copy<Iterator><<< grid, block >>>(
    dst_params,
    dst_tensor.device_data(),
    src_params,
    src_tensor.device_data(),
    copy_extent
  );

  cudaError_t result = cudaGetLastError();
  EXPECT_EQ(result, cudaSuccess) << " - CUDA error: " << cudaGetErrorString(result);
  
  dst_tensor.sync_host();

  for (int s = 0; s < alloc_extent[1]/4; ++s) {
    for (int c = 0; c < alloc_extent[0]/4; ++c) {
      for (int s1 = 0; s1 < 4; s1++){
        for(int c1 = 0; c1 < 4; c1++){
          Element expected = Element(0);

          int l_c = c * 4 + c1;
          int l_s = s * 4 + s1;

          int l_tc = c * 4 + s1;
          int l_ts = s * 4 + c1;

          if (l_c < copy_extent[0] && l_s < copy_extent[1]) {
            expected = src_tensor.at({l_c, l_s});
          }
          else {
            expected = oob_value;
          }    

          Element got = dst_tensor.at({l_tc, l_ts});
          bool equal = (expected == got);

          EXPECT_EQ(expected, got)
            << "Source:\n" << src_tensor.host_view() << "\n\n"
            << "Destination:\n" << dst_tensor.host_view() << "\n";
          if (!equal) {
            return;
          }
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
