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
    \brief Statically sized array of elements that accommodates all CUTLASS-supported numeric types
           and is safe to use in a union.
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/array.h"
#include "cutlass/util/device_memory.h"
#pragma warning( disable : 4800)
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace core {

/// Each thread clears its array and writes to global memory. No PRMT instructions should
/// be generated if Array<T, N> is a multiple of 32 bits.
template <typename T, int N>
__global__ void test_array_clear(cutlass::Array<T, N> *ptr) {

  cutlass::Array<T, N> storage;

  storage.clear();

  ptr[threadIdx.x] = storage;
}

/// Each thread writes its thread index into the elements of its array and then writes the result
/// to global memory.
template <typename T, int N>
__global__ void test_array_threadid(cutlass::Array<T, N> *ptr) {

  cutlass::Array<T, N> storage;

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N; ++i) {
    storage.at(i) = T(int(threadIdx.x));
  }

  ptr[threadIdx.x] = storage;
}

/// Each thread writes its thread index into the elements of its array and then writes the result
/// to global memory.
template <typename T, int N>
__global__ void test_array_sequence(cutlass::Array<T, N> *ptr) {

  cutlass::Array<T, N> storage;

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N; ++i) {
    storage.at(i) = T(i);
  }

  ptr[threadIdx.x] = storage;
}

} // namespace core
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int N>
class TestArray {
public:

  //
  // Data members
  //

  /// Number of threads
  int const kThreads = 32;

  typedef cutlass::Array<T, N> ArrayTy;


  //
  // Methods
  //

  /// Ctor
  TestArray() {

  }

  /// Runs the test
  void run() {

    /// Device memory containing output
    cutlass::device_memory::allocation< ArrayTy > output(static_cast<size_t>(kThreads));
    std::vector< ArrayTy > output_host(static_cast<size_t>(kThreads));

    dim3 grid(1,1);
    dim3 block(kThreads, 1, 1);

    test::core::test_array_clear<<< grid, block >>>(output.get());

    cudaError_t result = cudaDeviceSynchronize();
    ASSERT_EQ(result, cudaSuccess) << "CUDA error: " << cudaGetErrorString(result);

    //
    // Verify contains all zeros
    //

    cutlass::device_memory::copy_to_host(output_host.data(), output.get(), kThreads);

    result = cudaGetLastError();
    ASSERT_EQ(result, cudaSuccess) << "CUDA error: " << cudaGetErrorString(result);

    char const *ptr_host = reinterpret_cast<char const *>(output_host.data());
    for (size_t i = 0; i < sizeof(ArrayTy) * kThreads; ++i) {
      EXPECT_FALSE(ptr_host[i]);
    }

    //
    // Verify each element contains the low bits of the thread Id
    //

    test::core::test_array_threadid<<< grid, block >>>(output.get());

    result = cudaDeviceSynchronize();
    ASSERT_EQ(result, cudaSuccess) << "CUDA error: " << cudaGetErrorString(result);

    cutlass::device_memory::copy_to_host(output_host.data(), output.get(), kThreads);

    result = cudaGetLastError();
    ASSERT_EQ(result, cudaSuccess) << "CUDA error: " << cudaGetErrorString(result);

    for (int i = 0; i < kThreads; ++i) {
      T tid = T(i);

      ArrayTy thread = output_host.at(i);

      // Element-wise access
      for (int j = 0; j < N; ++j) {
        EXPECT_TRUE(tid == thread[j]);
      }

      // Iterator access
      for (auto it = thread.begin(); it != thread.end(); ++it) {
        EXPECT_TRUE(tid == *it);
      }

      // Range-based for
      for (auto const & x : thread) {
        EXPECT_TRUE(tid == x);
      }
    }

    //
    // Verify each element
    //

    test::core::test_array_sequence<<< grid, block >>>(output.get());

    result = cudaDeviceSynchronize();
    ASSERT_EQ(result, cudaSuccess) << "CUDA error: " << cudaGetErrorString(result);

    cutlass::device_memory::copy_to_host(output_host.data(), output.get(), kThreads);

    result = cudaGetLastError();
    ASSERT_EQ(result, cudaSuccess) << "CUDA error: " << cudaGetErrorString(result);

    for (int i = 0; i < kThreads; ++i) {

      ArrayTy thread = output_host.at(i);

      // Element-wise access
      for (int j = 0; j < N; ++j) {
        T got = T(j);
        EXPECT_TRUE(got == thread[j]);
      }

      // Iterator access
      int j = 0;
      for (auto it = thread.begin(); it != thread.end(); ++it, ++j) {
        T got = T(j);
        EXPECT_TRUE(got == *it);
      }

      // Range-based for
      j = 0;
      for (auto const & x : thread) {
        T got = T(j);
        EXPECT_TRUE(got == x);
        ++j;
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
TEST(Array, Int8x16) {
  TestArray<int8_t, 16>().run();
}

TEST(Array, Int32x4) {
  TestArray<int, 4>().run();
}

#if __CUDA_ARCH__ >= 520
TEST(Array, Float16x8) {
  TestArray<cutlass::half_t, 8>().run();
}
#endif

TEST(Array, FloatBF16x8) {
  TestArray<cutlass::bfloat16_t, 8>().run();
}

TEST(Array, FloatTF32x4) {
  TestArray<cutlass::tfloat32_t, 4>().run();
}

TEST(Array, Float32x4) {
  TestArray<float, 4>().run();
}

TEST(Array, Int4x32) {
  TestArray<cutlass::int4b_t, 32>().run();
}

TEST(Array, Uint4x32) {
  TestArray<cutlass::uint4b_t, 32>().run();
}

TEST(Array, Bin1x128) {
  TestArray<cutlass::bin1_t, 128>().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
