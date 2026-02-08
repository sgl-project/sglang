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
    \brief Unit testbed for kernel-level GEMM
*/

#pragma once

#include <fstream>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/cutlass.h"
#include "cutlass/platform/platform.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/numeric_types.h"

#include "cutlass/core_io.h"
#include "cutlass/util/host_tensor_planar_complex.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/gemm_planar_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma>
__global__ void kernel_mma_planar_complex(
  cutlass::gemm::GemmCoord problem_size,
  typename Mma::IteratorA::Params params_A,
  typename Mma::IteratorA::Element *ptr_A,
  int64_t imaginary_stride_A,
  typename Mma::IteratorB::Params params_B,
  typename Mma::IteratorB::Element *ptr_B,
  int64_t imaginary_stride_B,
  typename Mma::ElementC *ptr_C, 
  typename Mma::LayoutC::Stride::Index ldc, int64_t imaginary_stride_C) {

  // Shared storage needed by threadblock-scoped matrix multiply-accumulate
  __shared__ typename Mma::SharedStorage shared_storage;

  // Compute threadblock location
  cutlass::gemm::GemmCoord tb_tile_offset = {int(blockIdx.x), int(blockIdx.y),
                                             0};

  cutlass::MatrixCoord tb_offset_A{tb_tile_offset.m() * Mma::Shape::kM,
                                   tb_tile_offset.k()};

  cutlass::MatrixCoord tb_offset_B{tb_tile_offset.k(),
                                   tb_tile_offset.n() * Mma::Shape::kN};

  // Compute position within threadblock
  int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  // Construct iterators to A operand
  typename Mma::IteratorA iterator_A_real(params_A, ptr_A,
                                     {problem_size.m(), problem_size.k()},
                                     tb_thread_id, tb_offset_A);
  
  typename Mma::IteratorA iterator_A_imag(params_A, ptr_A + imaginary_stride_A,
                                     {problem_size.m(), problem_size.k()},
                                     tb_thread_id, tb_offset_A);
  
  // Construct iterators to B operand
  typename Mma::IteratorB iterator_B_real(params_B, ptr_B,
                                     {problem_size.k(), problem_size.n()},
                                     tb_thread_id, tb_offset_B);

  typename Mma::IteratorB iterator_B_imag(params_B, ptr_B + imaginary_stride_B,
                                     {problem_size.k(), problem_size.n()},
                                     tb_thread_id, tb_offset_B);

  int warp_id = threadIdx.y;
  int lane_id = threadIdx.x;

  // Construct thread-scoped matrix multiply
  Mma mma(shared_storage, tb_thread_id, warp_id, threadIdx.x);

  typename Mma::FragmentC accum;

  accum.clear();

  int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

  // Compute threadblock-scoped matrix multiply-add
  mma(gemm_k_iterations, accum, iterator_A_real, iterator_A_imag, iterator_B_real, iterator_B_imag, accum);

  // Output results
  typename Mma::Operator::IteratorC iterator_C({ptr_C, ldc}, lane_id);

  iterator_C.add_tile_offset(
      {(tb_tile_offset.m() * Mma::WarpCount::kM) +
           (warp_id % Mma::WarpCount::kM),
       (tb_tile_offset.n() * Mma::WarpCount::kN) +
           (warp_id / Mma::WarpCount::kM)});

  iterator_C.store(accum.real);

  iterator_C.store_with_pointer_offset(accum.imag, imaginary_stride_C);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
    /// Threadblock-level matrix multiply-accumulate
    typename Mma_>
struct TestbedPlanarComplex {

  using Mma = Mma_;
  using ThreadblockShape = typename Mma::Shape;
  using IteratorA = typename Mma::IteratorA;
  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using IteratorB = typename Mma::IteratorB;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename Mma::ElementC;
  using ElementAccumulator = typename Mma::ElementC;
  using LayoutC = typename Mma::LayoutC;
  using ThreadMapA = typename Mma::IteratorA::ThreadMap;
  using ThreadMapB = typename Mma::IteratorB::ThreadMap;
  using AccessTypeA = cutlass::Array<ElementA, ThreadMapA::kElementsPerAccess>;
  using AccessTypeB = cutlass::Array<ElementB, ThreadMapB::kElementsPerAccess>;
  static int const Stages = Mma::kStages;
  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      Mma::kCacheOpA;
  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      Mma::kCacheOpB;

  //
  // Data members
  //

  cutlass::HostTensorPlanarComplex<ElementA, LayoutA> matrix_A;
  cutlass::HostTensorPlanarComplex<ElementB, LayoutB> matrix_B;
  cutlass::HostTensorPlanarComplex<ElementC, LayoutC> matrix_C_computed;
  cutlass::HostTensorPlanarComplex<ElementC, LayoutC> matrix_C_reference;

  cutlass::gemm::GemmCoord problem_size;

  //
  // Methods
  //

  /// Allocates workspace in device memory
  TestbedPlanarComplex(int m, int n, int k)
      : problem_size(m, n, k) {

    matrix_A.reset(cutlass::make_Coord(m, k));
    matrix_B.reset(cutlass::make_Coord(k, n));
    matrix_C_computed.reset(cutlass::make_Coord(m, n));
    matrix_C_reference.reset(cutlass::make_Coord(m, n), false);
  }

  /// Runs the test
  bool run(
      dim3 grid, dim3 block,
      cutlass::Distribution::Kind init_A = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B = cutlass::Distribution::Uniform) {

    //
    // initialize device memory
    //

    if (init_A == cutlass::Distribution::Uniform) {
      
      int scope_max = 8;
      int scope_min = -8;

      if (cutlass::sizeof_bits<ElementA>::value == 4) {
        scope_max = 2;
        scope_min = -2;
      } else if (cutlass::sizeof_bits<ElementA>::value == 1) {
        scope_max = 2;
        scope_min = 0;
      }

      uint64_t seed = 7;
      cutlass::reference::host::TensorFillRandomUniform(
          matrix_A.host_view(), seed, scope_max, scope_min, 0);
      
    } else if (init_A == cutlass::Distribution::Sequential) {
      
      for (int i = 0; i < matrix_A.capacity() * 2; ++i) {
        matrix_A.host_data()[i] = cutlass::half_t(float(i % 5) - 2);
      }
      /*
      cutlass::reference::host::BlockFillSequential(matrix_A.host_data(),
                                                    matrix_A.capacity() * 2);
      */
    } else if (init_A == cutlass::Distribution::Identity) {
      //cutlass::reference::host::TensorFillIdentity(matrix_A.host_view());
    } else {
      return false;
    }

    if (init_B == cutlass::Distribution::Uniform) {

      
      int scope_max = 8;
      int scope_min = -8;

      if (cutlass::sizeof_bits<ElementB>::value == 4) {
        scope_max = 2;
        scope_min = -2;
      } else if (cutlass::sizeof_bits<ElementB>::value == 1) {
        scope_max = 2;
        scope_min = 0;
      }

      uint64_t seed = 7;
      cutlass::reference::host::TensorFillRandomUniform(
          matrix_B.host_view(), seed + 16, scope_max, scope_min, 0);
      

    } else if (init_B == cutlass::Distribution::Sequential) {

      cutlass::reference::host::BlockFillSequential(matrix_B.host_data(),
                                                    matrix_B.capacity() * 2);

      for (int i = 0; i < matrix_B.capacity() * 2; ++i) {
        matrix_B.host_data()[i] = cutlass::half_t(float((i + 3) % 5) - 2);
      }


    } else if (init_B == cutlass::Distribution::Identity) {

      //cutlass::reference::host::TensorFillIdentity(matrix_B.host_view());

    } else {
      return false;
    }

    matrix_A.sync_device();
    matrix_B.sync_device();
    matrix_C_computed.sync_device();

    typename IteratorA::Params params_A(matrix_A.layout());
    typename IteratorB::Params params_B(matrix_B.layout());

    test::gemm::threadblock::kernel_mma_planar_complex<Mma><<<grid, block>>>(
        problem_size, 
        params_A, 
        matrix_A.device_data(),
        matrix_A.imaginary_stride(),
        params_B,
        matrix_B.device_data(), 
        matrix_B.imaginary_stride(),
        matrix_C_computed.device_data(),
        matrix_C_computed.layout().stride(0), 
        matrix_C_computed.imaginary_stride()
      );


    //
    // Check error code
    //

    cudaError_t result = cudaDeviceSynchronize();
    EXPECT_EQ(result, cudaSuccess)
        << " kernel error: " << cudaGetErrorString(result);

    matrix_C_computed.sync_host();

    cutlass::reference::host::GemmPlanarComplex<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementAccumulator
    >(
      problem_size,
      cutlass::complex<ElementAccumulator>(ElementAccumulator(1)),
      matrix_A.host_ref(),
      Mma::kTransformA,
      matrix_B.host_ref(),
      Mma::kTransformB,
      cutlass::complex<ElementAccumulator>(ElementAccumulator(0)),
      matrix_C_reference.host_ref(),
      matrix_C_reference.host_ref()
    );
    
    bool passed = cutlass::reference::host::TensorEquals(
      matrix_C_computed.host_view(), 
      matrix_C_reference.host_view()
    );

    EXPECT_TRUE(passed);

    if (!passed) {
      std::ofstream output("mma_pipelined_testbed_errors.txt");

      output
        << "A:\n" << matrix_A.host_view() << "\n"
        << "B:\n" << matrix_B.host_view() << "\n"
        << "Reference:\n"
        << matrix_C_reference.host_view() << "\n"
        << "Computed:\n"
        << matrix_C_computed.host_view() << "\n";
    }

    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace test
