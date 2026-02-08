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

#include "cutlass/aligned_buffer.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/numeric_types.h"

#include "cutlass/core_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"
#include "cutlass/cutlass.h"
#include "cutlass/platform/platform.h"

namespace test {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma>
__global__ void kernel_mma(cutlass::gemm::GemmCoord problem_size,
                           typename Mma::IteratorA::Params params_A,
                           typename Mma::IteratorA::TensorRef ref_A,
                           typename Mma::IteratorB::Params params_B,
                           typename Mma::IteratorB::TensorRef ref_B,
                           typename Mma::ElementC *ptr_C,
                           typename Mma::LayoutC::Stride::Index ldc) {
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

  // Construct iterators to A and B operands
  typename Mma::IteratorA iterator_A(params_A, ref_A.data(),
                                     {problem_size.m(), problem_size.k()},
                                     tb_thread_id, tb_offset_A);

  typename Mma::IteratorB iterator_B(params_B, ref_B.data(),
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
  mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

  // Output results
  typename Mma::Operator::IteratorC iterator_C({ptr_C, ldc}, lane_id);

  iterator_C.add_tile_offset(
      {(tb_tile_offset.m() * Mma::WarpCount::kM) +
           (warp_id % Mma::WarpCount::kM),
       (tb_tile_offset.n() * Mma::WarpCount::kN) +
           (warp_id / Mma::WarpCount::kM)});

  iterator_C.store(accum);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
    /// Threadblock-level matrix multiply-accumulate
    typename MmaCore_,
    /// Number of stages
    int Stages = 2>
struct Testbed {
  /// Threadblock-level GEMM implementation
  using MmaCore = MmaCore_;
  using ThreadblockShape = typename MmaCore::Shape;
  using WarpShape = typename MmaCore::WarpShape;
  using InstructionShape = typename MmaCore::InstructionShape;
  using ElementA = typename MmaCore::ElementA;
  using LayoutA = typename MmaCore::LayoutA;
  using ElementB = typename MmaCore::ElementB;
  using LayoutB = typename MmaCore::LayoutB;
  using ElementC = typename MmaCore::ElementC;
  using LayoutC = typename MmaCore::LayoutC;
  static const int kStages = Stages;

  // Define iterators over tiles from the A operand
  static const bool use_idp4a = cutlass::platform::is_same<ElementA, int8_t>::value && 
                                cutlass::platform::is_same<ElementB, int8_t>::value && 
                                cutlass::platform::is_same<typename MmaCore::OperatorClass, cutlass::arch::OpClassSimt>::value;

  static const bool transposeA =  cutlass::platform::is_same< LayoutA, cutlass::layout::ColumnMajor >::value;
  static const bool transposeB =  cutlass::platform::is_same< LayoutB, cutlass::layout::RowMajor >::value;

  using IteratorA = typename cutlass::platform::conditional< use_idp4a,
      cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, transposeA> ,
        
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA>
      >::type;

  // Define iterators over tiles from the B operand
  using IteratorB = typename cutlass::platform::conditional< use_idp4a,
      cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, transposeB> ,

      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB>
      >::type;

  // Define MmaPipeline Single Stage
  using MmaPipelineSingleStage =  cutlass::gemm::threadblock::MmaSingleStage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementC, LayoutC,
      typename MmaCore::MmaPolicy>;

  // Define MmaPipeline Two Stages
  using MmaPipelineTwoStages =  cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementC, LayoutC,
      typename MmaCore::MmaPolicy>;
  
  // Define the threadblock-scoped pipelined matrix multiply (Select between Single vs. Two stages)
  using Mma = typename cutlass::platform::conditional<(kStages==1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;
  //
  // Data members
  //

  cutlass::HostTensor<ElementA, LayoutA> matrix_A;
  cutlass::HostTensor<ElementB, LayoutB> matrix_B;
  cutlass::HostTensor<ElementC, LayoutC> matrix_C_computed;
  cutlass::HostTensor<ElementC, LayoutC> matrix_C_reference;

  cutlass::gemm::GemmCoord problem_size;
  float alpha, beta;

  //
  // Methods
  //

  /// Allocates workspace in device memory
  Testbed(int m, int n, int k, float alpha_, float beta_)
      : problem_size(m, n, k), alpha(alpha_), beta(beta_) {
    matrix_A.reset(cutlass::make_Coord(m, k));
    matrix_B.reset(cutlass::make_Coord(k, n));
    matrix_C_computed.reset(cutlass::make_Coord(m, n));
    matrix_C_reference.reset(cutlass::make_Coord(m, n), false);
  }

  bool sufficient() {
    return true;
  }

  /// Runs the test
  bool run(
      dim3 grid, dim3 block,
      cutlass::Distribution::Kind init_A = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B = cutlass::Distribution::Uniform) {

    // Waive test if insufficient CUDA device
    if (!sufficient()) {
      if (CUTLASS_TEST_UNIT_ENABLE_WARNINGS) {
        std::cerr << "Test waived due to insufficient CUDA device." << std::endl;
      }
      return true;
    }


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
      cutlass::reference::host::BlockFillSequential(matrix_A.host_data(),
                                                    matrix_A.capacity());
    } else if (init_A == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(matrix_A.host_view());
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
                                                    matrix_B.capacity());
    } else if (init_B == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(matrix_B.host_view());
    } else {
      return false;
    }

    cutlass::reference::host::TensorFill(matrix_C_computed.host_view());

    cutlass::reference::host::TensorFill(matrix_C_reference.host_view());

    matrix_A.sync_device();
    matrix_B.sync_device();
    matrix_C_computed.sync_device();

    typename IteratorA::Params params_A(matrix_A.layout());
    typename IteratorB::Params params_B(matrix_B.layout());

    test::gemm::threadblock::kernel_mma<Mma><<<grid, block>>>(
        problem_size, params_A, matrix_A.device_ref(), params_B,
        matrix_B.device_ref(), matrix_C_computed.device_data(),
        matrix_C_computed.layout().stride(0));

    //
    // Check error code
    //

    cudaError_t result = cudaDeviceSynchronize();
    EXPECT_EQ(result, cudaSuccess)
        << " kernel error: " << cudaGetErrorString(result) << " on device " << GetCudaDevice();

    matrix_C_computed.sync_host();

    cutlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                   ElementC, LayoutC, ElementC, ElementC,
                                   typename MmaCore::Operator>
        reference_gemm;

    reference_gemm(
        problem_size, ElementC(alpha), matrix_A.host_view(),
        matrix_B.host_view(), ElementC(beta), matrix_C_reference.host_view());

    bool passed = cutlass::reference::host::TensorEquals(
        matrix_C_computed.host_view(), matrix_C_reference.host_view());

    EXPECT_TRUE(passed) << "Failed on device " << GetCudaDevice();

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
