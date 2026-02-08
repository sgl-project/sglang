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

#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
#include "cutlass/cutlass.h"
#include "cutlass/platform/platform.h"

namespace test {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma>
__global__ void kernel_multistage_mma(cutlass::gemm::GemmCoord problem_size,
                           typename Mma::IteratorA::Params params_A,
                           typename Mma::IteratorA::TensorRef ref_A,
                           typename Mma::IteratorB::Params params_B,
                           typename Mma::IteratorB::TensorRef ref_B,
                           typename Mma::ElementC **ptr_C,
                           typename Mma::LayoutC::Stride::Index ldc) {
  // Shared storage needed by threadblock-scoped matrix multiply-accumulate

  // Dynamic shared memory base pointer
  extern __shared__ int GemmSharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  typename Mma::SharedStorage *shared_storage =
      reinterpret_cast<typename Mma::SharedStorage *>(GemmSharedStorageBase);

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

  int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
  int lane_id = threadIdx.x;

  int partitionsK_idx = warp_id / (Mma::WarpCount::kM * Mma::WarpCount::kN);

  // Construct thread-scoped matrix multiply
  Mma mma(*shared_storage, tb_thread_id, warp_id, threadIdx.x);

  typename Mma::FragmentC accum;

  accum.clear();

  int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

  // Compute threadblock-scoped matrix multiply-add
  mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

  // Output results
  typename Mma::Operator::IteratorC iterator_C({ptr_C[partitionsK_idx], ldc}, lane_id);

  int warp_idx_mn = warp_id % (Mma::WarpCount::kM * Mma::WarpCount::kN);
  iterator_C.add_tile_offset(
      {(tb_tile_offset.m() * Mma::WarpCount::kM) +
           (warp_idx_mn % Mma::WarpCount::kM),
       (tb_tile_offset.n() * Mma::WarpCount::kN) +
           (warp_idx_mn / Mma::WarpCount::kM)});

  iterator_C.store(accum);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
    /// Threadblock-level matrix multiply-accumulate
    typename MmaCore_>
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
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeA = cutlass::Array<ElementA, ThreadMapA::kElementsPerAccess>;
  using AccessTypeB = cutlass::Array<ElementB, ThreadMapB::kElementsPerAccess>;
  static int const Stages = MmaCore::kStages;
  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      MmaCore::kCacheOpA;
  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      MmaCore::kCacheOpB;

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

  // Define the threadblock-scoped pipelined matrix multiply
  using Mma = cutlass::gemm::threadblock::MmaMultistage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA, CacheOpA,
      IteratorB, typename MmaCore::SmemIteratorB, CacheOpB, ElementC, LayoutC,
      typename MmaCore::MmaPolicy, Stages>;

  static int const kPartitionsK = MmaCore::MmaPolicy::kPartitionsK; 

  //
  // Data members
  //

  cutlass::HostTensor<ElementA, LayoutA> matrix_A;
  cutlass::HostTensor<ElementB, LayoutB> matrix_B;
  cutlass::HostTensor<ElementC, LayoutC> matrix_C_computed[kPartitionsK];
  cutlass::HostTensor<ElementC, LayoutC> matrix_C_reference;
  cutlass::HostTensor<ElementC*, cutlass::layout::PackedVectorLayout> matrix_C_pointers;

  cutlass::gemm::GemmCoord problem_size;
  float alpha, beta;

  //
  // Methods
  //

  /// Allocates workspace in device memory
  Testbed(int m, int n, int k, float alpha_ = float(1), float beta_ = float(0))
      : problem_size(m, n, k), alpha(alpha_), beta(beta_) {
    matrix_A.reset(cutlass::make_Coord(m, k));
    matrix_B.reset(cutlass::make_Coord(k, n));

    CUTLASS_PRAGMA_UNROLL
    for(int k = 0; k < kPartitionsK; k++)
      matrix_C_computed[k].reset(cutlass::make_Coord(m, n));

    matrix_C_reference.reset(cutlass::make_Coord(m, n), false);
    matrix_C_pointers.reset(cutlass::Coord<1>(kPartitionsK));
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

    CUTLASS_PRAGMA_UNROLL
    for(int k = 0; k < kPartitionsK; k++)
      cutlass::reference::host::TensorFill(matrix_C_computed[k].host_view());

    cutlass::reference::host::TensorFill(matrix_C_reference.host_view());

    matrix_A.sync_device();
    matrix_B.sync_device();

    CUTLASS_PRAGMA_UNROLL
    for(int k = 0; k < kPartitionsK; k++)
      matrix_C_computed[k].sync_device();

    typename IteratorA::Params params_A(matrix_A.layout());
    typename IteratorB::Params params_B(matrix_B.layout());

    CUTLASS_PRAGMA_UNROLL
    for(int k = 0; k < kPartitionsK; k++)
      matrix_C_pointers.at(cutlass::Coord<1>(k)) = matrix_C_computed[k].device_data();

    matrix_C_pointers.sync_device();

    cudaError_t result;

    int smem_size = int(sizeof(typename Mma::SharedStorage));
    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(
          test::gemm::threadblock::kernel_multistage_mma<Mma>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

      EXPECT_EQ(result, cudaSuccess)
          << " cudaFuncSetAttribute "
             "cudaFuncAttributeMaxDynamicSharedMemorySize error: "
          << cudaGetErrorString(result);

      result = cudaFuncSetAttribute(
          test::gemm::threadblock::kernel_multistage_mma<Mma>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100);

      EXPECT_EQ(result, cudaSuccess)
          << " cudaFuncSetAttribute "
             "cudaFuncAttributePreferredSharedMemoryCarveout error: "
          << cudaGetErrorString(result);
    }

    test::gemm::threadblock::kernel_multistage_mma<Mma><<<grid, block, smem_size, 0>>>(
        problem_size, params_A, matrix_A.device_ref(), params_B,
        matrix_B.device_ref(), matrix_C_pointers.device_data(),
        matrix_C_computed[0].layout().stride(0));

    //
    // Check error code
    //

    result = cudaDeviceSynchronize();
    EXPECT_EQ(result, cudaSuccess)
        << " kernel error: " << cudaGetErrorString(result);

    CUTLASS_PRAGMA_UNROLL
    for(int k = 0; k < kPartitionsK; k++)
      matrix_C_computed[k].sync_host();

    // TODO: this is temporary. it will be removed after slicing can de
    // reduction
    //
    // Reduce matrix_C_computed
    //
    CUTLASS_PRAGMA_UNROLL
    for(int k = 1; k < kPartitionsK; k++) {
      CUTLASS_PRAGMA_UNROLL
      for(int m = 0; m < matrix_C_computed[0].extent().row(); m++){
        CUTLASS_PRAGMA_UNROLL
        for(int n = 0; n < matrix_C_computed[0].extent().column(); n++){
          matrix_C_computed[0].at({m, n}) += matrix_C_computed[k].at({m, n});
        }
      }
    }

    cutlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                   ElementC, LayoutC, ElementC, ElementC,
                                   typename MmaCore::Operator>
        reference_gemm;

    reference_gemm(
        problem_size, ElementC(alpha), matrix_A.host_view(),
        matrix_B.host_view(), ElementC(beta), matrix_C_reference.host_view());

    bool passed = cutlass::reference::host::TensorEquals(
        matrix_C_computed[0].host_view(), matrix_C_reference.host_view());

    EXPECT_TRUE(passed);

    if (!passed) {
      std::ofstream output("mma_multistage_testbed_errors.txt");

      output
        << "A:\n" << matrix_A.host_view() << "\n"
        << "B:\n" << matrix_B.host_view() << "\n"
        << "Reference:\n"
        << matrix_C_reference.host_view() << "\n"
        << "Computed:\n"
        << matrix_C_computed[0].host_view() << "\n";
    }

    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace test
