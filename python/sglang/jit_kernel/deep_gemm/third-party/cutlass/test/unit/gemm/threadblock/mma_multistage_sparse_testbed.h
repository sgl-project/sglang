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

#include "../../common/cutlass_unit_test.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/core_io.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma_core_sparse_sm80.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/host_uncompress.h"

namespace test {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <typename Mma>
__global__ void kernel_multistage_mma_sparse(cutlass::gemm::GemmCoord problem_size,
                                      typename Mma::IteratorA::Params params_A,
                                      typename Mma::IteratorA::TensorRef ref_A,
                                      typename Mma::IteratorB::Params params_B,
                                      typename Mma::IteratorB::TensorRef ref_B,
                                      typename Mma::ElementC *ptr_C,
                                      typename Mma::LayoutC::Stride::Index ldc,
                                      typename Mma::IteratorE::Params params_E,
                                      typename Mma::IteratorE::TensorRef ref_E) {
  // Shared storage needed by threadblock-scoped matrix multiply-
  // Dynamic shared memory base pointer
  extern __shared__ int GemmSharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  typename Mma::SharedStorage *shared_storage =
      reinterpret_cast<typename Mma::SharedStorage *>(GemmSharedStorageBase);

  // Compute threadblock location
  cutlass::gemm::GemmCoord tb_tile_offset = {int(blockIdx.x), int(blockIdx.y),
                                             0};

  cutlass::MatrixCoord tb_offset_A{tb_tile_offset.m() * Mma::Shape::kM,
                                   tb_tile_offset.k() / Mma::kSparse};

  cutlass::MatrixCoord tb_offset_B{tb_tile_offset.k(),
                                   tb_tile_offset.n() * Mma::Shape::kN};

  cutlass::MatrixCoord tb_offset_E{tb_tile_offset.m() * Mma::Shape::kM,
                                   tb_tile_offset.k() / Mma::kSparse};

  // Compute position within threadblock
  int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  // Construct iterators to A and B operands
  typename Mma::IteratorA iterator_A(params_A, ref_A.data(),
                                     {problem_size.m(), problem_size.k() / Mma::kSparse},
                                     tb_thread_id, tb_offset_A);

  typename Mma::IteratorB iterator_B(params_B, ref_B.data(),
                                     {problem_size.k(), problem_size.n()},
                                     tb_thread_id, tb_offset_B);

  typename Mma::IteratorE iterator_E(
      params_E, ref_E.data(),
      {problem_size.m(),
       problem_size.k() / Mma::kSparse / Mma::kElementsPerElementE},
      tb_thread_id, tb_offset_E);

  int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);

  // Construct thread-scoped matrix multiply
  Mma mma(*shared_storage, tb_thread_id, warp_id, threadIdx.x);

  typename Mma::FragmentC accum;

  accum.clear();

  int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

  // Compute threadblock-scoped matrix multiply-add
  mma(gemm_k_iterations, accum, iterator_A, iterator_B, iterator_E, accum);

  // Output results
  typename Mma::Operator::IteratorC iterator_C({ptr_C, ldc}, threadIdx.x);

  iterator_C.add_tile_offset(
      {(tb_tile_offset.m() * Mma::WarpCount::kM) +
           (warp_id % Mma::WarpCount::kM),
       (tb_tile_offset.n() * Mma::WarpCount::kN) +
           (warp_id / Mma::WarpCount::kM)});

  iterator_C.store(accum);
}

////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
    /// Threadblock-level matrix multiply-accumulate
    typename MmaCore_>
struct SparseTestbed {
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
  using ElementE = typename MmaCore::ElementE;
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using ThreadMapE = typename MmaCore::IteratorThreadMapE;
  using AccessTypeA = cutlass::Array<ElementA, ThreadMapA::kElementsPerAccess>;
  using AccessTypeB = cutlass::Array<ElementB, ThreadMapB::kElementsPerAccess>;
  using AccessTypeE = cutlass::Array<ElementE, ThreadMapE::kElementsPerAccess>;
  static int const Stages = MmaCore::kStages;
  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      MmaCore::kCacheOpA;
  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      MmaCore::kCacheOpB;
  static cutlass::arch::CacheOperation::Kind const CacheOpE =
      MmaCore::kCacheOpE;

  static int const Sparse = MmaCore::kSparse;
  static int const MetaSizeInBits = MmaCore::kMetaSizeInBits;
  static int const MaxID2 = MmaCore::kMaxID2;

  using LayoutE = cutlass::layout::RowMajor;
  using ReorderedLayoutE = typename MmaCore::GmemLayoutE;

  static int const ElementsPerElementE = MmaCore::kElementsPerElementE;

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK / Sparse>,
          ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

  // Define iterators over tiles from the E operand
  using IteratorE =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK /
                                                         Sparse /
                                                         ElementsPerElementE>,
          ElementE, ReorderedLayoutE, 1, ThreadMapE, AccessTypeE>;

  // Define the threadblock-scoped pipelined matrix multiply
  using Mma = cutlass::gemm::threadblock::SparseMmaMultistage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      CacheOpA, IteratorB, typename MmaCore::SmemIteratorB, CacheOpB, ElementC,
      LayoutC, IteratorE, typename MmaCore::SmemIteratorE, CacheOpE,
      typename MmaCore::MmaPolicy, Stages>;

  //
  // Data members
  //

  cutlass::HostTensor<ElementA, LayoutA> matrix_A;
  cutlass::HostTensor<ElementA, LayoutA> matrix_A_uncompressed;
  cutlass::HostTensor<ElementB, LayoutB> matrix_B;
  cutlass::HostTensor<ElementC, LayoutC> matrix_C_computed;
  cutlass::HostTensor<ElementC, LayoutC> matrix_C_reference;
  cutlass::HostTensor<ElementE, LayoutE> matrix_E;
  cutlass::HostTensor<ElementE, ReorderedLayoutE> matrix_E_reordered;

  cutlass::gemm::GemmCoord problem_size;
  float alpha, beta;

  //
  // Methods
  //

  /// Allocates workspace in device memory
  SparseTestbed(int m, int n, int k, float alpha_ = float(1), float beta_ = float(0))
      : problem_size(m, n, k), alpha(alpha_), beta(beta_) {
    matrix_A.reset(cutlass::make_Coord(m, k / Sparse));
    matrix_A_uncompressed.reset(cutlass::make_Coord(m, k));
    matrix_B.reset(cutlass::make_Coord(k, n));
    matrix_C_computed.reset(cutlass::make_Coord(m, n));
    matrix_C_reference.reset(cutlass::make_Coord(m, n), false);
    matrix_E.reset(cutlass::make_Coord(m, k / Sparse / ElementsPerElementE));
    matrix_E_reordered.reset(
        cutlass::make_Coord(m, k / Sparse / ElementsPerElementE));
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    return true;
  }

  /// Runs the test
  bool run(
      dim3 grid, dim3 block,
      cutlass::Distribution::Kind init_A = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_E = cutlass::Distribution::Uniform) {

    // Waive the test
    if (!sufficient()) {
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

    if (init_E == cutlass::Distribution::Uniform) {
      uint64_t seed = 7;
      cutlass::reference::host::TensorFillRandomSparseMeta(
          matrix_E.host_view(), seed, MetaSizeInBits);
    } else if (init_E == cutlass::Distribution::Identity) {
      uint32_t content = (MaxID2 == 1) ? 0x44444444 : 0x4444;
      cutlass::reference::host::TensorFill(matrix_E.host_view(),
                                           (ElementE)(content));
    } else {
      return false;
    }

    cutlass::reorder_meta(matrix_E_reordered.host_ref(), matrix_E.host_ref(),
                          {problem_size.m(), problem_size.n(),
                           problem_size.k() / Sparse / ElementsPerElementE});

    matrix_A.sync_device();
    matrix_B.sync_device();
    matrix_C_computed.sync_device();
    matrix_E_reordered.sync_device();

    typename IteratorA::Params params_A(matrix_A.layout());
    typename IteratorB::Params params_B(matrix_B.layout());
    typename IteratorE::Params params_E(matrix_E_reordered.layout());

    cudaError_t result;

    int smem_size = int(sizeof(typename Mma::SharedStorage));
    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(
          test::gemm::threadblock::kernel_multistage_mma_sparse<Mma>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

      if (result != cudaSuccess) {
          return true;
      }

      result = cudaFuncSetAttribute(
          test::gemm::threadblock::kernel_multistage_mma_sparse<Mma>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100);

      if (result != cudaSuccess) {
          return true;
      }
    }

    test::gemm::threadblock::kernel_multistage_mma_sparse<Mma>
        <<<grid, block, smem_size, 0>>>(
            problem_size, params_A, matrix_A.device_ref(), params_B,
            matrix_B.device_ref(), matrix_C_computed.device_data(),
            matrix_C_computed.layout().stride(0), params_E,
            matrix_E_reordered.device_ref());

    //
    // Check error code
    //

    result = cudaDeviceSynchronize();
    EXPECT_EQ(result, cudaSuccess)
        << " kernel error: " << cudaGetErrorString(result);

    matrix_C_computed.sync_host();

    cutlass::uncompress(matrix_A_uncompressed.host_ref(), matrix_A.host_ref(),
                        matrix_E.host_ref(), problem_size.m(),
                        problem_size.k());

    cutlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                   ElementC, LayoutC, ElementC, ElementC>
        reference_gemm;

    reference_gemm(problem_size, ElementC(alpha),
                   matrix_A_uncompressed.host_view(), matrix_B.host_view(),
                   ElementC(beta), matrix_C_reference.host_view());

    bool passed = cutlass::reference::host::TensorEquals(
        matrix_C_computed.host_view(), matrix_C_reference.host_view());

    EXPECT_TRUE(passed);

    if (!passed && CUTLASS_TEST_UNIT_ENABLE_WARNINGS) {

      std::cout
        << __FILE__ << ":" << __LINE__ << "  "
        << "A:\n" << matrix_A.host_view() << "\n"
        << "B:\n" << matrix_B.host_view() << "\n"
        << "E:\n" << matrix_E.host_view() << "\n"
        << "Reference:\n"
        << matrix_C_reference.host_view() << "\n"
        << "Computed:\n"
        << matrix_C_computed.host_view() << "\n";
    }

    EXPECT_GT(cutlass::reference::host::TensorNorm(matrix_C_reference.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(matrix_C_computed.host_view()), 0);

    return passed;
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace test
