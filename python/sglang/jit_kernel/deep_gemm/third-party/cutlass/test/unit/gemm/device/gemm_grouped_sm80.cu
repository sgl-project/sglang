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
    \brief Tests for device-wide GEMM interface

*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed_grouped.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Visitor class to abstract away the algorithm for iterating over tiles.
//
// This is the prototype. We will delete this when the efficient kernel is
// available.
struct GemmGroupedProblemVisitor {

  struct Params {
    cutlass::gemm::GemmCoord const *problem_sizes;
    int32_t                         problem_count;
    int64_t const                  *tile_count;
  };

  struct SharedStorage {
    //
    // Nothing for now. As an optimization step, we could consider parallel
    // argmin or prefix sums across the block.
    //
  };

  //
  // Data members
  //

  SharedStorage &shared_storage;
  Params const &params;
  cutlass::MatrixCoord threadblock_shape;

  int64_t tile_idx;
  int64_t tile_count_sum;
  int64_t problem_tile_start;
  int32_t problem_idx;

  //
  // Methods
  //
  CUTLASS_DEVICE
  GemmGroupedProblemVisitor(
    SharedStorage &shared_storage_,
    Params const &params_,
    cutlass::MatrixCoord threadblock_shape_,
    int32_t block_idx
  ):
    shared_storage(shared_storage_),
    params(params_),
    threadblock_shape(threadblock_shape_),
    tile_idx(block_idx),
    tile_count_sum(0),
    problem_idx(0)
  {

    cutlass::gemm::GemmCoord problem = params.problem_sizes[problem_idx];

    cutlass::gemm::GemmCoord  grid = grid_shape(problem);

    problem_tile_start = 0;
    tile_count_sum = grid.m() * grid.n();
  }

  /// Get the grid shape
  CUTLASS_HOST_DEVICE
  static cutlass::gemm::GemmCoord grid_shape(
    cutlass::gemm::GemmCoord const &problem,
    cutlass::MatrixCoord const & block_shape) {

    return cutlass::gemm::GemmCoord(
      ((problem.m() - 1 + block_shape.row()) / block_shape.row()),
      ((problem.n() - 1 + block_shape.column()) / block_shape.column()),
      1);
  }

  /// Get the grid shape
  CUTLASS_DEVICE
  cutlass::gemm::GemmCoord grid_shape(cutlass::gemm::GemmCoord const &problem) const {
    return grid_shape(problem, threadblock_shape);
  }

  /// Returns true if there is a tile to compute
  CUTLASS_DEVICE
  bool next_tile() {

    if (tile_idx < tile_count_sum) {
      return true;
    }

    do {
      ++problem_idx;

      if (problem_idx >= params.problem_count) {
        return false;
      }

      cutlass::gemm::GemmCoord problem = params.problem_sizes[problem_idx];
      cutlass::gemm::GemmCoord  grid = grid_shape(problem);

      int64_t tile_count = grid.m() * grid.n();

      problem_tile_start = tile_count_sum;
      tile_count_sum += tile_count;

    } while (tile_count_sum <= tile_idx);

    return true;
  }

  /// Gets the global tile index
  CUTLASS_HOST_DEVICE
  int64_t tile_index() const {
    return tile_idx;
  }

  /// Gets the index of the problem
  CUTLASS_HOST_DEVICE
  int32_t problem_index() const {
    return problem_idx;
  }

  /// Returns the problem size for the current problem
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord problem_size() const {
    return params.problem_sizes[problem_idx];
  }

  CUTLASS_HOST_DEVICE
  int64_t threadblock_idx() const {
    return tile_idx - problem_tile_start;
  }

  CUTLASS_DEVICE
  void advance(int32_t grid_size) {
    tile_idx += grid_size;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int ThreadblockShapeM, int ThreadblockShapeN>
__global__ void GroupedBatchedKernel(GemmGroupedProblemVisitor::Params params) {

  __shared__ GemmGroupedProblemVisitor::SharedStorage shared_storage;

  GemmGroupedProblemVisitor problem_visitor(
    shared_storage,
    params,
    {ThreadblockShapeM, ThreadblockShapeN},
    blockIdx.x);

  while (problem_visitor.next_tile()) {

    cutlass::gemm::GemmCoord problem_size = problem_visitor.problem_size();
    int64_t threadblock_idx                       = problem_visitor.threadblock_idx();

    cutlass::gemm::GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

    int threadblock_tile_m_idx = int(threadblock_idx / grid_shape.n());
    int threadblock_tile_n_idx = int(threadblock_idx % grid_shape.n());

    //
    // Do the MMA
    //

    if (threadIdx.x == 0) {
      #if 0
      printf("Block %d - tile: %lld, problem %d, threadblock_idx: %lld, threadblock(m: %d, n: %d)\n",
        blockIdx.x,
        static_cast<long long>(problem_visitor.tile_index()),
        problem_visitor.problem_index(),
        threadblock_idx,
        threadblock_tile_m_idx,
        threadblock_tile_n_idx);
      #endif
    }

    // Next tile
    problem_visitor.advance(gridDim.x);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGrouped_scheduler, 64x64x32_32x32x32) {

  int32_t problem_count = 16;

  int const kThreadblockShapeM = 64;
  int const kThreadblockShapeN = 64;

  std::vector<cutlass::gemm::GemmCoord> problem_sizes(problem_count);
  std::vector<int64_t> tile_counts(problem_count);

  // construct a few problems of random sizes
  srand(1921);
  for (int32_t i = 0; i < problem_count; ++i) {
    problem_sizes.at(i) = cutlass::gemm::GemmCoord(
      8 * (rand() % 48) + 64,
      8 * (rand() % 48) + 64,
      8 * (rand() % 48) + 64);
  }

  // compute prefix sum
  int64_t tile_count = 0;

  for (int32_t i = 0; i < problem_count; ++i) {

    cutlass::gemm::GemmCoord grid_shape = GemmGroupedProblemVisitor::grid_shape(
      problem_sizes.at(i), {kThreadblockShapeM, kThreadblockShapeN});

    int32_t problem_tile_count = (grid_shape.m() * grid_shape.n());

    int64_t tile_start = tile_count;

    tile_count += problem_tile_count;
    tile_counts.at(i) = tile_count;

    if (false) {
      std::cout << "Problem " << i << " size("
        << problem_sizes.at(i).m() << "-by-" << problem_sizes.at(i).n()
        << ") - tiles: " << problem_tile_count << ",  grid(" << grid_shape.m() << ", " << grid_shape.n()
        << "), tiles[" << tile_start << ", " << tile_count << ")" << std::endl;
    }
  }

  // Copy to device memory
  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device(problem_count);
  cutlass::DeviceAllocation<int64_t>                  tile_counts_device(problem_count);

  problem_sizes_device.copy_from_host(problem_sizes.data());
  tile_counts_device.copy_from_host(tile_counts.data());

  GemmGroupedProblemVisitor::Params params;
  params.problem_sizes = problem_sizes_device.get();
  params.problem_count = problem_count;
  params.tile_count = tile_counts_device.get();

  // Launch the kernel
  dim3 grid(108, 1, 1);
  dim3 block(128, 1, 1);

  GroupedBatchedKernel<kThreadblockShapeM, kThreadblockShapeN><<< grid, block >>>(params);

  // wait
  cudaDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGrouped_f16n_f16t_f32n_tensor_op_f32, 128x128x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    8,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    8,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(24);
  EXPECT_TRUE(passed);

}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGrouped_f16n_f16t_f32t_tensor_op_f32, 128x128x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    8,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    8,
    ElementOutput, cutlass::layout::RowMajor,    // row major
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(24);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGrouped_f16t_f16n_f32n_tensor_op_f32, 128x64x32_64x32x32) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    8,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    8,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGrouped_f16t_f16n_f32t_tensor_op_f32, 128x64x32_64x32x32) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    8,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    8,
    ElementOutput, cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGrouped_f64t_f64t_f64n_tensor_op_f64, 64x64x16_32x32x16) {

  using ElementInput = double;
  using ElementOutput = double;
  using ElementAccumulator = double;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    ElementInput,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 1,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGrouped_f32t_f32t_f32n_simt_f32, 128x128x8_64x32x1) {

  using ElementInput = float;
  using ElementOutput = float;
  using ElementAccumulator = float;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    ElementInput,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<64, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 1,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGrouped_f32t_f32t_f32t_simt_f32, 128x128x8_64x32x1) {

  using ElementInput = float;
  using ElementOutput = float;
  using ElementAccumulator = float;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    ElementInput,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    ElementOutput, cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<64, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 1,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGrouped_f32t_f32t_f32n_simt_f32, 128x64x8_64x32x1) {

  using ElementInput = float;
  using ElementOutput = float;
  using ElementAccumulator = float;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    ElementInput,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 8>,
    cutlass::gemm::GemmShape<64, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 1,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGrouped_f32t_f32t_f32t_simt_f32, 128x64x8_64x32x1) {

  using ElementInput = float;
  using ElementOutput = float;
  using ElementAccumulator = float;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    ElementInput,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    ElementOutput, cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 8>,
    cutlass::gemm::GemmShape<64, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 1,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGrouped_cf32n_cf32n_cf32n_tensorop_f32, 64x64x16_32x32x16) {

  using ElementInput = cutlass::complex<float>;
  using ElementOutput = cutlass::complex<float>;
  using ElementAccumulator = cutlass::complex<float>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    1,
    ElementInput,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    1,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 1,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
    cutlass::arch::OpMultiplyAddComplex>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGrouped_cf32c_cf32t_cf32n_tensorop_f32, 64x64x16_32x32x16) {

  using ElementInput = cutlass::complex<float>;
  using ElementOutput = cutlass::complex<float>;
  using ElementAccumulator = cutlass::complex<float>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kConjugate,
    1,
    ElementInput,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kConjugate,
    1,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 1,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
    cutlass::arch::OpMultiplyAddComplex>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGrouped_cf32c_cf32t_cf32t_tensorop_f32, 64x64x16_32x32x16) {

  using ElementInput = cutlass::complex<float>;
  using ElementOutput = cutlass::complex<float>;
  using ElementAccumulator = cutlass::complex<float>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kConjugate,
    1,
    ElementInput,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kConjugate,
    1,
    ElementOutput, cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 1,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
    cutlass::arch::OpMultiplyAddComplex>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGrouped_cf32t_cf32h_cf32n_tensorop_f32, 64x64x16_16x16x16) {

  using ElementInput = cutlass::complex<double>;
  using ElementOutput = cutlass::complex<double>;
  using ElementAccumulator = cutlass::complex<double>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    ElementInput,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kConjugate,
    1,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 1,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
    cutlass::arch::OpMultiplyAddComplex>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif // #if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
