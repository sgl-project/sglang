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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed_splitk.h"

// These operators are assert(0) unless extended PTX is used. 
#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_GemmSplitKParallel_f16n_f16t_f32t_tensor_op_f32, 64x64x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

TEST(SM75_Device_GemmSplitKParallel_f16n_f16t_f32n_tensor_op_f32, 64x64x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

TEST(SM75_Device_GemmSplitKParallel_f16n_f16t_f16t_tensor_op_f32, 128x128x32_64x64x32) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

TEST(SM75_Device_GemmSplitKParallel_f16n_f16t_f16n_tensor_op_f32, 128x128x32_64x64x32) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

TEST(SM75_Device_GemmSplitKParallel_f16n_f16t_f16t_tensor_op_f16, 64x128x32_32x64x32) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;

  using Gemm = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

TEST(SM75_Device_GemmSplitKParallel_f16n_f16t_f16n_tensor_op_f16, 64x128x32_32x64x32) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;

  using Gemm = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_GemmSplitKParallel_f16t_f16n_f32t_tensor_op_f32, 128x256x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

TEST(SM75_Device_GemmSplitKParallel_f16t_f16n_f32n_tensor_op_f32, 128x256x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

TEST(SM75_Device_GemmSplitKParallel_f16t_f16n_f16t_tensor_op_f32, 128x128x32_64x64x32) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

TEST(SM75_Device_GemmSplitKParallel_f16t_f16n_f16n_tensor_op_f32, 128x128x32_64x64x32) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

TEST(SM75_Device_GemmSplitKParallel_f16t_f16n_f16t_tensor_op_f16, 64x128x32_32x64x32) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;

  using Gemm = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

TEST(SM75_Device_GemmSplitKParallel_f16t_f16n_f16n_tensor_op_f16, 64x128x32_32x64x32) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;

  using Gemm = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif
