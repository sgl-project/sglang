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
#include "cutlass/gemm/device/gemm.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed.h"

#define N cutlass::layout::ColumnMajor
#define T cutlass::layout::RowMajor

#define RUN_GEMM(X, Y)                     \
  using ElementOutput = int8_t;            \
  using ElementAccumulator = int32_t;      \
  using ElementCompute = float;            \
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 4>;         \
  using Gemm = cutlass::gemm::device::Gemm<                           \
    int8_t,                                                           \
    X,                                                                \
    int8_t,                                                           \
    Y,                                                                \
    ElementOutput,                                                    \
    cutlass::layout::RowMajor,                                        \
    int32_t,                                                          \
    cutlass::arch::OpClassSimt,                                       \
    cutlass::arch::Sm61,                                              \
    ThreadBlockShape,                                                 \
    WarpShape,                                                        \
    InstructionShape,                                                 \
    cutlass::epilogue::thread::LinearCombinationClamp<                \
      ElementOutput,                                                  \
      1,                                                              \
      ElementAccumulator,                                             \
      ElementCompute                                                  \
    >,                                                                \
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,       \
    2                                                                 \
  >;                                                                  \
  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());

////////////////////////////////////////////////////////////////////////////////

TEST(SM61_Device_Gemm_s8n_s8t_simt_op_dp4a, 64x64x16_64x64x4) {
  using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  RUN_GEMM(N, T)
}

TEST(SM61_Device_Gemm_s8n_s8t_simt_op_dp4a, 256x128x64_64x64x4) {
  using ThreadBlockShape = cutlass::gemm::GemmShape<256, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  RUN_GEMM(N, T)
}

TEST(SM61_Device_Gemm_s8n_s8t_simt_op_dp4a, 256x256x16_128x64x4) {
  using ThreadBlockShape = cutlass::gemm::GemmShape<256, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 64, 16>;
  RUN_GEMM(N, T)
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM61_Device_Gemm_s8t_s8n_simt_op_dp4a, 64x64x16_64x64x4) {
  using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  RUN_GEMM(T, N)
}

TEST(SM61_Device_Gemm_s8t_s8n_simt_op_dp4a, 256x128x64_64x64x4) {
  using ThreadBlockShape = cutlass::gemm::GemmShape<256, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  RUN_GEMM(T, N)
}

TEST(SM61_Device_Gemm_s8t_s8n_simt_op_dp4a, 256x256x16_128x64x4) {
  using ThreadBlockShape = cutlass::gemm::GemmShape<256, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 64, 16>;
  RUN_GEMM(T, N)
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM61_Device_Gemm_s8n_s8n_simt_op_dp4a, 64x64x16_64x64x4) {
  using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  RUN_GEMM(N, N)
}

TEST(SM61_Device_Gemm_s8n_s8n_simt_op_dp4a, 256x128x64_64x64x4) {
  using ThreadBlockShape = cutlass::gemm::GemmShape<256, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  RUN_GEMM(N, N)
}

TEST(SM61_Device_Gemm_s8n_s8n_simt_op_dp4a, 256x256x16_128x64x4) {
  using ThreadBlockShape = cutlass::gemm::GemmShape<256, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 64, 16>;
  RUN_GEMM(N, N)
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM61_Device_Gemm_s8t_s8t_simt_op_dp4a, 64x64x16_64x64x4) {
  using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  RUN_GEMM(T, T)
}

TEST(SM61_Device_Gemm_s8t_s8t_simt_op_dp4a, 256x128x64_64x64x4) {
  using ThreadBlockShape = cutlass::gemm::GemmShape<256, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  RUN_GEMM(T, T)
}

TEST(SM61_Device_Gemm_s8t_s8t_simt_op_dp4a, 256x256x16_128x64x4) {
  using ThreadBlockShape = cutlass::gemm::GemmShape<256, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 64, 16>;
  RUN_GEMM(T, T)
}
