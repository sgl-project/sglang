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

#include "testbed_gemv.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_batched_gemv, 1x64x64x1_1x64x4x1_1x4x4x1_rcr_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 1);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 1;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x64x64x4_1x64x4x2_1x4x4x2_rcr_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 4);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 2;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x256x256x64_1x64x4x8_1x4x4x8_rcr_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 256, 256, 64);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 8;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x7x256x4096_1x8x4x64_1x1x4x64_rcr_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 7, 256, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 4>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x64x27x4096_1x8x1x64_1x1x1x64_rcr_alpha_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 27, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 1>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size, -0.5f);
}

TEST(SM50_batched_gemv, 1x64x27x4096_1x8x1x64_1x1x1x64_rcr_alpha_beta_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 27, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 1>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size, 4.5f, -0.5f);
}

TEST(SM50_batched_gemv, 1x64x24x4096_1x8x4x64_1x1x4x64_rcr_alpha_beta_fp16_fp16)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 24, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 4>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, cutlass::half_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size, cutlass::half_t(4.5f), cutlass::half_t(-0.5f));
}

///

TEST(SM50_batched_gemv, 1x64x64x1_1x64x4x1_1x4x4x1_rcr_fp16_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 1);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 1;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x64x64x4_1x64x4x2_1x4x4x2_rcr_fp16_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 4);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 2;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x256x256x64_1x64x4x8_1x4x4x8_rcr_fp16_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 256, 256, 64);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 8;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x7x256x4096_1x8x4x64_1x1x4x64_rcr_fp16_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 7, 256, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 4>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

///

TEST(SM50_batched_gemv, 1x64x64x1_1x64x4x1_1x4x4x1_rcr_fp16_fp16)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 1);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 1;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, cutlass::half_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x64x64x4_1x64x4x2_1x4x4x2_rcr_fp16_fp16)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 4);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 2;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, cutlass::half_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x256x256x64_1x64x4x8_1x4x4x8_rcr_fp16_fp16)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 256, 256, 64);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 8;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, cutlass::half_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x7x256x4096_1x8x4x64_1x1x4x64_rcr_fp16_fp16)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 7, 256, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 4>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, cutlass::half_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

///

TEST(SM50_batched_gemv, 1x64x64x1_1x64x4x1_1x4x4x1_rcr_i8_i32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 1);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 1;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    int8_t, int32_t, int32_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x64x64x4_1x64x4x2_1x4x4x2_rcr_i8_i32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 4);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 2;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    int8_t, int32_t, int32_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x256x256x64_1x64x4x8_1x4x4x8_rcr_i8_i32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 256, 256, 64);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 8;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    int8_t, int32_t, int32_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x7x256x4096_1x8x4x64_1x1x4x64_rcr_i8_i32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 7, 256, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 4>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    int8_t, int32_t, int32_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    kBatchTileSize>(problem_size);
}

/////////////

TEST(SM50_batched_gemv, 1x64x64x1_1x64x4x1_1x4x4x1_crc_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 1);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 1;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x64x64x4_1x64x4x2_1x4x4x2_crc_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 4);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 2;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x256x256x64_1x64x4x8_1x4x4x8_crc_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 256, 256, 64);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 8;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x7x256x4096_1x8x4x64_1x1x4x64_crc_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 7, 256, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 4>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

///

TEST(SM50_batched_gemv, 1x64x64x1_1x64x4x1_1x4x4x1_crc_fp16_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 1);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 1;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, float,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x64x64x4_1x64x4x2_1x4x4x2_crc_fp16_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 4);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 2;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, float,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x256x256x64_1x64x4x8_1x4x4x8_crc_fp16_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 256, 256, 64);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 8;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, float,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x7x256x4096_1x8x4x64_1x1x4x64_crc_fp16_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 7, 256, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 4>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, float,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

///

TEST(SM50_batched_gemv, 1x64x64x1_1x64x4x1_1x4x4x1_crc_fp16_fp16)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 1);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 1;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, cutlass::half_t,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x64x64x4_1x64x4x2_1x4x4x2_crc_fp16_fp16)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 4);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 2;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, cutlass::half_t,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x256x256x64_1x64x4x8_1x4x4x8_crc_fp16_fp16)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 256, 256, 64);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 8;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float,  cutlass::half_t,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x7x256x4096_1x8x4x64_1x1x4x64_crc_fp16_fp16)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 7, 256, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 4>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, cutlass::half_t,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

///

TEST(SM50_batched_gemv, 1x64x64x1_1x64x4x1_1x4x4x1_crc_i8_i32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 1);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 1;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    int8_t, int32_t, int32_t,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x64x64x4_1x64x4x2_1x4x4x2_crc_i8_i32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 4);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 2;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    int8_t, int32_t, int32_t,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x256x256x64_1x64x4x8_1x4x4x8_crc_i8_i32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 256, 256, 64);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 8;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    int8_t, int32_t, int32_t,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x7x256x4096_1x8x4x64_1x1x4x64_crc_i8_i32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 7, 256, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 4>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    int8_t, int32_t, int32_t,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x64x27x4096_1x8x1x64_1x1x1x64_crc_alpha_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 27, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 1>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size, -0.5f);
}

TEST(SM50_batched_gemv, 1x64x27x4096_1x8x1x64_1x1x1x64_crc_alpha_beta_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 27, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 1>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size, 4.5f, -0.5f);
}

TEST(SM50_batched_gemv, 1x64x24x4096_1x8x4x64_1x1x4x64_crc_alpha_beta_fp16_fp16)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 24, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 4>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, cutlass::half_t,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size, cutlass::half_t(4.5f), cutlass::half_t(-0.5f));
}

/////////////

TEST(SM50_batched_gemv, 1x64x64x1_1x64x4x1_1x4x4x1_rcc_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 1);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 1;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x64x64x4_1x64x4x2_1x4x4x2_rcc_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 4);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 2;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x256x256x64_1x64x4x8_1x4x4x8_rcc_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 256, 256, 64);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 8;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x7x256x4096_1x8x4x64_1x1x4x64_rcc_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 7, 256, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 4>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

///

TEST(SM50_batched_gemv, 1x64x64x1_1x64x4x1_1x4x4x1_rcc_fp16_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 1);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 1;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x64x64x4_1x64x4x2_1x4x4x2_rcc_fp16_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 4);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 2;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x256x256x64_1x64x4x8_1x4x4x8_rcc_fp16_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 256, 256, 64);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 8;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x7x256x4096_1x8x4x64_1x1x4x64_rcc_fp16_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 7, 256, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 4>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

///

TEST(SM50_batched_gemv, 1x64x64x1_1x64x4x1_1x4x4x1_rcc_fp16_fp16)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 1);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 1;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, cutlass::half_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x64x64x4_1x64x4x2_1x4x4x2_rcc_fp16_fp16)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 4);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 2;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, cutlass::half_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x256x256x64_1x64x4x8_1x4x4x8_rcc_fp16_fp16)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 256, 256, 64);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 8;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float,  cutlass::half_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x7x256x4096_1x8x4x64_1x1x4x64_rcc_fp16_fp16)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 7, 256, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 4>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, cutlass::half_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

///

TEST(SM50_batched_gemv, 1x64x64x1_1x64x4x1_1x4x4x1_rcc_i8_i32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 1);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 1;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    int8_t, int32_t, int32_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x64x64x4_1x64x4x2_1x4x4x2_rcc_i8_i32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 64, 4);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 2;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    int8_t, int32_t, int32_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x256x256x64_1x64x4x8_1x4x4x8_rcc_i8_i32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 256, 256, 64);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
  static int const kBatchTileSize = 8;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    int8_t, int32_t, int32_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x7x256x4096_1x8x4x64_1x1x4x64_rcc_i8_i32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 7, 256, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 4>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    int8_t, int32_t, int32_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size);
}

TEST(SM50_batched_gemv, 1x64x27x4096_1x8x1x64_1x1x1x64_rcc_alpha_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 27, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 1>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size, -0.5f);
}

TEST(SM50_batched_gemv, 1x64x27x4096_1x8x1x64_1x1x1x64_rcc_alpha_beta_fp32_fp32)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 27, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 1>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    float, float, float,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size, 4.5f, -0.5f);
}

TEST(SM50_batched_gemv, 1x64x24x4096_1x8x4x64_1x1x4x64_rcc_alpha_beta_fp16_fp16)
{
  cutlass::gemm::BatchedGemmCoord problem_size(1, 64, 24, 4096);

  using ThreadBlockShape = cutlass::gemm::GemmShape<1, 8, 4>;
  using ThreadShape = cutlass::gemm::GemmShape<1, 1, 4>;
  static int const kBatchTileSize = 64;

  test::gemm::kernel::batched_gemv_kernel_test<
                                    ThreadBlockShape,
                                    ThreadShape,
                                    cutlass::half_t, float, cutlass::half_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::ColumnMajor,
                                    kBatchTileSize>(problem_size, cutlass::half_t(4.5f), cutlass::half_t(-0.5f));
}
