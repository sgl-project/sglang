/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cute/atom/mma_traits_sm90_gmma.hpp"                       // cute::GMMA::Major
#include "cutlass/arch/config.h"                                    // CUTLASS_ARCH_MMA_SM90_SUPPORTED
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"      // StructuredSparseCompressor
#include "cutlass/transform/device/transform_universal_adapter.hpp" // TransformUniversalAdapter
#include "cutlass/gemm/collective/builders/sm90_common.inl"         // gmma_ss_tag_to_major_A
#include "cutlass/gemm/collective/builders/sm90_sparse_config.inl"  // Sm90GemmSparseConfig
#include "testbed_sparse_gemm_compressor.hpp"                       // TestbedSparseGemmCompressor

///////////////////////////////////////////////////////////////////////////////////////////////////
// * Test Plan
// ElementA : fp8
// LayoutA : row / col
// Gemm : 1x 2x 3x multiplier of alignment requirement. corner case that smaller than alignment requirement
///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

TEST(SM90_Structured_Sparse_Gemm_Compressor_Device, f8_t)
{
  // Test Settings
  using ElementA = cutlass::float_e4m3_t;
  using LayoutATag = cutlass::layout::RowMajor;

  // Deduct From Test Setting
  static constexpr cute::GMMA::Major GmmaMajorA = cutlass::gemm::collective::detail::gmma_rs_tag_to_major_A<LayoutATag>();
  using ElementAMma = cute::sparse_elem<2, ElementA>;
  using ElementEMma = cute::sparse_elem<8, uint8_t>;

  using SparseConfig = cutlass::Sm90GemmSparseConfig<ElementAMma, GmmaMajorA, ElementEMma, cute::Int<64>>;

  using CompressorKernel = cutlass::transform::kernel::
      StructuredSparseCompressor<cute::Shape<int, int, int, int>, ElementA, LayoutATag, SparseConfig, cutlass::arch::Sm90>;

  using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

  // Test Bed
  test::transform::device::TestbedSparseGemmCompressor<Compressor> testbed;
  EXPECT_TRUE(testbed.run_auto());
}

TEST(SM90_Structured_Sparse_Gemm_Compressor_Device, f8_n)
{
  // Test Settings
  using ElementA = cutlass::float_e5m2_t;
  using LayoutATag = cutlass::layout::ColumnMajor;

  // Deduct From Test Setting
  static constexpr cute::GMMA::Major GmmaMajorA = cutlass::gemm::collective::detail::gmma_rs_tag_to_major_A<LayoutATag>();
  using ElementAMma = cute::sparse_elem<2, ElementA>;
  using ElementEMma = cute::sparse_elem<8, uint8_t>;

  using SparseConfig = cutlass::Sm90GemmSparseConfig<ElementAMma, GmmaMajorA, ElementEMma, cute::Int<64>>;

  using CompressorKernel = cutlass::transform::kernel::
      StructuredSparseCompressor<cute::Shape<int, int, int, int>, ElementA, LayoutATag, SparseConfig, cutlass::arch::Sm90>;

  using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

  // Test Bed
  test::transform::device::TestbedSparseGemmCompressor<Compressor> testbed;
  EXPECT_TRUE(testbed.run_auto());
}

#endif // #if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
