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
    \brief Templates implementing warp-level per channel scale+bias+relu before
   matrix multiply-accumulate operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/platform/platform.h"

#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/mma_sm75.h" 
#include "cutlass/arch/mma_sm80.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FragmentActivations, typename FragmentVarMean, typename FragmentGammaBeta>
struct LayernormScaleBiasTransform {

  using T = typename FragmentActivations::Element;

  static int const NumActivations = FragmentActivations::kElements;
  static int const NumVarMean = FragmentVarMean::kElements;
  static int const NumGammaBeta = FragmentGammaBeta::kElements;
  static int const MmaElements = 2;
  // One element has one scale and one bias
  static int const MmaScaleBiasPair = 2;
  // 16816 has 2 columns and 2 rows
  static int const MmaCols = 2;
  static int const MmaRows = 2;

  using MmaOperand = Array<T, MmaElements>;
  using VarMeanOperand = Array<__half2, MmaScaleBiasPair>;
  using GammaBetaOperand = Array<T, MmaElements * MmaScaleBiasPair>;

  CUTLASS_DEVICE
  void transform(MmaOperand &activations,
                 VarMeanOperand const &var_mean,
                 GammaBetaOperand const &gamma_beta) {

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    uint32_t *ptr_activations = reinterpret_cast<uint32_t *>(&activations);
    uint32_t const *ptr_var_mean = reinterpret_cast<uint32_t const *>(&var_mean);
    uint32_t const *ptr_gamma_beta = reinterpret_cast<uint32_t const *>(&gamma_beta);

    // Apply per channel scale+bias+relu if the data is not a special NaN
    // (0x7eff).  If it is a special NaN (0x7eff), hard code the output to 0.

    // We assumes the pair of FP16 are either both inbound or both out-of-bound.
    // It requires C to be an even number.
    asm volatile(
        "{\n\t"
        " fma.rn.f16x2 %0, %1, %2, %3;\n"
        " fma.rn.f16x2 %0, %4, %0, %5;\n"
        "}\n"
        : "=r"(ptr_activations[0])
        : "r"(ptr_var_mean[0]), "r"(ptr_activations[0]),
          "r"(ptr_var_mean[1]),
          "r"(ptr_gamma_beta[0]), "r"(ptr_gamma_beta[1]));
#else
    assert(0);
#endif
  }

  CUTLASS_DEVICE
  void operator()(FragmentActivations &activations,
                  FragmentVarMean const &var_mean,
                  FragmentGammaBeta const &gamma_beta) {
    MmaOperand *ptr_activations = reinterpret_cast<MmaOperand *>(&activations);
    VarMeanOperand const *ptr_var_mean =
        reinterpret_cast<VarMeanOperand const *>(&var_mean);
    GammaBetaOperand const *ptr_gamma_beta =
        reinterpret_cast<GammaBetaOperand const *>(&gamma_beta);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < (NumActivations / MmaElements); ++i) {
      transform(ptr_activations[i],
                ptr_var_mean[i / (MmaCols * MmaRows) * MmaRows + i % MmaRows],
                ptr_gamma_beta[(i / MmaScaleBiasPair) % MmaCols]);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm 
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
