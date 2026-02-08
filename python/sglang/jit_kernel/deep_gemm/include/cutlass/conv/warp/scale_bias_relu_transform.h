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
namespace conv {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FragmentActivations, typename FragmentScaleBias>
struct FpropScaleBiasReluTransform {

  using T = typename FragmentActivations::Element;

  static int const NumActivations = FragmentActivations::kElements;
  static int const NumScaleBias = FragmentScaleBias::kElements;
  static int const MmaElements = 2;
  // One element has one scale and one bias
  static int const MmaScaleBiasPair = 2;
  // 16816 has 2 columns
  static int const MmaCols = 2;

  using MmaOperand = Array<T, MmaElements>;
  using ScaleBiasOperand = Array<T, MmaElements * MmaScaleBiasPair>;

  CUTLASS_DEVICE
  void transform(MmaOperand &activations, ScaleBiasOperand const &scale_bias) {

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    uint32_t *ptr_activations = reinterpret_cast<uint32_t *>(&activations);
    uint32_t const *ptr_scale_bias = reinterpret_cast<uint32_t const *>(&scale_bias);

    // Apply per channel scale+bias+relu if the data is not a special NaN
    // (0x7eff).  If it is a special NaN (0x7eff), hard code the output to 0.

    // We assumes the pair of FP16 are either both inbound or both out-of-bound.
    // It requires C to be an even number.
    asm volatile(
        "{\n\t"
        " .reg .pred %%p;\n\t"
        " .reg .b32 t1;\n\t"
        " setp.eq.u32 %%p, %2, %4;\n\t"
        " fma.rn.f16x2.relu t1, %1, %2, %3;\n"
        " selp.u32 %0, 0, t1, %%p;\n\t"
        "}\n"
        : "=r"(ptr_activations[0])
        : "r"(ptr_scale_bias[0]), "r"(ptr_activations[0]),
          "r"(ptr_scale_bias[1]), "n"(cutlass::arch::OOB_NAN_F16x2));
#else
    assert(0);
#endif
  }

  CUTLASS_DEVICE
  void operator()(FragmentActivations &activations,
                  FragmentScaleBias const &scale_bias) {
    MmaOperand *ptr_activations = reinterpret_cast<MmaOperand *>(&activations);
    ScaleBiasOperand const *ptr_scale_bias =
        reinterpret_cast<ScaleBiasOperand const *>(&scale_bias);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < (NumActivations / MmaElements); ++i) {
      transform(ptr_activations[i], ptr_scale_bias[(i / MmaScaleBiasPair) % MmaCols]);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FragmentActivations, typename FragmentScaleBias>
struct WgradScaleBiasReluTransform {

  using T = typename FragmentActivations::Element;

  static int const NumActivations = FragmentActivations::kElements;
  static int const NumScaleBias = FragmentScaleBias::kElements;
  static int const MmaElements = 2;
  // One element has one scale and one bias
  static int const MmaScaleBiasPair = 2;
  // 16816 has 2 rows
  static int const MmaRows = 2;

  using MmaOperand = Array<T, MmaElements>;
  using ScaleBiasOperand = Array<__half2, MmaScaleBiasPair>;

  CUTLASS_DEVICE
  void transform(MmaOperand &activations, ScaleBiasOperand const &scale_bias) {

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

    __half2 *ptr_activations = reinterpret_cast<__half2 *>(&activations);
    uint32_t const *ptr_scale_bias = reinterpret_cast<uint32_t const *>(&scale_bias);

#if 1 
    // CUDA + PTX version

    bool h1_oob = (reinterpret_cast<uint16_t &>(ptr_activations[0].x) == cutlass::arch::OOB_NAN_F16);
    bool h2_oob = (reinterpret_cast<uint16_t &>(ptr_activations[0].y) == cutlass::arch::OOB_NAN_F16);

    // Apply per channel scale+bias+relu if the data is not a special NaN
    // (0x7eff).  If it is a special NaN (0x7eff), hard code the output to 0.

    // We cannot gurantee that the pair of F16 are both in bound or both 
    // out-of-bound because C x R x S can be an odd number.
    asm volatile(
        "{\n\t"
        " fma.rn.f16x2.relu %0, %1, %2, %3;\n"
        "}"
        : "=r"(reinterpret_cast<uint32_t &>(ptr_activations[0]))
        : "r"(ptr_scale_bias[0]), "r"(reinterpret_cast<uint32_t &>(ptr_activations[0])),
          "r"(ptr_scale_bias[1]));

    reinterpret_cast<uint32_t &>(ptr_activations[0]) = h1_oob ?
            (reinterpret_cast<uint32_t &>(ptr_activations[0]) & 0xffff0000) :
            reinterpret_cast<uint32_t &>(ptr_activations[0]);

    reinterpret_cast<uint32_t &>(ptr_activations[0]) = h2_oob ?
            (reinterpret_cast<uint32_t &>(ptr_activations[0]) & 0xffff) :
            reinterpret_cast<uint32_t &>(ptr_activations[0]);
#else
    // pure PTX version

    // Apply per channel scale+bias+relu if the data is not a special NaN
    // (0x7eff).  If it is a special NaN (0x7eff), hard code the output to 0.
    asm volatile(
        "{\n"
        " .reg .b16 t1, t2;\n"
        " .reg .b32 t3, t4, t5, t6;\n"
        " .reg .pred p1, p2;\n"
        " mov.b32 {t1, t2}, %2;\n"
        " setp.eq.s16 p1, t1, %4;\n"
        " setp.eq.s16 p2, t2, %4;\n"
        " fma.rn.f16x2.relu t3, %1, %2, %3;\n"
        " and.b32 t4, t3, %5;\n"
        " selp.b32 t5, t4, t3, p1;\n"
        " and.b32 t6, t5, %6;\n"
        " selp.b32 %0, t6, t5, p2;\n"
        "}\n"
        : "=r"(reinterpret_cast<uint32_t &>(ptr_activations[0]))
        : "r"(ptr_scale_bias[0]), "r"(reinterpret_cast<uint32_t &>(ptr_activations[0])),
          "r"(ptr_scale_bias[1]), "n"(cutlass::arch::OOB_NAN_F16), "n"(0xffff0000), "n"(0x0000ffff));
#endif
#else
    assert(0);
#endif
  }

  CUTLASS_DEVICE
  void operator()(FragmentActivations &activations,
                  FragmentScaleBias const &scale_bias) {
    MmaOperand *ptr_activations = reinterpret_cast<MmaOperand *>(&activations);
    ScaleBiasOperand const *ptr_scale_bias =
        reinterpret_cast<ScaleBiasOperand const *>(&scale_bias);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < (NumActivations / MmaElements); ++i) {
      transform(ptr_activations[i], ptr_scale_bias[(i / MmaRows)]);
    }
  }
};
} // namespace warp
} // namespace conv 
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
