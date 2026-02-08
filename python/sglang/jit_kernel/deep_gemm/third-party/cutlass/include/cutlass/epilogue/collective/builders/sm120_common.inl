/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::collective::detail {

///////////////////////////////////////////////////////////////////////////////

// Selects the largest vectorized smem store atom available
template <class GmemStrideTypeD, class ElementD>
constexpr auto
sm120_get_smem_store_op_for_accumulator() {
  using namespace cute;

  if constexpr (sizeof(ElementD) == 2 && size<0>(GmemStrideTypeD{}) == 1) {
    return SM90_U16x4_STSM_T{};
  }
  else if constexpr (sizeof(ElementD) == 2 && size<1>(GmemStrideTypeD{}) == 1) {
    return SM90_U32x2_STSM_N{};
  }
  else {
    // auto-vectorizing store
    return AutoVectorizingCopyWithAssumedAlignment{};
  }
}

// Selects the largest vectorized smem load atom available
template <class GmemStrideTypeC, class ElementC>
constexpr auto
sm120_get_smem_load_op_for_source() {
  using namespace cute;

  if constexpr (sizeof(ElementC) == 2) {
    if constexpr (size<0>(GmemStrideTypeC{}) == 1) {
      return SM75_U16x4_LDSM_T{};
    }
    else if constexpr (size<1>(GmemStrideTypeC{}) == 1) {
      return SM75_U32x2_LDSM_N{};
    }
  }
  else {
    // auto-vectorizing load
    return AutoVectorizingCopyWithAssumedAlignment<128>{};
  }
}

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::collective::detail
