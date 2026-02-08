/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief Functor performing pipelined epilogues with bias add and elementwise activation functions.
         This collective is now DEPRECATED, will be removed in the next release. Use EVT instead.
*/

#pragma once

#include "sm90_epilogue_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  class BlockTileShape_,    //     (BLK_M,BLK_N,BLK_K)
  class EpilogueTileShape_, // (EPI_TILE_M,EPI_TILE_N)
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class FusionCallbacks_,
  class CopyOpG2S_,
  class SmemLayoutAtomC_,
  class CopyOpS2R_,
  class CopyOpS2G_,
  class SmemLayoutAtomD_,
  class CopyOpR2S_,
  class CopyAtomC_,
  class CopyOpR2R_
>
class Sm90EpilogueTmaWarpSpecializedBiasElementwise
  : public CollectiveEpilogue<
      Sm90TmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, false, false>,
      BlockTileShape_,
      EpilogueTileShape_,
      ElementC_,
      StrideC_,
      ElementD_,
      StrideD_,
      FusionCallbacks_,
      CopyOpG2S_,
      SmemLayoutAtomC_,
      CopyOpS2R_,
      CopyOpS2G_,
      SmemLayoutAtomD_,
      CopyOpR2S_,
      CopyAtomC_,
      CopyOpR2R_
> {
private:
  using Impl =
    CollectiveEpilogue<
      Sm90TmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, false, false>,
      BlockTileShape_,
      EpilogueTileShape_,
      ElementC_,
      StrideC_,
      ElementD_,
      StrideD_,
      FusionCallbacks_,
      CopyOpG2S_,
      SmemLayoutAtomC_,
      CopyOpS2R_,
      CopyOpS2G_,
      SmemLayoutAtomD_,
      CopyOpR2S_,
      CopyAtomC_,
      CopyOpR2R_
    >;
public:
  using DispatchPolicy = Sm90TmaWarpSpecializedBiasElementwise<StagesC_, StagesD_, FragmentSize_>;
  using ElementCompute = typename Impl::ThreadEpilogueOp::ElementCompute;
  using ElementBias = typename Impl::ThreadEpilogueOp::ElementBias;
  using ElementT = typename Impl::ThreadEpilogueOp::ElementAux;

  // Constructor inheritance
  using Impl::Impl;

  // Host side epilogue arguments
  struct [[deprecated("use Sm90TmaWarpSpecialized Arguments instead")]]
  Arguments {
    struct ThreadArgs {
      ElementCompute alpha{1};
      ElementCompute beta{0};
      ElementCompute const *alpha_ptr{nullptr};
      ElementCompute const *beta_ptr{nullptr};
    } thread;
    ElementC_ const* ptr_C{nullptr};
    StrideC_ dC{};
    ElementD_* ptr_D{nullptr};
    StrideD_ dD{};
    ElementBias const* ptr_Bias{nullptr};
    ElementT* ptr_T{nullptr};

    CUTLASS_HOST_DEVICE
    operator typename Impl::Arguments() const {
      typename Impl::Arguments arguments;
      arguments.thread.alpha = thread.alpha;
      arguments.thread.beta = thread.beta;
      arguments.thread.alpha_ptr = thread.alpha_ptr;
      arguments.thread.beta_ptr = thread.beta_ptr;
      if constexpr (not cute::is_void_v<ElementBias>) {
        arguments.thread.bias_ptr = ptr_Bias;
      }
      if constexpr (not cute::is_void_v<ElementT>) {
        arguments.thread.aux_ptr = ptr_T;
        arguments.thread.dAux = dD;
      }
      arguments.ptr_C = ptr_C;
      arguments.dC = dC;
      arguments.ptr_D = ptr_D;
      arguments.dD = dD;

      return arguments;
    }
  };

};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
