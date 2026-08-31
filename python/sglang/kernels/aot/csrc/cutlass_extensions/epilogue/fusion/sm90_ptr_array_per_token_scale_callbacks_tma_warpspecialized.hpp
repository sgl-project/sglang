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
  \brief Pointer-array row-scale fusion callbacks for the sm90 TMA warp-specialized epilogue.
*/

#pragma once

#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class ElementOutput_, class ElementCompute_, class ElementScalar_ = ElementCompute_,
          int AlignmentScalar_ = 128 / cute::sizeof_bits_v<ElementScalar_>,
          FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest>
struct PtrArrayPerTokenScaledAcc
    : ScaledAcc<ElementOutput_, ElementCompute_, ElementScalar_, RoundStyle_> {
  static constexpr int AlignmentScalar = AlignmentScalar_;
};

template <class CtaTileShapeMNK, class ElementOutput, class ElementCompute,
          class ElementScalar = ElementCompute,
          int AlignmentScalar = 128 / sizeof_bits_v<ElementScalar>,
          FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>
using Sm90PtrArrayPerTokenScaledAcc =
    Sm90EVT<Sm90Compute<multiplies, ElementOutput, ElementCompute, RoundStyle>,
            Sm90RowBroadcast<0, CtaTileShapeMNK, ElementScalar*, ElementCompute, Stride<_0, _1, _0>,
                             AlignmentScalar>,
            Sm90AccFetch>;

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
          int NumEpilogueWarpGroups, class ElementOutput, class ElementCompute, class ElementScalar,
          int AlignmentScalar, FloatRoundStyle RoundStyle, class CtaTileShapeMNK,
          class EpilogueTile>
struct FusionCallbacks<
    epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC,
                                             DelayTmaStore, NumEpilogueWarpGroups>,
    fusion::PtrArrayPerTokenScaledAcc<ElementOutput, ElementCompute, ElementScalar, AlignmentScalar,
                                      RoundStyle>,
    CtaTileShapeMNK, EpilogueTile>
    : Sm90PtrArrayPerTokenScaledAcc<
          CtaTileShapeMNK, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
          ElementCompute, ElementScalar, AlignmentScalar, RoundStyle> {
  using Impl = Sm90PtrArrayPerTokenScaledAcc<
      CtaTileShapeMNK, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
      ElementCompute, ElementScalar, AlignmentScalar, RoundStyle>;

  struct Arguments {
    ElementScalar token_scale_default = ElementScalar(1);
    ElementScalar const* const* token_scale_ptr_array = nullptr;

    using StrideTokenScale = Stride<_0, _1, _0>;
    StrideTokenScale dTokenScale = {_0{}, _1{}, _0{}};

    operator typename Impl::Arguments() const {
      return {{token_scale_ptr_array, token_scale_default, dTokenScale}, {}, {}};
    }
  };

  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
