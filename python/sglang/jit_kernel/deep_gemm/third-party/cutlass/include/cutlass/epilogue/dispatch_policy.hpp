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
#pragma once

#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"

//////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue {

//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
//
// Builder Epilogue Schedules
//
//////////////////////////////////////////////////////////////////////////////
// Pre-Hopper schedules
struct PtrArrayDefault {};
struct EpilogueSimtVectorized {};
struct EpiloguePtrArraySimtVectorized {};
// Hopper direct store schedules
struct NoSmemWarpSpecialized {};
struct PtrArrayNoSmemWarpSpecialized {};
struct PtrArrayNoSmemWarpSpecializedTransposed {};
// Hopper TMA schedules
struct TmaWarpSpecialized {};
struct TmaWarpSpecializedCooperative {};
struct PtrArrayTmaWarpSpecialized { static constexpr int NumEpilogueWarpGroups = 1; };
struct PtrArrayTmaWarpSpecializedPingpong { static constexpr int NumEpilogueWarpGroups = 2; };
struct PtrArrayTmaWarpSpecializedCooperative { static constexpr int NumEpilogueWarpGroups = 2; };
// Blackwell direct store schedules
struct NoSmemWarpSpecialized1Sm {};
struct NoSmemWarpSpecialized2Sm {};
struct FastF32NoSmemWarpSpecialized1Sm : NoSmemWarpSpecialized1Sm {};
struct FastF32NoSmemWarpSpecialized2Sm : NoSmemWarpSpecialized2Sm {};
struct BlockwiseNoSmemWarpSpecialized1Sm : NoSmemWarpSpecialized1Sm {};
struct BlockwiseNoSmemWarpSpecialized2Sm : NoSmemWarpSpecialized2Sm {};
struct PtrArrayNoSmemWarpSpecialized1Sm : NoSmemWarpSpecialized1Sm {};
struct PtrArrayNoSmemWarpSpecialized2Sm : NoSmemWarpSpecialized2Sm {};
struct PtrArrayFastF32NoSmemWarpSpecialized1Sm : PtrArrayNoSmemWarpSpecialized1Sm {};
struct PtrArrayFastF32NoSmemWarpSpecialized2Sm : PtrArrayNoSmemWarpSpecialized2Sm {};
struct PtrArrayBlockwiseNoSmemWarpSpecialized1Sm : PtrArrayNoSmemWarpSpecialized1Sm {};
struct PtrArrayBlockwiseNoSmemWarpSpecialized2Sm : PtrArrayNoSmemWarpSpecialized2Sm {};
// Blackwell TMA schedules 
struct TmaWarpSpecialized1Sm {};
struct TmaWarpSpecialized2Sm {};
struct PtrArrayTmaWarpSpecialized1Sm : TmaWarpSpecialized1Sm {};
struct PtrArrayTmaWarpSpecialized2Sm : TmaWarpSpecialized2Sm {};
struct TmaWarpSpecialized1SmNvf4     final : TmaWarpSpecialized1Sm {};
struct TmaWarpSpecialized2SmNvf4     final : TmaWarpSpecialized2Sm {};
struct TmaWarpSpecialized1SmMxf4     final : TmaWarpSpecialized1Sm {};
struct TmaWarpSpecialized2SmMxf4     final : TmaWarpSpecialized2Sm {};
struct TmaWarpSpecialized1SmMxf8f6f4 final : TmaWarpSpecialized1Sm {};
struct TmaWarpSpecialized2SmMxf8f6f4 final : TmaWarpSpecialized2Sm {};
// Cooperative epilogue schedule for sm120 sparse kernels
struct SparseTmaWarpSpecializedCooperativeSm120 : public TmaWarpSpecializedCooperative {};

// DEPRECATED schedules, will be removed in next release
struct TmaWarpSpecializedElementwiseBase : public TmaWarpSpecialized {};
struct TmaWarpSpecializedCooperativeElementwiseBase : public TmaWarpSpecializedCooperative {};
template <
  template <class T> class ActivationFunctor_,
  thread::ScaleType::Kind Scale_ = thread::ScaleType::Default,
  FloatRoundStyle Round_ = FloatRoundStyle::round_to_nearest
>
struct [[deprecated("Use TmaWarpSpecialized with fusion::LinCombEltAct instead")]]
TmaWarpSpecializedElementwise : public TmaWarpSpecializedElementwiseBase {
  template <class T>
  using ActivationFunctor = ActivationFunctor_<T>;
  static constexpr thread::ScaleType::Kind Scale = Scale_;
  static constexpr FloatRoundStyle Round = Round_;
};

template <
  template <class T> class ActivationFunctor_,
  thread::ScaleType::Kind Scale_ = thread::ScaleType::Default,
  FloatRoundStyle Round_ = FloatRoundStyle::round_to_nearest
>
struct [[deprecated("Use TmaWarpSpecializedCooperative with fusion::LinCombEltAct instead")]]
TmaWarpSpecializedCooperativeElementwise : public TmaWarpSpecializedCooperativeElementwiseBase {
  template <class T>
  using ActivationFunctor = ActivationFunctor_<T>;
  static constexpr thread::ScaleType::Kind Scale = Scale_;
  static constexpr FloatRoundStyle Round = Round_;
};

struct TmaWarpSpecializedBiasElementwiseBase : public TmaWarpSpecialized{};
struct TmaWarpSpecializedCooperativeBiasElementwiseBase : public TmaWarpSpecializedCooperative {};

template <
  template <class T> class ActivationFunctor_,
  class ElementT_,
  template <class T> class BiasOp_,
  bool StoreT_,
  class ElementBias_
>
struct [[deprecated("Use TmaWarpSpecialized with fusion::LinCombPerRowBiasEltActAux instead")]]
TmaWarpSpecializedBiasElementwise : public TmaWarpSpecializedBiasElementwiseBase {
  template <class T>
  using ActivationFunctor = ActivationFunctor_<T>;
  using ElementT = ElementT_;

  template <class T>
  using BiasOp = BiasOp_<T>;

  static constexpr bool StoreT = StoreT_;
  using ElementBias = ElementBias_;
};

template <
  template <class T> class ActivationFunctor_,
  class ElementT_,
  template <class T> class BiasOp_,
  bool StoreT_,
  class ElementBias_
>
struct [[deprecated("Use TmaWarpSpecializedCooperative with fusion::LinCombPerRowBiasEltActAux instead")]]
TmaWarpSpecializedCooperativeBiasElementwise : public TmaWarpSpecializedCooperativeBiasElementwiseBase {
  template <class T>
  using ActivationFunctor = ActivationFunctor_<T>;

  using ElementT = ElementT_;

  template <class T>
  using BiasOp = BiasOp_<T>;

  static constexpr bool StoreT = StoreT_;
  using ElementBias = ElementBias_;
};

//////////////////////////////////////////////////////////////////////////////
//
// Collective Dispatch Policies
//
//////////////////////////////////////////////////////////////////////////////

template<
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  bool ReuseSmemC_,
  bool DelayTmaStore_
>
struct Sm90TmaWarpSpecialized {
  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  constexpr static int FragmentSize = FragmentSize_;
  constexpr static bool ReuseSmemC = ReuseSmemC_;
  constexpr static bool DelayTmaStore = DelayTmaStore_;
};

template<
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  bool ReuseSmemC_,
  bool DelayTmaStore_,
  int NumEpilogueWarpGroups_
>
struct Sm90PtrArrayTmaWarpSpecialized {
  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  constexpr static int FragmentSize = FragmentSize_;
  constexpr static bool ReuseSmemC = ReuseSmemC_;
  constexpr static bool DelayTmaStore = DelayTmaStore_;
  constexpr static int NumEpilogueWarpGroups = NumEpilogueWarpGroups_;
};

// DEPRECATED policies, will be removed in next release
template<
  int StagesC_,
  int StagesD_,
  int FragmentSize_ = 2
>
struct Sm90TmaWarpSpecializedBiasElementwise {
  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  constexpr static int FragmentSize = FragmentSize_;
};


template<
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  bool ReuseSmemC_,
  bool DelayTmaStore_
>
struct Sm100TmaWarpSpecialized {
  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  constexpr static int FragmentSize = FragmentSize_;
  constexpr static bool ReuseSmemC = ReuseSmemC_;
  constexpr static bool DelayTmaStore = DelayTmaStore_;
};

template<
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  bool ReuseSmemC_,
  bool DelayTmaStore_
>
struct Sm100PtrArrayTmaWarpSpecialized {
  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  constexpr static int FragmentSize = FragmentSize_;
  constexpr static bool ReuseSmemC = ReuseSmemC_;
  constexpr static bool DelayTmaStore = DelayTmaStore_;

  static_assert(StagesC >= 1, "StagesC must be >= 1");
  static_assert(StagesD >= 1, "StagesD must be >= 1");
};

struct Sm100NoSmem {
  constexpr static int StagesC = 1;
  constexpr static int StagesD = 1;
  constexpr static int FragmentSize = 1;
};

struct Sm100NoSmemWarpSpecialized {
  constexpr static int StagesC = 1;
  constexpr static int StagesD = 1;
  constexpr static int FragmentSize = 1;
};

struct Sm100PtrArrayNoSmem {
  constexpr static int StagesC = 1;
  constexpr static int StagesD = 1;
  constexpr static int FragmentSize = 1;
};

struct Sm100PtrArrayNoSmemWarpSpecialized {
  constexpr static int StagesC = 1;
  constexpr static int StagesD = 1;
  constexpr static int FragmentSize = 1;
};
template<
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  bool ReuseSmemC_,
  bool DelayTmaStore_
>
struct Sm120TmaWarpSpecialized {
  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  constexpr static int FragmentSize = FragmentSize_;
  constexpr static bool ReuseSmemC = ReuseSmemC_;
  constexpr static bool DelayTmaStore = DelayTmaStore_;
};

template<
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  bool ReuseSmemC_,
  bool DelayTmaStore_,
  int NumEpilogueWarpGroups_
>
struct Sm120PtrArrayTmaWarpSpecialized {
  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  constexpr static int FragmentSize = FragmentSize_;
  constexpr static bool ReuseSmemC = ReuseSmemC_;
  constexpr static bool DelayTmaStore = DelayTmaStore_;
  constexpr static int NumEpilogueWarpGroups = NumEpilogueWarpGroups_;
};

//////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue
