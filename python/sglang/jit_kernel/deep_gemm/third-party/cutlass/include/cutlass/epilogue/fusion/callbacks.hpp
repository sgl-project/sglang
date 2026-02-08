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

#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Dispatch interface for epilogue fusion callbacks
// For visitor fusions, this is just a convenience wrapper to provide metadata and non-nested args.
// It is also valid to just pass visitor callbacks directly to the collective, e.g. fusion::Sm90LinearCombination,
// provided the collective supports a visitor callbacks interface. This is useful for implementing custom fusions.
template <
  class DispatchPolicy,  // specialize on collective's dispatch policy since callbacks API will depend on collective's algorithm
  class Operation,       // the fusion operation being performed, e.g. fusion::LinearCombination
  class CtaTile_MNK,     // computed tile per CTA
  class EpilogueTile_MN, // epilogue subtile size
  class... Args          // callbacks implementation dependent args (e.g. copy atoms, smem layouts)
>
struct FusionCallbacks {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy, Operation>, "Could not find a callbacks specialization.");
};

// Metadata helper to handle custom EVTs or other non-FusionCallbacks types
template <class T>
struct FusionCallbacksTraits {
  using DispatchPolicy = void;
  using Callbacks = T;
  using Operation = FusionOperation;
  using CtaTile_MNK = void;
  using EpilogueTile_MN = void;
  using ElementCompute = void;
};

template <
  class DispatchPolicy_,
  class Operation_,
  class CtaTile_MNK_,
  class EpilogueTile_MN_,
  class... Args
>
struct FusionCallbacksTraits<
  FusionCallbacks<DispatchPolicy_, Operation_, CtaTile_MNK_, EpilogueTile_MN_, Args...>
> {
  using DispatchPolicy = DispatchPolicy_;
  using Callbacks = FusionCallbacks<DispatchPolicy_, Operation_, CtaTile_MNK_, EpilogueTile_MN_, Args...>;
  using Operation = Operation_;
  using CtaTile_MNK = CtaTile_MNK_;
  using EpilogueTile_MN = EpilogueTile_MN_;
  using ElementCompute = typename Operation::ElementCompute;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
