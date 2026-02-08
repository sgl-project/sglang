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

namespace cutlass::gemm::kernel::detail {

////////////////////////////////////////////////////////////////////////////////

enum class RasterOrder {
  AlongM,
  AlongN
};

enum class RasterOrderOptions {
  Heuristic,
  AlongM,
  AlongN
};

////////////////////////////////////////////////////////////////////////////////

// Strategies for computing reductions between CTAs computing portions of a given output tile
enum class ReductionMode {
  // Participating CTAs perform reduction in a turnstile fashion in order of the K extent
  // covered by each CTA. This requires a lock to be held exclusively by the CTA that is
  // currently accumulating.
  //
  // Turnstile accumulation ensures deterministic numeric behavior when using this mode.
  Deterministic,

  // Participating CTAs perform reduction atomically to the same workspace (mostly) without locking.
  // Locks are used only to wait for the first CTA to write its partial values (to initialize the
  // workspace), and for all but the final CTA to have accumulated (so that the final CTA can load
  // the accumulated value and accumulate it into registers on top of which the epilogue will
  // be performed).
  //
  // Due to the nondeterminsitic ordering of accumulation, deterministic numeric behavior cannot
  // be guaranteed with this mode (e.g., floating-point rounding error will depend on the order
  // of accumulation)
  Nondeterministic
};

////////////////////////////////////////////////////////////////////////////////

// Strategies for decomposing the problem
enum class DecompositionMode {
  // Use a heuristic to determine whether data-parallel, split-K, or stream-K decomposition should be performed
  Heuristic,
  // Force a data-parallel decomposition
  DataParallel,
  // Force a split-K decomposition. This should be paired with setting the `splits` parameter
  SplitK,
  // Force a stream-K decomposition
  StreamK
};

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel::detail
