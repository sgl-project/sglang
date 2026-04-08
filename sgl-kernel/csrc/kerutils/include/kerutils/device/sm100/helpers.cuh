// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Copyright (c) 2025 DeepSeek. All Rights Reserved.
 *
 * Licensed under the MIT License.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <cute/tensor.hpp>

#include "kerutils/device/common.h"

namespace kerutils {

// Perform SS UTCMMA
// sA and sB should be shared memory tensors (i.e. make_tensor(make_shared_ptr(XXX), XXX)) while tC_frag should be tmem
// fragment
template <typename TiledMMA, typename TensorA, typename TensorB, typename TensorFragC>
CUTE_DEVICE void utcmma_ss(TiledMMA& tiled_mma, TensorA sA, TensorB sB, TensorFragC tC_frag, bool clear_accum) {
  using namespace cute;
  tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
  ThrMMA thr_mma = tiled_mma.get_slice(_0{});  // Since A/B/C are already CTA-local tiles, this number does not matter
  auto sA_frag = thr_mma.partition_fragment_A(sA);
  auto sB_frag = thr_mma.partition_fragment_B(sB);
  static_assert(size<2>(sA_frag) == size<2>(sB_frag));
  static_assert(size<1>(sA_frag) == size<1>(tC_frag));
  static_assert(size<1>(sB_frag) == size<2>(tC_frag));
  CUTE_UNROLL
  for (int k = 0; k < size<2>(sA_frag); ++k) {
    cute::gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
    tiled_mma.accumulate_ = UMMA::ScaleOut::One;
  }
}

// Perform TS UTCMMA
// sB should be shared memory tensors (i.e. make_tensor(make_shared_ptr(XXX), XXX)) while tA_frag and tC_frag should be
// tmem fragment
template <typename TiledMMA, typename TensorA, typename TensorB, typename TensorFragC>
CUTE_DEVICE void utcmma_ts(TiledMMA& tiled_mma, TensorA tA_frag, TensorB sB, TensorFragC tC_frag, bool clear_accum) {
  using namespace cute;
  tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
  ThrMMA thr_mma = tiled_mma.get_slice(_0{});  // Since A/B/C are already CTA-local tiles, this number does not matter
  auto sB_frag = thr_mma.partition_fragment_B(sB);
  static_assert(size<2>(tA_frag) == size<2>(sB_frag));
  CUTE_UNROLL
  for (int k = 0; k < size<2>(tA_frag); ++k) {
    cute::gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
    tiled_mma.accumulate_ = UMMA::ScaleOut::One;
  }
}

}  // namespace kerutils
