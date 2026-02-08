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
    \brief Unit tests for thread-level GEMM
*/

#pragma once

#include "cutlass/array.h"

namespace test {
namespace nvrtc {
namespace kernel {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Thread-level matrix multiply-accumulate
template <typename Mma>
__global__ void testbed_kernel(
  typename Mma::ElementC *D,
  typename Mma::ElementA const *A,
  typename Mma::ElementB const *B,
  typename Mma::ElementC const *C) {

  auto ptr_D = reinterpret_cast<cutlass::Array<typename Mma::ElementC, Mma::Shape::kMN> *>(D);
  auto ptr_A = reinterpret_cast<cutlass::Array<typename Mma::ElementA, Mma::Shape::kMK> const *>(A);
  auto ptr_B = reinterpret_cast<cutlass::Array<typename Mma::ElementB, Mma::Shape::kKN> const *>(B);
  auto ptr_C = reinterpret_cast<cutlass::Array<typename Mma::ElementC, Mma::Shape::kMN> const *>(C);

  Mma mma;

  auto a = *ptr_A;
  auto b = *ptr_B;
  auto c = *ptr_C;

  cutlass::Array<typename Mma::ElementC, Mma::Shape::kMN> d;

  mma(d, a, b, c);

  *ptr_D = d;
}

}
}
}
}

