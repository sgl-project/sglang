/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Distributed gemm device layer helpers.
*/

#pragma once

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::distributed::device::detail {


cutlass::Status check_cuda_status(cudaError_t status) {
  if (status != cudaSuccess) {
    auto result = cudaGetLastError();
    CUTLASS_TRACE_HOST("  error message: " << cudaGetErrorString(result));
    return cutlass::Status::kErrorInternal;
  }
  return cutlass::Status::kSuccess;                   
}

// DistGemmBufferHelper computes required buffer size and offsets for GEMM operands.
template <
  typename Tiler_, 
  typename ElementA_,
  typename ElementB_,
  typename ElementC_,
  typename ElementD_>
struct DistGemmBufferHelper {

  using Tiler = Tiler_;

  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;
  using ElementD = ElementD_;

  static constexpr int NumBuffersA = Tiler::NumBuffersA;
  static constexpr int NumBuffersB = Tiler::NumBuffersB;
  static constexpr int NumBuffersC = Tiler::NumBuffersC;
  static constexpr int NumBuffersD = Tiler::NumBuffersD;

  template <typename ProblemShape>
  static auto
  get_buffer_size_a(ProblemShape problem_shape) {
    auto a_buffer_layout = cute::make_layout(
        cute::make_shape(NumBuffersA, Tiler::get_local_a_shape(problem_shape), sizeof(ElementA))
    );
    return size(a_buffer_layout);
  }

  template <typename ProblemShape>
  static auto
  get_buffer_size_b(ProblemShape problem_shape) {
    auto b_buffer_layout = cute::make_layout(
        cute::make_shape(NumBuffersB, Tiler::get_local_b_shape(problem_shape), sizeof(ElementB))
    );
    return size(b_buffer_layout);
  }

  template <typename ProblemShape>
  static auto
  get_buffer_size_c(ProblemShape problem_shape) {
    auto c_buffer_layout = cute::make_layout(
        cute::make_shape(NumBuffersC, Tiler::get_local_c_shape(problem_shape), sizeof(ElementC))
    );
    return size(c_buffer_layout);
  }

  template <typename ProblemShape>
  static auto
  get_buffer_size_d(ProblemShape problem_shape) {
    auto d_buffer_layout = cute::make_layout(
        cute::make_shape(NumBuffersD, Tiler::get_local_d_shape(problem_shape), sizeof(ElementD))
    );
    return size(d_buffer_layout);
  }

  template <typename ProblemShape>
  static auto
  get_buffer_size(ProblemShape problem_shape) {
    size_t buffer_size = 0;

    if constexpr (NumBuffersA > 0) {
      buffer_size += get_buffer_size_a(problem_shape);
    }
    if constexpr (NumBuffersB > 0) {
      buffer_size += get_buffer_size_b(problem_shape);
    }
    if constexpr (NumBuffersC > 0) {
      buffer_size += get_buffer_size_c(problem_shape);
    }
    if constexpr (NumBuffersD > 0) {
      buffer_size += get_buffer_size_d(problem_shape);
    }

    return buffer_size;
  }

  // Buffer space: |  buffer_A  |  buffer_B  |  buffer_C  |  buffer_D  |
  // And buffer_{A,B,C,D}: |  iter 1  |  iter 2  | ... |  iter TP - 1 |
  template <typename ProblemShape>
  static size_t
  get_buffer_offset_A(ProblemShape problem_shape) {
    return 0;
  }

  template <typename ProblemShape>
  static size_t
  get_buffer_offset_B(ProblemShape problem_shape) {
    return get_buffer_size_a(problem_shape);
  }

  template <typename ProblemShape>
  static size_t
  get_buffer_offset_C(ProblemShape problem_shape) {
    return get_buffer_size_a(problem_shape) + get_buffer_size_b(problem_shape);
  }

  template <typename ProblemShape>
  static size_t
  get_buffer_offset_D(ProblemShape problem_shape) {
    return get_buffer_size_a(problem_shape) + get_buffer_size_b(problem_shape) + get_buffer_size_c(problem_shape);
  }
};

} // namespace cutlass::distributed::device::detail

///////////////////////////////////////////////////////////////////////////////

