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
    \brief 
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_params.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator_params.h"

#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmParams {

  //
  // Type definitions
  //
  using Index = int32_t;
  using LongIndex = int64_t;

  using MmaIteratorParams = typename cutlass::transform::threadblock::PredicatedTileAccessIteratorParams;  
  using EpilogueIteratorParams = typename cutlass::epilogue::threadblock::PredicatedTileIteratorParams;

  //
  // Data members
  //

  cutlass::gemm::GemmCoord problem_size{};
  cutlass::gemm::GemmCoord grid_tiled_shape{};
  int swizzle_log_tile{};

  GemmUniversalMode mode{GemmUniversalMode::kGemm};
  int batch_count{1};
  int gemm_k_size{0};

  void * ptr_A{nullptr};
  void * ptr_B{nullptr};
  void * ptr_C{nullptr};
  void * ptr_D{nullptr};

  LongIndex lda{0};
  LongIndex ldb{0};
  LongIndex ldc{0};
  LongIndex ldd{0};

  LongIndex batch_stride_A{0};
  LongIndex batch_stride_B{0};
  LongIndex batch_stride_C{0};
  LongIndex batch_stride_D{0};

  int *semaphore{nullptr};

  //
  // Methods
  //

  GemmParams() = default;

  CUTLASS_HOST_DEVICE
  GemmParams(
    cutlass::gemm::GemmCoord problem_size_,
    cutlass::gemm::GemmCoord grid_tiled_shape_,
    int swizzle_log_tile_,
    GemmUniversalMode mode_,
    int batch_count_,
    int gemm_k_size_,
    void const * ptr_A_,
    void const * ptr_B_,
    void const * ptr_C_,
    void * ptr_D_,
    LongIndex lda_,
    LongIndex ldb_, 
    LongIndex ldc_, 
    LongIndex ldd_,
    int64_t batch_stride_A_,
    int64_t batch_stride_B_,
    int64_t batch_stride_C_,
    int64_t batch_stride_D_,
    MmaIteratorParams const & params_itr_a_,
    MmaIteratorParams const & params_itr_b_,
    EpilogueIteratorParams const & params_itr_c_,
    EpilogueIteratorParams const & params_itr_d_,
    void *workspace_ = nullptr) :
      problem_size(problem_size_),
      grid_tiled_shape(grid_tiled_shape_),
      swizzle_log_tile(swizzle_log_tile_),
      mode(mode_),
      batch_count(batch_count_),
      gemm_k_size(gemm_k_size_),
      ptr_A(const_cast<void *>(ptr_A_)),
      ptr_B(const_cast<void *>(ptr_B_)),
      ptr_C(const_cast<void *>(ptr_C_)),
      ptr_D(ptr_D_),
      lda(lda_),
      ldb(ldb_),
      ldc(ldc_),
      ldd(ldd_),
      batch_stride_A(batch_stride_A_),
      batch_stride_B(batch_stride_B_),
      batch_stride_C(batch_stride_C_),
      batch_stride_D(batch_stride_D_),
      params_itr_a(params_itr_a_),
      params_itr_b(params_itr_b_),      
      params_itr_c(params_itr_c_),
      params_itr_d(params_itr_d_),
      semaphore(static_cast<int *>(workspace_)
    ) { }


  CUTLASS_HOST_DEVICE
  void update(
    void const * ptr_A_,
    void const * ptr_B_,
    void const * ptr_C_,
    void * ptr_D_,
    int64_t batch_stride_A_,
    int64_t batch_stride_B_,
    int64_t batch_stride_C_,
    int64_t batch_stride_D_,
    void *workspace_ = nullptr) {

    ptr_A = const_cast<void *>(ptr_A_);
    ptr_B = const_cast<void *>(ptr_B_);
    ptr_C = const_cast<void *>(ptr_C_);
    ptr_D = ptr_D_;

    batch_stride_A = batch_stride_A_;
    batch_stride_B = batch_stride_B_;
    batch_stride_C = batch_stride_C_;
    batch_stride_D = batch_stride_D_;


    semaphore = static_cast<int *>(workspace_);
    CUTLASS_TRACE_HOST("GemmParams::update()");
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
