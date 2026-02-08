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

#pragma once

#include <cstdio>
#include "cutlass/cutlass.h"

/**
 * \file
 * \brief C++ interface to dump fragments and shared memory contents for
 * debugging.
 */

namespace cutlass {
namespace debug {

/******************************************************************************
 * Dump the fragments
 ******************************************************************************/

/// The first N threads dump the first M elements from their fragments with a
/// stride of S elements.  If N is not specified, dump the data of all the
/// threads.  If M is not specified, dump all the elements of the fragment.
template <typename Fragment>
CUTLASS_DEVICE void dump_fragment(Fragment const& frag, int N = 0, int M = 0,
                                  int S = 1) {
  int total_threads = blockDim.x * blockDim.y * blockDim.z;
  int block_id =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int thread_id = (threadIdx.z * (blockDim.x * blockDim.y)) +
                  (threadIdx.y * blockDim.x) + threadIdx.x;

  if (N < 0 || N > total_threads) {
    if (thread_id == 0 && block_id == 0)
      printf("Thread number N = %d should between [1, %d].\n", N,
             total_threads);

    __syncthreads();

    return;
  }

  int total_elements = int(frag.size());

  if (M < 0 || M > total_elements) {
    if (thread_id == 0 && block_id == 0)
      printf("Element number M = %d should between [1, %d].\n", M,
             total_elements);

    __syncthreads();

    return;
  }

  if (N == 0) N = total_threads;

  if (M == 0) M = total_elements;

  if (S < 1 || S > M) {
    if (thread_id == 0 && block_id == 0)
      printf("Stride S = %d should between [1, %d].\n", S, M);

    __syncthreads();

    return;
  }

  if (thread_id == 0 && block_id == 0)
    printf("\n*******************Dumping the fragments*******************\n\n");

  CUTLASS_PRAGMA_NO_UNROLL
  for (int tid = 0; tid < N; ++tid) {
    if (tid == thread_id) {
      printf("TB%d W%d T%d: ", block_id, tid / 32, tid & 31);
      CUTLASS_PRAGMA_NO_UNROLL
      for (int i = 0; i < M; i += S) {
        printf("%.0f ", float(typename Fragment::value_type(frag[i])));
      }
      printf("\n");
    }

    __syncthreads();
  }

  if (thread_id == 0 && block_id == 0)
    printf("\n***********************************************************\n\n");

  __syncthreads();

  return;
}

/******************************************************************************
 * Dump the shared memory
 ******************************************************************************/

#define SHMEM_ROW_SIZE 128

/// Dump the shared memory contents.  ptr is the begin address, size specifies
/// the number of elements that need to be dumped, and S specifies the stride.
template <typename Element>
CUTLASS_DEVICE void dump_shmem(Element const* ptr, size_t size, int S = 1) {
  int block_id =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int thread_id = (threadIdx.z * (blockDim.x * blockDim.y)) +
                  (threadIdx.y * blockDim.x) + threadIdx.x;

  if (ptr == nullptr) {
    if (thread_id == 0 && block_id == 0) printf("ptr is null.\n");

    __syncthreads();
    return;
  }

  if (size < 1) {
    if (thread_id == 0 && block_id == 0)
      printf("Element size is less than 1\n");

    __syncthreads();

    return;
  }

  int row_elements = SHMEM_ROW_SIZE / sizeof(Element);

  if (S < 1 || S > row_elements) {
    if (thread_id == 0 && block_id == 0)
      printf("Stride S = %d should between [1, %d].\n", S, row_elements);

    __syncthreads();

    return;
  }

  __syncthreads();

  if (thread_id == 0)
    printf("\n********Dumping the shared memory of TB %d*******\n\n", block_id);

  if (thread_id == 0) {
    for (int i = 0; i < size; i += row_elements) {
      for (int j = 0; j < row_elements; j += S) {
        printf("%.0f ", float(ptr[i + j]));
      }

      printf("\n");
    }
  }

  if (thread_id == 0)
    printf("\n***********************************************************\n\n");

  __syncthreads();

  return;
}
}  // namespace debug
}  // namespace cutlass
