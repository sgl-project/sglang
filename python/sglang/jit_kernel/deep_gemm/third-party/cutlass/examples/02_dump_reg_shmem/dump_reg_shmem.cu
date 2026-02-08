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
  \brief Demonstrate CUTLASS debugging tool for dumping fragments and shared
  memory
 */

///////////////////////////////////////////////////////////////////////////////////////////////////

// Standard Library includes

#include <iostream>

//
// CUTLASS includes
//

#include "cutlass/aligned_buffer.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/core_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h"

#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"

#define EXAMPLE_MATRIX_ROW 64
#define EXAMPLE_MATRIX_COL 32

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element, typename GmemIterator, typename SmemIterator>
__global__ void kernel_dump(typename GmemIterator::Params params,
                            typename GmemIterator::TensorRef ref) {
  extern __shared__ Element shared_storage[];

  // Construct the global iterator and load the data to the fragments.
  int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  GmemIterator gmem_iterator(params, ref.data(),
                             {EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL},
                             tb_thread_id);

  typename GmemIterator::Fragment frag;

  frag.clear();
  gmem_iterator.load(frag);

  // Call dump_fragment() with different parameters.
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nAll threads dump all the elements:\n");
  cutlass::debug::dump_fragment(frag);

  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nFirst thread dumps all the elements:\n");
  cutlass::debug::dump_fragment(frag, /*N = */ 1);

  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nFirst thread dumps first 16 elements:\n");
  cutlass::debug::dump_fragment(frag, /*N = */ 1, /*M = */ 16);

  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nFirst thread dumps first 16 elements with a stride of 8:\n");
  cutlass::debug::dump_fragment(frag, /*N = */ 1, /*M = */ 16, /*S = */ 8);

  // Construct the shared iterator and store the data to the shared memory.
  SmemIterator smem_iterator(
      typename SmemIterator::TensorRef(
          {shared_storage, SmemIterator::Layout::packed(
                               {EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL})}),
      tb_thread_id);

  smem_iterator.store(frag);

  // Call dump_shmem() with different parameters.
  if (threadIdx.x == 0 && blockIdx.x == 0) printf("\nDump all the elements:\n");
  cutlass::debug::dump_shmem(shared_storage,
                             EXAMPLE_MATRIX_ROW * EXAMPLE_MATRIX_COL);

  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nDump all the elements with a stride of 8:\n");
  cutlass::debug::dump_shmem(
      shared_storage, EXAMPLE_MATRIX_ROW * EXAMPLE_MATRIX_COL, /*S = */ 8);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point for dump_reg_shmem example.
//
// usage:
//
//   02_dump_reg_shmem
//
int main() {
  // Initialize a 64x32 column major matrix with sequential data (1,2,3...).
  using Element = cutlass::half_t;
  using Layout = cutlass::layout::ColumnMajor;

  cutlass::HostTensor<Element, Layout> matrix(
      {EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL});
  cutlass::reference::host::BlockFillSequential(matrix.host_data(),
                                                matrix.capacity());

  // Dump the matrix.
  std::cout << "Matrix:\n" << matrix.host_view() << "\n";

  // Copy the matrix to the device.
  matrix.sync_device();

  // Define a global iterator, a shared iterator and their thread map.
  using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>,
      32, cutlass::layout::PitchLinearShape<8, 4>, 8>;

  using GmemIterator =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>, Element,
          Layout, 1, ThreadMap>;

  typename GmemIterator::Params params(matrix.layout());

  using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>, Element,
      cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<16, 64>, 1,
      ThreadMap>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  int smem_size =
      int(sizeof(Element) * EXAMPLE_MATRIX_ROW * EXAMPLE_MATRIX_COL);

  kernel_dump<Element, GmemIterator, SmemIterator>
      <<<grid, block, smem_size, 0>>>(params, matrix.device_ref());

  cudaError_t result = cudaDeviceSynchronize();

  if (result != cudaSuccess) {
    std::cout << "Failed" << std::endl;
  }

  return (result == cudaSuccess ? 0 : -1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
