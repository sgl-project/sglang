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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

// This example extends `tiled_copy` using predicate tensors to guard memory accesses performed
// by `cute::copy_if()`. This enables tensors to have shapes that are not integer multiples of
// block sizes.
//
// This is accomplished by instantiating a tensor of coordinates which correspond to tensor elements
// to be accessed and then computing a predicate tensor which masks accesses. The example demonstrates
// how constructing of an identity tensor containing coordinates and a predicate tensor containing
// mask bits can be implemented using the same CuTe operations used to tile the tensors in
// Global Memory.
//
// This example implements two variants:
//  - copy_if_kernel() uses `cute::local_partition()` to construct each thread's slice
//  - copy_if_kernel_vectorized() uses `make_tiled_copy() to implement vectorized memory accesses.
//
// The tensor shapes and strides must be divisible by the shape of the vector access.
//

/// Simple copy kernel.
//
// Uses local_partition() to partition a tile among threads arranged as (THR_M, THR_N).
template <class TensorS, class TensorD, class BlockShape, class ThreadLayout>
__global__ void copy_if_kernel(TensorS S, TensorD D, BlockShape block_shape, ThreadLayout)
{
  using namespace cute;

  // Construct a coordinate tensor whose elements are the coordinates used to access tensors S and D.
  auto shape_S = shape(S);
  Tensor C = make_identity_tensor(shape_S);
  // Construct a predicate tensor which compares the coordinates with the original shape
  Tensor P = cute::lazy::transform(C, [&](auto c) { return elem_less(c, shape_S); });

  // Tile the input tensor into blocks
  auto block_coord = make_coord(blockIdx.x, blockIdx.y);
  Tensor tile_S = local_tile(S, block_shape, block_coord);   // (BlockShape_M, BlockShape_N)
  Tensor tile_D = local_tile(D, block_shape, block_coord);   // (BlockShape_M, BlockShape_N)
  Tensor tile_P = local_tile(P, block_shape, block_coord);   // (BlockShape_M, BlockShape_N)

  // Construct a partitioning of the tile among threads with the given thread arrangement.

  // Concept:                         Tensor  ThrLayout       ThrIndex
  Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{}, threadIdx.x);
  Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{}, threadIdx.x);
  Tensor thr_tile_P = local_partition(tile_P, ThreadLayout{}, threadIdx.x);

  // Copy from GMEM to GMEM using `thr_tile_P` to guard accesses.
  copy_if(thr_tile_P, thr_tile_S, thr_tile_D);
}

/// Vectorized copy kernel.
///
/// Uses `make_tiled_copy()` to perform a copy using vector instructions. This operation
/// has the precondition that pointers are aligned to the vector size.
///
template <class TensorS, class TensorD, class BlockShape, class Tiled_Copy>
__global__ void copy_if_kernel_vectorized(TensorS S, TensorD D, BlockShape block_shape, Tiled_Copy tiled_copy)
{
  using namespace cute;

  // Construct a coordinate tensor whose elements are the coordinates used to access tensors S and D.
  auto shape_S = shape(S);
  Tensor C = make_identity_tensor(shape_S);
  // Construct a predicate tensor which compares the coordinates with the original shape
  Tensor P = cute::lazy::transform(C, [&](auto c) { return elem_less(c, shape_S); });

  // Tile the input tensor into blocks
  auto block_coord = make_coord(blockIdx.x, blockIdx.y);
  Tensor tile_S = local_tile(S, block_shape, block_coord);       // (BlockShape_M, BlockShape_N)
  Tensor tile_D = local_tile(D, block_shape, block_coord);       // (BlockShape_M, BlockShape_N)
  Tensor tile_P = local_tile(P, block_shape, block_coord);       // (BlockShape_M, BlockShape_N)

  //
  // Construct a Tensor corresponding to each thread's slice.
  //
  ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
  Tensor thr_tile_S = thr_copy.partition_S(tile_S);              // (CPY, CPY_M, CPY_N)
  Tensor thr_tile_D = thr_copy.partition_D(tile_D);              // (CPY, CPY_M, CPY_N)
  Tensor thr_tile_P = thr_copy.partition_S(tile_P);              // (CPY, CPY_M, CPY_N)

#if 0
  // Copy from GMEM to GMEM
  copy_if(tiled_copy, thr_tile_P, thr_tile_S, thr_tile_D);
#else
  // make_fragment_like() constructs a tensor in RMEM with the same shape as thr_tile_S.
  Tensor frag = make_fragment_like(thr_tile_S);

  // Copy from GMEM to RMEM and from RMEM to GMEM
  copy_if(tiled_copy, thr_tile_P, thr_tile_S, frag);
  copy_if(tiled_copy, thr_tile_P, frag,       thr_tile_D);
#endif
}

/// Main function
int main(int argc, char** argv)
{
  //
  // Given a 2D shape, perform an efficient copy
  //

  using namespace cute;
  using Element = float;

  // Define a tensor shape with dynamic extents (m, n)
  auto tensor_shape = make_shape(528, 300);

  thrust::host_vector<Element> h_S(size(tensor_shape));
  thrust::host_vector<Element> h_D(size(tensor_shape));

  //
  // Initialize
  //

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
    h_D[i] = Element{};
  }

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;
  thrust::device_vector<Element> d_Zero = h_D;

  //
  // Make tensors
  //

  Tensor tensor_S = make_tensor(make_gmem_ptr(d_S.data().get()), make_layout(tensor_shape));
  Tensor tensor_D = make_tensor(make_gmem_ptr(d_D.data().get()), make_layout(tensor_shape));

  //
  // Partition
  //

  // Define a statically sized block (M, N).
  //
  // Note, by convention, capital letters are used to represent static modes.
  auto block_shape = make_shape(Int<128>{}, Int<64>{});

  // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
  // shape, and modes (m', n') correspond to the number of tiles.
  //
  // These will be used to determine the CUDA kernel grid dimensinos.
  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);        // ((M, N), m', n')

  // Describes the layout of threads which is then replicated to tile 'block_shape.'
  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int< 8>{}));  // (ThrM, ThrN)

  //
  // Determine grid and block dimensions
  //

  dim3 gridDim (size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));   // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(thr_layout));

  //
  // Launch the kernel
  //

  // copy_if()
  copy_if_kernel<<< gridDim, blockDim >>>(
    tensor_S,
    tensor_D,
    block_shape,
    thr_layout);

  cudaError result = cudaDeviceSynchronize();

  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  h_D = d_D;

  //
  // Verification
  //

  auto verify = [](thrust::host_vector<Element> const &S, thrust::host_vector<Element> const &D){

    int32_t errors = 0;
    int32_t const kErrorLimit = 10;

    if (S.size() != D.size()) {
      return 1;
    }

    for (size_t i = 0; i < D.size(); ++i) {
      if (S[i] != D[i]) {
        std::cerr << "Error. S[" << i << "]: " << S[i] << ",   D[" << i << "]: " << D[i] << std::endl;

        if (++errors >= kErrorLimit) {
          std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
          return errors;
        }
      }
    }

    return errors;
  };

  if (verify(h_D, h_S)) {
    return -1;
  } else {
    std::cout << "Success." << std::endl;
  }

  thrust::copy(d_Zero.begin(), d_Zero.end(), d_D.begin());

  // Construct a TiledCopy with a specific access pattern.
  //   This version uses a
  //   (1) Layout-of-Threads to describe the number and arrangement of threads (e.g. row-major, col-major, etc),
  //   (2) Layout-of-Values that each thread will access.

  // Value arrangement per thread
  Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));   // (4,1) -> val_idx

  // Define `AccessType` which controls the size of the actual memory access instruction.
  using CopyOp = UniversalCopy<uint_byte_t<sizeof(Element) * size(val_layout)>>;     // A very specific access width copy instruction
  //using CopyOp = UniversalCopy<cutlass::AlignedArray<Element, size(val_layout)>>;  // A more generic type that supports many copy strategies
  //using CopyOp = AutoVectorizingCopy;                                              // An adaptable-width instruction that assumes maximal alignment of inputs

  // A Copy_Atom corresponds to one CopyOperation applied to Tensors of type Element.
  using Atom = Copy_Atom<CopyOp, Element>;

  // Construct tiled copy, a tiling of copy atoms.
  //
  // Note, this assumes the vector and thread layouts are aligned with contigous data
  // in GMEM. Alternative thread layouts are possible but may result in uncoalesced
  // reads. Alternative value layouts are also possible, though incompatible layouts
  // will result in compile time errors.
  TiledCopy tiled_copy = make_tiled_copy(Atom{},             // Access strategy
                                         thr_layout,         // thread layout (e.g. 32x4 Col-Major)
                                         val_layout);        // value layout (e.g. 4x1)

  // copy_if() with vectorization
  copy_if_kernel_vectorized<<< gridDim, blockDim >>>(
    tensor_S,
    tensor_D,
    block_shape,
    tiled_copy);

  result = cudaDeviceSynchronize();

  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  h_D = d_D;

  if (verify(h_D, h_S)) {
    return -1;
  } else {
    std::cout << "Success." << std::endl;
  }
  return 0;
}

