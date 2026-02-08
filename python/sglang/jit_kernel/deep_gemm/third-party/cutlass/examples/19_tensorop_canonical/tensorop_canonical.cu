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

/*
  This example requires NVIDIA Ampere GPU or later.
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// CUTLASS Includes
#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"

// CUTLASS Utility Includes
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gemm_complex.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

// Define the overall warp-level problem shape
int const kM = 27;
int const kN = 31;
int const kK = 17;

///////////////////////////////////////////////////////////////////////////////////////////////////

// Define a warp-level GEMM operator.
//
// This template could be part of the CUTLASS Template Library or implemented internally. This
// wraps the matrix multiply operation and epilogue with a GEMM-like interface that can be
// instantiated in device code.

namespace cutlass {
namespace gemm {
namespace warp {

template <
  typename Shape,
  typename InstructionShape,
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementScalar
>
class GemmTensorOp {
public:

  using WarpShape = GemmShape<
    ((Shape::kM + InstructionShape::kM - 1) / InstructionShape::kM) * InstructionShape::kM,
    ((Shape::kN + InstructionShape::kN - 1) / InstructionShape::kN) * InstructionShape::kN,
    InstructionShape::kK
  >;

  using MmaWarp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    WarpShape,
    InstructionShape,
    double,                             // Data type of A elements
    cutlass::layout::RowMajor,          // Layout of A matrix
    double,                             // Data type of B elements
    cutlass::layout::ColumnMajor,       // Layout of B matrix
    double,                             // Data type of C elements
    cutlass::layout::RowMajor           // Layout of C matrix
  >::Type;
 
  // Number of 'K groups' 
  int const kKgroups = (Shape::kK + InstructionShape::kK - 1) / InstructionShape::kK;

  // Define a 'FragmentIterator' to iterate over slices of accumulators
  using FragmentIterator = cutlass::epilogue::warp::FragmentIteratorTensorOp<
    typename MmaWarp::Shape,
    InstructionShape,
    double,
    typename MmaWarp::Policy::Operator::FragmentC,
    cutlass::layout::RowMajor
  >;

  // Define an epilogue 'Tile Iteterator' to iterate over slices of elements in Shared Memory
  using AccumulatorTileIterator = cutlass::epilogue::warp::TileIteratorTensorOpCanonical<
    typename MmaWarp::Shape,
    InstructionShape,
    double,
    cutlass::layout::RowMajor
  >;

  using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
  using TensorRefB = typename MmaWarp::IteratorB::TensorRef;
  using TensorRefC = typename AccumulatorTileIterator::TensorRef;

public:
  CUTLASS_HOST_DEVICE
  GemmTensorOp() { }

  CUTLASS_DEVICE
  void operator()(
    ElementScalar alpha, 
    TensorRefA ref_A, 
    TensorRefB ref_B, 
    ElementScalar beta,
    TensorRefC ref_C,
    TensorRefC ref_D,
    int lane_id) const {
  
    // Instantiate iterators pointing to slices of the A and B matrices in shared memory
    typename MmaWarp::IteratorA iter_A(ref_A, {Shape::kM, Shape::kK}, lane_id);
    typename MmaWarp::IteratorB iter_B(ref_B, {Shape::kK, Shape::kN}, lane_id);

    // Instantiate and clear accumulator tile holding the C matrix
    typename MmaWarp::FragmentC accum;
    accum.clear();
  
    // Instantiate the warp-level matrix multiply operator
    MmaWarp mma_op;

    // Instantiate fragments holding the slice of the matrix held by each warp
    typename MmaWarp::FragmentA frag_A[2];
    typename MmaWarp::FragmentB frag_B[2];
      
    // Load fragments from shared memory
    iter_A.load(frag_A[0]);
    iter_B.load(frag_B[0]);

    ++iter_A;
    ++iter_B;

    // Load fragments from shared memory
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kKgroups; ++k) {

      // Load fragments from shared memory
      iter_A.load(frag_A[(k + 1) % 2]);
      iter_B.load(frag_B[(k + 1) % 2]);

      ++iter_A;
      ++iter_B;

      // Compute the matrix multiply
      mma_op(accum, frag_A[k % 2], frag_B[k % 2], accum);
    }
  
    // Instantiate iterators
    FragmentIterator accum_frag_it(accum);
    AccumulatorTileIterator source_tile_it(ref_C, {Shape::kM, Shape::kN}, lane_id);
    AccumulatorTileIterator dest_tile_it(ref_D, {Shape::kM, Shape::kN}, lane_id);

    // Define function objects for linear scaling operation
    cutlass::multiplies<typename FragmentIterator::Fragment> mul_source;
    cutlass::multiply_add<typename FragmentIterator::Fragment> mul_add_accumulator;

    // Iterate over the epilogue components
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < FragmentIterator::kIterations; ++idx) {

      // Define storage for slices of the accumulators
      typename FragmentIterator::Fragment accum_fragment;
      typename FragmentIterator::Fragment source_fragment;

      // Select a slice of accumulators from the accumulator tile
      accum_frag_it.load(accum_fragment);
      ++accum_frag_it;

      // Load a corresponding slice from Shared memory
      source_tile_it.load(source_fragment);
      ++source_tile_it;

      // Compute linear scaling - alpha * AB + beta * C
      source_fragment = mul_source(beta, source_fragment);
      accum_fragment = mul_add_accumulator(alpha, accum_fragment, source_fragment);

      // Store the result to shared memory
      dest_tile_it.store(accum_fragment);
      ++dest_tile_it;
    }
  }
};

} // namespace warp
} // namespace gemm
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

// Sample kernel demonstrating a collective GEMM operation by a warp on arbitrary matrices held
// in Shared Memory.
__global__ void kernel(
  double *D_gmem, 
  double alpha, 
  double const *A_gmem, 
  double const *B_gmem, 
  double beta,
  double const *C_gmem) {

  // Define several matrices in shared memory
  __shared__ double A[kM][kK];
  __shared__ double B[kN][kK];
  __shared__ double C[kM][kN];

  // Copy data into SMEM
  if (threadIdx.x == 0) {
    CUTLASS_PRAGMA_NO_UNROLL
    for (int m = 0; m < kM; ++m) {
      for (int k = 0; k < kK; ++k) {
        A[m][k] = A_gmem[m * kK + k];
      }
    }
    CUTLASS_PRAGMA_NO_UNROLL
    for (int n = 0; n < kN; ++n) {
      for (int k = 0; k < kK; ++k) {
        B[n][k] = B_gmem[n * kK + k];
      }
    }
    CUTLASS_PRAGMA_NO_UNROLL
    for (int m = 0; m < kM; ++m) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int n = 0; n < kN; ++n) {
        C[m][n] = C_gmem[m * kN + n];
      }
    }
  }

  __syncthreads();
  
  //
  // Instantiate a warp-level matrix multiply operator given the fundamental instruction shape (8x8x4),
  // overall shape, data type of each operand, and layout of each operand.
  //

  using GemmTensorOp = cutlass::gemm::warp::GemmTensorOp<
    cutlass::gemm::GemmShape<kM, kN, kK>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    double,                             // Data type of A elements
    cutlass::layout::RowMajor,          // Layout of A matrix
    double,                             // Data type of B elements
    cutlass::layout::ColumnMajor,       // Layout of B matrix
    double,                             // Data type of C elements
    cutlass::layout::RowMajor,          // Layout of C matrix
    double                              // Scalar type of alpha and beta
  >;

  // Instantiate the GEMM operator
  GemmTensorOp gemm;

  // Execute the warp-level GEMM operation
  gemm(
    alpha, 
    {&A[0][0], kK},
    {&B[0][0], kK},
    beta,
    {&C[0][0], kN},
    {&C[0][0], kN},
    threadIdx.x);

  __syncthreads();
  
  // Copy data into SMEM
  if (threadIdx.x == 0) {
    CUTLASS_PRAGMA_NO_UNROLL
    for (int m = 0; m < kM; ++m) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int n = 0; n < kN; ++n) {
        D_gmem[m * kN + n] = C[m][n];
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to canonical warp-level GEMM operation
int main(int argc, const char *arg[]) {

  bool notSupported = false;

  // CUTLASS must be compiled with CUDA 11 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "NVIDIA Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "This example requires compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    // Return 0 so tests are considered passing if run on unsupported platforms.
    return 0;
  }

  cutlass::HostTensor<double, cutlass::layout::RowMajor> A({kM, kK});
  cutlass::HostTensor<double, cutlass::layout::ColumnMajor> B({kK, kN});
  cutlass::HostTensor<double, cutlass::layout::RowMajor> C({kM, kN});
  cutlass::HostTensor<double, cutlass::layout::RowMajor> D({kM, kN});

  uint64_t seed = 2020;
  double max = 8;
  double min = -8;

  cutlass::reference::host::TensorFillRandomUniform(
    A.host_view(),
    seed,
    max,
    min,
    0
  );

  cutlass::reference::host::TensorFillRandomUniform(
    B.host_view(),
    seed + 17,
    max,
    min,
    0
  );

  cutlass::reference::host::TensorFillRandomUniform(
    C.host_view(),
    seed + 31,
    max,
    min,
    0
  );

  A.sync_device();
  B.sync_device();
  C.sync_device();
  D.sync_device();

  dim3 grid(1,1);
  dim3 block(32, 1, 1);

  double alpha = 2.25;
  double beta = 1.24;

  kernel<<< grid, block >>>(
    D.device_data(),
    alpha,
    A.device_data(),
    B.device_data(),
    beta,
    C.device_data()
  );

  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "Failed to synchronize device after kernel launch." << std::endl;
    return -1;
  }

  D.sync_host();
  
  // Compute reference on host
  cutlass::HostTensor<double, cutlass::layout::RowMajor> D_ref({kM, kN}, false);

  cutlass::reference::host::GemmComplex(
    {kM, kN, kK},
    alpha,
    A.host_ref(),
    cutlass::ComplexTransform::kNone,
    B.host_ref(),
    cutlass::ComplexTransform::kNone,
    beta,
    C.host_ref(),
    D_ref.host_ref(),
    double()
  );

  // Verify reference matches computed
  if (!cutlass::reference::host::TensorEquals(
    D.host_view(),
    D_ref.host_view())) {

    std::cerr 
      << "A =\n" << A.host_view() 
      << "\n\nB = \n" << B.host_view() 
      << "\n\nC = " << C.host_view() 
      << "\n\nRef =\n" << D_ref.host_view()
      << "\n\nD =\n" << D.host_view() << "\n\n";

    std::cerr << "Error - device results mismatch host reference." << std::endl;

    return -1;
  }

  std::cout << "Passed" << std::endl;

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
