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
/* \file
  \brief Defines device-side elementwise operations on TensorView. Note, the operations defined
    in this header are not specialized for any particular data layout and are therefore not
    intended to offer the best possible performance. Rather, they are intended to be generic
    reference implementations to support the CUTLASS unit tests.
*/

#pragma once

// Cutlass includes
#include "cutlass/cutlass.h"
#include "cutlass/tensor_view.h"

#include "cutlass/gemm/gemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reference {
namespace device {

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace kernel {

template <
  typename TensorRefIn,                   ///< Input TensorRef Type
  typename TensorRefOut,                  ///< Output TensorRef Type
  typename ScalarType,                    ///< alpha Type
  typename TensorRefScalar,               ///< Scale/Bias TensorRef Type
  typename OutputTile,
  typename ConvertOp = NumericConverter<typename TensorRefOut::Element, ScalarType>
>
__global__ void TensorScaleBiasGemm(
  gemm::GemmCoord problem_size,
  TensorRefIn tensor_in,                  ///< input tensor
  TensorRefOut tensor_out,                ///< output tensor
  ScalarType alpha,                       ///< alpha
  TensorRefScalar tensor_scale,           ///< scale tensor
  TensorRefScalar tensor_bias             ///< bias tensor
) {

  ConvertOp convert_op;

  MatrixCoord output_coord(
    MatrixCoord::Index((threadIdx.x + blockIdx.x * blockDim.x) * OutputTile::kRow),
    MatrixCoord::Index((threadIdx.y + blockIdx.y * blockDim.y) * OutputTile::kColumn)
  );

  // Update the output tensor
  for (int j = 0; j < OutputTile::kRow; ++j) {
    for (int i = 0; i < OutputTile::kColumn; ++i) {
      MatrixCoord coord = output_coord + MatrixCoord(i, j);
      if (coord.row() < problem_size.m() && coord.column() < problem_size.n()) {

        ScalarType scale = alpha;
        if(tensor_scale.good())
          scale = tensor_scale.at({0, coord.column()});

        ScalarType bias = ScalarType(0);

        if(tensor_bias.good())
          bias = tensor_bias.at({0, coord.column()});

        tensor_out.at(coord) = convert_op(
          scale * ScalarType(tensor_in.at(coord)) + bias);
      }
    }
  }
}

template <
  typename TensorRefIn,                   ///< Input TensorRef Type
  typename TensorRefOut,                  ///< Output TensorRef Type
  typename ScalarType,                    ///< alpha Type
  typename TensorRefScalar,               ///< Scale/Bias TensorRef Type
  typename ConvertOp = NumericConverter<typename TensorRefOut::Element, ScalarType>,
  int kMblock = 4,
  int kNblock = 4
>
__global__ void TensorScaleBiasGemmBatched(
  gemm::GemmCoord problem_size,
  TensorRefIn tensor_in,                  ///< input tensor
  TensorRefOut tensor_out,                ///< output tensor
  ScalarType alpha,                       ///< alpha
  TensorRefScalar tensor_scale,           ///< scale tensor
  TensorRefScalar tensor_bias,             ///< bias tensor
  int batch_count = 1,
  int64_t batch_stride_tensor_in = 0,
  int64_t batch_stride_tensor_out = 0,
  int64_t batch_stride_tensor_scale = 0,
  int64_t batch_stride_tensor_bias = 0
) {

  ConvertOp convert_op;
  int row_block = (blockIdx.x * blockDim.x + threadIdx.x) * kMblock;
  int col_block = (blockIdx.y * blockDim.y + threadIdx.y) * kNblock;
  int batch_idx = blockIdx.z;

  tensor_in.add_pointer_offset(batch_idx * batch_stride_tensor_in);
  tensor_out.add_pointer_offset(batch_idx * batch_stride_tensor_out);
  tensor_scale.add_pointer_offset(batch_idx * batch_stride_tensor_scale);
  tensor_bias.add_pointer_offset(batch_idx * batch_stride_tensor_bias);

  for (; batch_idx < batch_count; batch_idx += gridDim.z) {
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kNblock; j++) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kMblock; i++) {
        int row = row_block + i;
        int col = col_block + j;
        MatrixCoord coord = MatrixCoord(row, col);
        if (coord.row() < problem_size.m() && coord.column() < problem_size.n()) {

          ScalarType scale = alpha;
          if(tensor_scale.good())
            scale = tensor_scale.at({0, coord.column()});

          ScalarType bias = ScalarType(0);

          if(tensor_bias.good())
            bias = tensor_bias.at({0, coord.column()});

          tensor_out.at(coord) = convert_op(
            scale * ScalarType(tensor_in.at(coord)) + bias);
        }
      }
    }
    tensor_in.add_pointer_offset(batch_stride_tensor_in * gridDim.z);
    tensor_out.add_pointer_offset(batch_stride_tensor_out * gridDim.z);
    tensor_scale.add_pointer_offset(batch_stride_tensor_scale * gridDim.z);
    tensor_bias.add_pointer_offset(batch_stride_tensor_bias * gridDim.z);
  }
}

template <
  typename TensorRefIn,                   ///< Input TensorRef Type
  typename TensorRefOut,                  ///< Output TensorRef Type
  typename ScalarType,                    ///< alpha Type
  typename TensorRefScalar,               ///< Scale/Bias TensorRef Type
  typename ConvertOp = NumericConverter<typename TensorRefOut::Element, ScalarType>,
  int kThreadM = 4,       // shape of a thread's tile in the GEMM M dimension
  int kThreadN = 4,       // shape of a thread's tile in the GEMM N dimension
  int kCtaShapeM = 16,    // shape of a threadblock in units of threads
  int kCtaShapeN = 8      // shape of a threadblock in units of threads
>
__global__ void TensorScaleBiasConv2d(
  conv::Conv2dProblemSize problem_size,
  TensorRefIn tensor_in,                  ///< input tensor
  TensorRefOut tensor_out,                ///< output tensor
  ScalarType alpha,                       ///< alpha
  TensorRefScalar tensor_scale,           ///< scale tensor
  TensorRefScalar tensor_bias             ///< bias tensor
) {

  ConvertOp convert_op;

  int64_t npq_start = int64_t(blockIdx.x) * kCtaShapeM * kThreadM + threadIdx.x * kThreadM;
  int k_start = blockIdx.y * kCtaShapeN * kThreadN + threadIdx.y * kThreadN;

  int thread_n[kThreadM];
  int thread_p[kThreadM];
  int thread_q[kThreadM];

  // Compute N, P, Q coordinates for each row of a thread's tile
  int64_t PQ = int64_t(problem_size.P) * problem_size.Q;

  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {

    int64_t npq = npq_start + m;

    thread_n[m] = int(npq / PQ);

    int64_t residual = npq % PQ;
    thread_p[m] = int(residual / problem_size.Q);
    thread_q[m] = int(residual % problem_size.Q);
  }

  // Write out the results
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kThreadM; ++m) {
    if (thread_n[m] < problem_size.N && thread_p[m] < problem_size.P && thread_q[m] < problem_size.Q) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < kThreadN; ++n) {
        int thread_k = k_start + n;
        if (thread_k < problem_size.K) {

          ScalarType scale = alpha;
          if(tensor_scale.good())
            scale = tensor_scale.at({0, thread_k});

          ScalarType bias = ScalarType(0);
          if(tensor_bias.good())
            bias = tensor_bias.at({0, thread_k});

          tensor_out.at({thread_n[m], thread_p[m], thread_q[m], thread_k}) = convert_op(
            scale * ScalarType(
              tensor_in.at({thread_n[m], thread_p[m], thread_q[m], thread_k})
            ) + bias);
        }
      }
    }
  }

}

}

/// Apply scale and bias on a tensor
template <
  typename ElementIn,                   ///< Input Type
  typename ElementOut,                  ///< Output Type
  typename Layout,                      ///< Layout of input/output tensor
  typename ScalarType,                  ///< alpha Type
  typename LayoutScaleBias,             ///< Layout of scale and bias
  typename ConvertOp = NumericConverter<ElementOut, ScalarType>
>
void TensorScaleBiasGemm(
  gemm::GemmCoord problem_size,
  TensorRef<ElementIn, Layout> tensor_in,              ///< input tensor
  TensorRef<ElementOut, Layout> tensor_out,            ///< output tensor
  ScalarType alpha,                                    ///< alpha
  TensorRef<ScalarType, LayoutScaleBias> tensor_scale, ///< scale tensor
  TensorRef<ScalarType, LayoutScaleBias> tensor_bias    ///< bias tensor
) {

  using OutputTile = MatrixShape<4, 4>;

  dim3 block(16, 8);

  dim3 grid(
    (problem_size.m() + block.x * OutputTile::kRow - 1) / (block.x * OutputTile::kRow),
    (problem_size.n() + block.y * OutputTile::kColumn - 1) / (block.y * OutputTile::kColumn)
  );

  kernel::TensorScaleBiasGemm<
    TensorRef<ElementIn, Layout>,
    TensorRef<ElementOut, Layout>,
    ScalarType,
    TensorRef<ScalarType, LayoutScaleBias>,
    OutputTile,
    ConvertOp
  ><<< grid, block >>> (
    problem_size,
    tensor_in,
    tensor_out,
    alpha,
    tensor_scale,
    tensor_bias
  );
}

/// Apply scale and bias on a tensor
template <
  typename ElementIn,                   ///< Input Type
  typename ElementOut,                  ///< Output Type
  typename Layout,                      ///< Layout of input/output tensor
  typename ScalarType,                  ///< alpha Type
  typename LayoutScaleBias,             ///< Layout of scale and bias
  typename ConvertOp = NumericConverter<ElementOut, ScalarType>
>
void TensorScaleBiasGemmBatched(
  gemm::GemmCoord problem_size,
  TensorRef<ElementIn, Layout> tensor_in,              ///< input tensor
  TensorRef<ElementOut, Layout> tensor_out,            ///< output tensor
  ScalarType alpha,                                    ///< alpha
  TensorRef<ScalarType, LayoutScaleBias> tensor_scale, ///< scale tensor
  TensorRef<ScalarType, LayoutScaleBias> tensor_bias,    ///< bias tensor
  int batch_count = 1,
  int64_t batch_stride_tensor_in = 0,
  int64_t batch_stride_tensor_out = 0,
  int64_t batch_stride_tensor_scale = 0,
  int64_t batch_stride_tensor_bias = 0
) {

  int const kMblock = 4;
  int const kNblock = 4;

  dim3 block(16, 8);
  dim3 grid(
    (problem_size.m() + block.x * kMblock - 1) / (block.x * kMblock),
    (problem_size.n() + block.y * kNblock - 1) / (block.y * kNblock),
    batch_count % std::numeric_limits<uint16_t>::max()
  );

  kernel::TensorScaleBiasGemmBatched<
    TensorRef<ElementIn, Layout>,
    TensorRef<ElementOut, Layout>,
    ScalarType,
    TensorRef<ScalarType, LayoutScaleBias>,
    ConvertOp,
    kMblock,
    kNblock
  ><<< grid, block >>> (
    problem_size,
    tensor_in,
    tensor_out,
    alpha,
    tensor_scale,
    tensor_bias,
    batch_count,
    batch_stride_tensor_in,
    batch_stride_tensor_out,
    batch_stride_tensor_scale,
    batch_stride_tensor_bias
  );
}

/// Apply scale and bias on a tensor
template <
  typename ElementIn,                   ///< Input Type
  typename ElementOut,                  ///< Output Type
  typename Layout,                      ///< Layout of input/output tensor
  typename ScalarType,                  ///< alpha Type
  typename LayoutScaleBias,             ///< Layout of scale and bias
  typename ConvertOp = NumericConverter<ElementOut, ScalarType>
>
void TensorScaleBiasConv2d(
  conv::Conv2dProblemSize problem_size,
  TensorRef<ElementIn, Layout> tensor_in,              ///< input tensor
  TensorRef<ElementOut, Layout> tensor_out,            ///< output tensor
  ScalarType alpha,                                    ///< alpha
  TensorRef<ScalarType, LayoutScaleBias> tensor_scale, ///< scale tensor
  TensorRef<ScalarType, LayoutScaleBias> tensor_bias    ///< bias tensor
) {

  int const kThreadM = 4;       // shape of a thread's tile in the GEMM M dimension
  int const kThreadN = 4;       // shape of a thread's tile in the GEMM N dimension
  int const kCtaShapeM = 16;    // shape of a threadblock in units of threads
  int const kCtaShapeN = 8;     // shape of a threadblock in units of threads

  int64_t npq = int64_t(problem_size.N) * problem_size.P * problem_size.Q;
  int64_t blocks_m = (npq + (kCtaShapeM * kThreadM) - 1) / (kCtaShapeM * kThreadM);

  dim3 block(kCtaShapeM, kCtaShapeN);
  dim3 grid(uint32_t(blocks_m), (problem_size.K + (kCtaShapeN * kThreadN) - 1) / (kCtaShapeN * kThreadN));


  kernel::TensorScaleBiasConv2d<
    TensorRef<ElementIn, Layout>,
    TensorRef<ElementOut, Layout>,
    ScalarType,
    TensorRef<ScalarType, LayoutScaleBias>,
    ConvertOp,
    kThreadM,
    kThreadN,
    kCtaShapeM,
    kCtaShapeN
  ><<< grid, block >>> (
    problem_size,
    tensor_in,
    tensor_out,
    alpha,
    tensor_scale,
    tensor_bias
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace reference
} // namespace cutlass
