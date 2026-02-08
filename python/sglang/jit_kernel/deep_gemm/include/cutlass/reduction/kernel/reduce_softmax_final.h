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
  \brief Kernel performing a final reduction for softmax
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reduction {
namespace kernel {

template <
  typename ElementNorm_,
  typename ElementSum_,
  typename ElementSoftmaxCompute_,
  typename ThreadblockShape_,
  bool GroupedProblem = false
>
class ApplySoftmaxFinalReduction {
public:

  using ElementNorm = ElementNorm_;
  using ElementSum = ElementSum_;
  using ElementSoftmaxCompute = ElementSoftmaxCompute_;
  using ThreadblockShape = ThreadblockShape_;
  static const bool isGroupedProblem = GroupedProblem;

  //
  // Arguments
  //

  struct Arguments {

    cutlass::gemm::GemmCoord*  problem_sizes{nullptr};
    cutlass::gemm::GemmCoord   problem_size{};
    ElementNorm*               block_Norm{nullptr};
    ElementSum*                block_Sum{nullptr};
    int64_t*                   offset_Norm_Device{nullptr};
    int64_t*                   offset_Sum_Device{nullptr};
    int64_t                    batch_stride_Max{0};
    int64_t                    batch_stride_Sum{0};

    //
    // Methods
    //
    Arguments() { }

    // Non-grouped constructor without batching
    Arguments(
      cutlass::gemm::GemmCoord  problem_size,
      ElementNorm*              block_Norm,
      ElementSum*               block_Sum
    ):
      problem_size(problem_size),
      block_Norm(block_Norm),
      block_Sum(block_Sum),
      problem_sizes(nullptr),
      offset_Norm_Device(nullptr),
      offset_Sum_Device(nullptr),
      batch_stride_Max(0),
      batch_stride_Sum(0)
    {

    }

    // Non-grouped constructor with batching
    Arguments(
      cutlass::gemm::GemmCoord  problem_size,
      ElementNorm*              block_Norm,
      ElementSum*               block_Sum,
      int64_t                   batch_stride_Max,
      int64_t                   batch_stride_Sum
    ):
      problem_size(problem_size),
      block_Norm(block_Norm),
      block_Sum(block_Sum),
      batch_stride_Max(batch_stride_Max),
      batch_stride_Sum(batch_stride_Sum),
      problem_sizes(nullptr),
      offset_Norm_Device(nullptr),
      offset_Sum_Device(nullptr)
    {

    }


    // Grouped constructor
    Arguments(
      cutlass::gemm::GemmCoord  *problem_sizes,
      ElementNorm*              block_Norm,
      ElementSum*               block_Sum,
      int64_t*                  offset_Norm_Device,
      int64_t*                  offset_Sum_Device
    ):
      problem_sizes(problem_sizes),
      problem_size(cutlass::gemm::GemmCoord(0, 0, 0)),
      block_Norm(block_Norm),
      block_Sum(block_Sum),
      offset_Norm_Device(offset_Norm_Device),
      offset_Sum_Device(offset_Sum_Device)
    {

    }
  };

  struct SharedStorage {


  };

  //
  // Params struct
  //

  struct Params {
    Arguments args;

    //
    // Methods
    //
    Params() { }

    Params(Arguments const &args_): args(args_) { }
  };

private:

public:

  CUTLASS_DEVICE
  ApplySoftmaxFinalReduction() { }

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    apply(params, shared_storage);
  }

private:

  /// Full reduction
  CUTLASS_DEVICE
  void apply(Params const &params, SharedStorage &shared_storage) {

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;
    
    int block_batch = blockIdx.z;

    // defining three vars for a general reduction module
    cutlass::gemm::GemmCoord problem_size = isGroupedProblem ? params.args.problem_sizes[bid] : params.args.problem_size;
    int m_dim_in_loop = isGroupedProblem ? problem_size.m() : tid + bdim;
    int access_offset = isGroupedProblem ? 0 : bid * bdim;

    if (!isGroupedProblem && access_offset + tid >= problem_size.m()) return;

    ElementNorm *curr_ptr_Max = isGroupedProblem ? \
              params.args.block_Norm + params.args.offset_Norm_Device[bid] : \
              params.args.block_Norm + block_batch * params.args.batch_stride_Max;
    ElementSum *curr_ptr_Sum = isGroupedProblem ? \
              params.args.block_Sum + params.args.offset_Sum_Device[bid] : \
              params.args.block_Sum + block_batch * params.args.batch_stride_Sum;

    int threadblock_num = (problem_size.n() + ThreadblockShape::kN - 1) / ThreadblockShape::kN;

    using ConvertSumOutput = cutlass::NumericConverter<ElementSum, ElementSoftmaxCompute>;
    using ConvertNormOutput = cutlass::NumericConverter<ElementNorm, ElementSoftmaxCompute>;

    using ConvertSum = cutlass::NumericConverter<ElementSoftmaxCompute, ElementSum>;
    using ConvertNorm = cutlass::NumericConverter<ElementSoftmaxCompute, ElementNorm>;

    ConvertSum   convert_sum;
    ConvertNorm  convert_norm;

    ConvertSumOutput   convert_sum_output;
    ConvertNormOutput  convert_norm_output;

    uint32_t float_max_bits = 0xff7fffff;
    float min_float = reinterpret_cast<float const &>(float_max_bits);

    CUTLASS_PRAGMA_UNROLL
    for (int idx_m = tid; idx_m < m_dim_in_loop; idx_m += bdim) {
      ElementNorm *access_n = curr_ptr_Max + idx_m + access_offset;
      ElementSum *access_s = curr_ptr_Sum + idx_m + access_offset;
      ElementNorm *access_n_bak = access_n;
      ElementSum *access_s_bak = access_s;
      ElementSoftmaxCompute max_val = ElementSoftmaxCompute(min_float);
      ElementSoftmaxCompute sum_val = ElementSoftmaxCompute(0);
      ElementNorm fetch_n;
      ElementSum fetch_s;

      CUTLASS_PRAGMA_UNROLL
      for (int idx_n = 0; idx_n < threadblock_num; idx_n++) {
        cutlass::arch::global_load<ElementNorm, sizeof(ElementNorm)>(fetch_n, access_n, true);
        max_val = cutlass::fast_max(max_val, convert_norm(fetch_n));
        access_n += problem_size.m();
      }

      access_n = access_n_bak;

      CUTLASS_PRAGMA_UNROLL
      for (int idx_n = 0; idx_n < threadblock_num; idx_n++) {
        cutlass::arch::global_load<ElementNorm, sizeof(ElementNorm)>(fetch_n, access_n, true);
        cutlass::arch::global_load<ElementSum, sizeof(ElementSum)>(fetch_s, access_s, true);
        sum_val += convert_sum(fetch_s) * cutlass::fast_exp(convert_norm(fetch_n) - max_val);
        access_n += problem_size.m();
        access_s += problem_size.m();
      }

      ElementSoftmaxCompute inv_sum = cutlass::constants::one<ElementSoftmaxCompute>() / sum_val;

      access_n = access_n_bak;
      access_s = access_s_bak;

      access_n[0] = convert_norm_output(max_val);
      access_s[0] = convert_sum_output(inv_sum);
    }

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace reduction
} // namespace cutlass
