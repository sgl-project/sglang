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
  \brief Epilogue visitor for threadblock scoped GEMMs that process softmax computations in epilogue.

  The epilogue finds max values in each row of the row-major output matrix and stores them.
  The max values are also used for a further round of threadblock scoped reduction operation, where
  the partial reduction results are stored in a pre-allocated array and used for further full reduction.

*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/arch/memory.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"  // cutlass::TensorRef

namespace cutlass
{
namespace epilogue
{
namespace threadblock
{

template <int kVectorSize_,
          typename ThreadShape_,
          typename ElementCompute_,
          typename ElementAccumulator_,
          typename ElementC_,
          typename ElementD_,
          typename ElementSFD_,
          typename LayoutOutput_,
          typename LayoutSFD_>
class GemvEpilogueWithScalingFactor
{
  public:
  using ThreadShape = ThreadShape_;
  using ElementCompute = ElementCompute_;          // f32
  using ElementAccumulator = ElementAccumulator_;  // f32
  using ElementC = ElementC_;                      // e2m1
  using ElementD = ElementD_;                      // e2m1
  using ElementSFD = ElementSFD_;                  // e4m3
  using LayoutOutput = LayoutOutput_;              // ColumnMajor
  using LayoutSFD = LayoutSFD_;                    // ColumnMajor
  using TensorRefD = TensorRef<ElementD, LayoutOutput_>;
  static constexpr int kVectorSize = kVectorSize_;
  // number of threads row
  static constexpr int kThreadsPerCol = ThreadShape::kM;  // 16
  // number of threads col
  static constexpr int kThreadsPerRow = ThreadShape::kN;                // 8
  static constexpr int kThreadCount = kThreadsPerCol * kThreadsPerRow;  // 128

  static_assert(kVectorSize == kThreadsPerCol, "vector size and number of threads row should be equal");
  static_assert(std::is_same_v<LayoutSFD, cutlass::layout::ColumnMajor> &&
                    std::is_same_v<LayoutOutput, cutlass::layout::ColumnMajor>,
                "Only support Mx1 (ColumnMajor) output and ColumnMajor scaling factor");
  static_assert(std::is_same_v<ElementCompute, float>, "ElementCompute should be float type");
  static_assert(cutlass::sizeof_bits<ElementD>::value == 4, "Output should be FP4 type");
  static_assert(cutlass::sizeof_bits<ElementSFD>::value == 8, "ElementSFD should be FP8 type");
  static_assert(std::is_same_v<LayoutOutput, LayoutSFD>, "only support same layout for D and SFD");

  // Hardcode static_assert on threadshape 16x8 to avoid bug
  static_assert(kThreadsPerCol == 16, "thread shape col false");
  static_assert(kThreadsPerRow == 8, "thread shape row false");
  static_assert(kThreadCount == 128, "thread count false");

  struct Params
  {
    TensorRefD tensor_d;
    ElementSFD *scale_factor_d_ptr{nullptr};
    ElementCompute alpha{0};
    ElementCompute beta{0};
    float st{0};
    int64_t batch_stride_sfd{0};  // Add batch stride for SFD
    int64_t stride_d{0};          // Add stride for D tensor
  };

  /// Shared storage
  struct SharedStorage
  {
    // fp32
    // Each thread store one fp32
#if 1
    ElementAccumulator reduction_buffer[kThreadsPerCol];
#else
    ElementAccumulator reduction_buffer[kThreadCount];
#endif
    // Buffer for collecting 4-bit values for packed store
    uint8_t packed_buffer[kThreadsPerCol];
  };

  private:
  Params const &params_;
  SharedStorage &shared_storage_;
  float st_scale_down{0};

  public:
  CUTLASS_HOST_DEVICE GemvEpilogueWithScalingFactor(Params const &params, SharedStorage &shared_storage)
      : params_(params)
      , shared_storage_(shared_storage)
  {
    const float fp_subtype_max = static_cast<float>(cutlass::platform::numeric_limits<ElementD>::max());
    this->st_scale_down = this->params_.st / fp_subtype_max;
  }

  CUTLASS_DEVICE void operator()(ElementAccumulator frag_acc, ElementC frag_c, int batch_idx)
  {
    const int block_idx = blockIdx.x;
    const int thread_idx_col = threadIdx.x;
    const int thread_idx_row = threadIdx.y;

    const float st_scale_down = this->st_scale_down;
    const float st = this->params_.st;

    // Compute D offset using batch_idx and stride_d
    const int output_d_base_offset = blockIdx.x * blockDim.y;
    const int d_batch_offset = batch_idx * params_.stride_d;
    ElementD* output_ptr = &params_.tensor_d.at({output_d_base_offset + d_batch_offset, 0});
    uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(output_ptr);
    // For 8x16 thread layout, 1 thread per 128 threads write to sf d
    // Every block write one SFD to gmem
    const bool is_write_sfd_thread = (thread_idx_row == 0);

    // Calculate SFD offset using proper batch stride
    const int output_sfd_offset = (block_idx / 4) * 512 + block_idx % 4 + batch_idx * params_.batch_stride_sfd;

    auto reduction_buffer = shared_storage_.reduction_buffer;
    // fp32
    ElementAccumulator max_accum_row0 = ElementAccumulator(0);
    ElementAccumulator max_accum_row1 = ElementAccumulator(0);

    // Thread in row contain duplicate frag_acc data
    if ( thread_idx_col == 0 ) {
      // 16 threads write to 16 contigious bank, no conflict
      reduction_buffer[thread_idx_row] = frag_acc;
    }

    __syncthreads();

    if (threadIdx.y == 0) {
      auto acc_0 = reduction_buffer[threadIdx.x * 2];
      auto acc_1 = reduction_buffer[threadIdx.x * 2 + 1];
      // compute the max for me using shuffling among 16 threads.
      ElementAccumulator max_accum = fabsf(acc_0);
      max_accum = cutlass::fast_max(max_accum, fabsf(acc_1));
      
      // Butterfly reduction pattern for 16 threads
      // Each iteration halves the number of active lanes
      max_accum = cutlass::fast_max(max_accum, __shfl_down_sync(0xFF, max_accum, 4));  // 8->4  
      max_accum = cutlass::fast_max(max_accum, __shfl_down_sync(0xFF, max_accum, 2));  // 4->2
      max_accum = cutlass::fast_max(max_accum, __shfl_down_sync(0xFF, max_accum, 1));  // 2->1
      
      // Broadcast the final result to all 8 threads
      max_accum = __shfl_sync(0xFF, max_accum, 0);

      float pvscale = max_accum * st_scale_down;
      ElementSFD qpvscale = static_cast<ElementSFD>(pvscale);
      float qpvscale_up = NumericConverter<ElementCompute, ElementSFD>{}(qpvscale);
      float qpvscale_up_rcp = __frcp_rn(qpvscale_up) * st;
      uint8_t qval_u8_compare;

      #if defined(CUDA_PTX_FP4FP6_CVT_ENABLED)
        uint32_t temp_result;
        asm volatile (
            "{\n"
            "  .reg .f32 output_fp32_0, output_fp32_1;\n"
            "  .reg .b8 byte0, byte1, byte2, byte3;\n"
            "  mul.f32 output_fp32_0, %1, %3;\n"
            "  mul.f32 output_fp32_1, %2, %3;\n"
            "  cvt.rn.satfinite.e2m1x2.f32 byte0, output_fp32_1, output_fp32_0;\n"
            "  mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
            "}\n"
            : "=r"(temp_result)                             // Output to uint32_t
            : "f"(acc_0), "f"(acc_1), "f"(qpvscale_up_rcp)
        );
        qval_u8_compare = temp_result & 0xFF;
      #else
        ElementD output_fp4_0 = NumericConverter<ElementD, ElementCompute>{}(acc_0 * qpvscale_up_rcp);
        ElementD output_fp4_1 = NumericConverter<ElementD, ElementCompute>{}(acc_1 * qpvscale_up_rcp);
        uint8_t raw_fp4_0 = reinterpret_cast<const uint8_t&>(output_fp4_0) & 0x0F;
        uint8_t raw_fp4_1 = reinterpret_cast<const uint8_t&>(output_fp4_1) & 0x0F;
        qval_u8_compare = (raw_fp4_1 << 4) | raw_fp4_0;
      #endif
      byte_ptr[threadIdx.x] = qval_u8_compare;

      arch::global_store<ElementSFD, sizeof(ElementSFD)>(qpvscale,
                                                        (void *)(params_.scale_factor_d_ptr + output_sfd_offset),
                                                        is_write_sfd_thread);

    }

  }  // end of operator()
};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass
