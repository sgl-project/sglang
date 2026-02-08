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
  \brief Kernel performing a reduction over one or more ranks of an affine tensor
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/fast_math.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/device_kernel.h"

#include "cutlass/reduction/device/tensor_reduce_affine_strided.h"
#include "cutlass/reduction/device/tensor_reduce_affine_contiguous.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reduction {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tensor reduction operator on specific CUTLASS layouts over exactly one index
template <
  typename ElementOutput_,
  typename ElementSource_,
  typename Layout_,
  typename ReductionOp_,
  int VectorLength_  = 1,
  typename ElementCompute_ = ElementOutput_
>
struct TensorReduction {

  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using Layout = Layout_;
  using ReductionOp = ReductionOp_;
  static int const kVectorLength = VectorLength_;
  using ElementCompute = ElementCompute_;

  using TensorCoord = typename Layout::TensorCoord;

  /// Reduction operator
  using ReductionDeviceStridedOperator = TensorReductionAffineStrided<
    4, 3, ElementOutput, ElementSource, ReductionOp, kVectorLength, ElementCompute
  >;

  using ReductionDeviceContiguousOperator = TensorReductionAffineContiguous<
    4, 3, ElementOutput, ElementSource, ReductionOp, kVectorLength, ElementCompute
  >;

  //
  // Data members
  //

  ReductionDeviceStridedOperator reduction_strided;
  ReductionDeviceContiguousOperator reduction_contiguous;
  int reduction_index;

  //
  // Methods
  //

  ///
  TensorReduction(
    TensorCoord extent, 
    int reduction_index_
  ): 
    reduction_index(reduction_index_) {

    Coord<4> extent_affine;

    switch (reduction_index) {
    case 0:
      extent_affine[0] = extent[1];
      extent_affine[1] = extent[2];
      extent_affine[2] = extent[0];
      extent_affine[3] = extent[3];
      break;
    case 1:
      extent_affine[0] = extent[0];
      extent_affine[1] = extent[2];
      extent_affine[2] = extent[1];
      extent_affine[3] = extent[3];
      break;
    case 2:
      extent_affine[0] = extent[0];
      extent_affine[1] = extent[1];
      extent_affine[2] = extent[2];
      extent_affine[3] = extent[3];
      break;
    case 3:
      extent_affine[0] = extent[0];
      extent_affine[1] = extent[1];
      extent_affine[2] = extent[2];
      extent_affine[3] = extent[3];
      break;
    default: break;
    }

    if (reduction_index == 3) {
      reduction_contiguous = ReductionDeviceContiguousOperator(extent_affine);  
    }
    else {
      reduction_strided = ReductionDeviceStridedOperator(extent_affine);  
    }
  }

  /// Simple check to verify the object is initialized correctly
  bool good() const {
    if (reduction_index == 3) {
      return reduction_contiguous.good();
    }
    return reduction_strided.good();
  }

  /// Size of one workspace
  int64_t workspace_stride() const {
    if (reduction_index == 3) {
      return reduction_contiguous.workspace_stride();
    }
    else {
      return reduction_strided.workspace_stride();
    }
  }

  /// Returns the size (in bytes) of a temporary workspace needed for reduction across CTAs
  int64_t workspace_size() const {
    if (reduction_index == 3) {
      return reduction_contiguous.workspace_size();
    }
    else {
      return reduction_strided.workspace_size();
    }
  }

  /// Helper to use overloaded function call operator
  Status reduce(
    TensorRef<ElementOutput, Layout> dst_ref,
    TensorRef<ElementSource, Layout> src_ref,
    void *device_workspace_ptr = nullptr,
    ElementCompute reduction_identity = ElementCompute(),
    ReductionOp reduction_op = ReductionOp(),
    cudaStream_t stream = nullptr) {

    int64_t src_stride[3];
    int64_t dst_stride[3];

    switch (reduction_index) {
    case 0:
      src_stride[0] = src_ref.stride()[1];
      src_stride[1] = src_ref.stride()[0];
      src_stride[2] = src_ref.stride()[2];
      dst_stride[0] = dst_ref.stride()[1];
      dst_stride[1] = dst_ref.stride()[0];
      break;
    case 1:
      src_stride[0] = src_ref.stride()[2];
      src_stride[1] = src_ref.stride()[0];
      src_stride[2] = src_ref.stride()[1];
      dst_stride[0] = dst_ref.stride()[2];
      dst_stride[1] = dst_ref.stride()[0];
      break;
    case 2:
      src_stride[0] = src_ref.stride()[2];
      src_stride[1] = src_ref.stride()[1];
      src_stride[2] = src_ref.stride()[0];
      dst_stride[0] = dst_ref.stride()[2];
      dst_stride[1] = dst_ref.stride()[1];
      break;
    case 3:
      src_stride[0] = src_ref.stride()[2];
      src_stride[1] = src_ref.stride()[1];
      src_stride[2] = src_ref.stride()[0];

      dst_stride[0] = dst_ref.stride()[2];
      dst_stride[1] = dst_ref.stride()[1];
      dst_stride[2] = dst_ref.stride()[0];

    default: break;
    }

    if (reduction_index == 3) {
      return reduction_contiguous(
        dst_ref.data(),
        dst_stride, 
        src_ref.data(), 
        src_stride, 
        device_workspace_ptr, 
        reduction_identity,
        reduction_op, 
        stream);
    }
    else {
      return reduction_strided(
        dst_ref.data(),
        dst_stride, 
        src_ref.data(), 
        src_stride, 
        device_workspace_ptr, 
        reduction_identity,
        reduction_op, 
        stream);
    }
  }

  Status operator()(
    TensorRef<ElementOutput, Layout> dst_ref,
    TensorRef<ElementSource, Layout> src_ref,
    void *device_workspace_ptr = nullptr,
    ElementCompute reduction_identity = ElementCompute(),
    ReductionOp reduction_op = ReductionOp(),
    cudaStream_t stream = nullptr) {

    return reduce(
      dst_ref, 
      src_ref, 
      device_workspace_ptr, 
      reduction_identity,
      reduction_op, 
      stream);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace reduction
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

