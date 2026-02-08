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
    \brief Defines layout functions used by TensorRef and derived classes for common 4-D and 5-D
      tensor formats.

    Layout functions map logical coordinates to linear memory. They often require additional
    data to describe strides between elements.

    Layout functions must implement all members in the public interface of IdentityTensorLayout<>
    defined in cutlass/tensor_ref.h.
*/
#pragma once
#include "cutlass/cutlass.h"
#include CUDA_STD_HEADER(cassert)
#include "cutlass/fast_math.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/coord.h"
#include "cutlass/tensor_coord.h"

namespace cutlass {
namespace layout {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines data layouts of various tensor formats usable by TensorRef and other classes.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag used for 3-D NWC tensors for 1-D convolutions; only used in 3.x API
class TensorNWC {};

/// Tag used for n-D KCSRT tensors for n-D convolutions; only used in 3.x API for wgrad output layouts
class TensorKCS {};
class TensorKCSR {};
class TensorKCSRT {};

/// Tag used for n-D CSRTK tensors for n-D convolutions; only used in 3.x API for wgrad output layouts
class TensorCSK {};
class TensorCSRK {};
class TensorCSRTK {};

/// Mapping function for 4-D NHWC tensors.
class TensorNHWC {
public:
  /// Logical rank of tensor
  static int const kRank = 4;

  /// Rank of stride vector
  static int const kStrideRank = 3;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate (n, h, w, c)
  using TensorCoord = Tensor4DCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank>;

private:
  //
  // Data members
  //

  /// Stride data member - [stride_w, stride_h, stride_n]
  Stride stride_;

public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorNHWC(Stride const &stride = Stride(0)): stride_(stride) { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorNHWC(
    typename Stride::Index stride_w,    ///< number of elements between adjacent W coordinates
    typename Stride::Index stride_h,    ///< number of elements between adjacent H coordinates
    typename Stride::Index stride_n     ///< number of elements between adjacent N coordinates
  ): 
    stride_(make_Coord(stride_w, stride_h, stride_n)) { }

  /// Constructor
  // Once convolutions implement 64b stride this ctor can be deleted
  CUTLASS_HOST_DEVICE
  TensorNHWC(Coord<kStrideRank, LongIndex> const &stride): 
    stride_(make_Coord(
      static_cast<typename Stride::Index>(stride[0]), 
      static_cast<typename Stride::Index>(stride[1]), 
      static_cast<typename Stride::Index>(stride[2]))
    ) { }

  /// Helper returns a layout to a tightly packed NHWC tensor.
  CUTLASS_HOST_DEVICE
  static TensorNHWC packed(TensorCoord const &extent) {
    return TensorNHWC(
      make_Coord(
        extent.c(), 
        extent.w() * extent.c(),
        extent.h() * extent.w() * extent.c()
      )
    );
  }
  
  /// Returns the offset of a coordinate (n, h, w, c) in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return coord.c() + 
      LongIndex(stride_[0] * coord.w()) + 
      LongIndex(stride_[1] * coord.h()) +
      LongIndex(stride_[2] * coord.n());
  }
  
  /// Returns the offset of a pitchlinear coordinate in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const {
    return coord.contiguous() + LongIndex(coord.strided() * stride_[2]);
  }

  /// Returns the logical coordinate (n, h, w, c) from a given offset in linear memory.
  CUTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex index) const {

    int n = 0, h = 0, w = 0, c = 0;

    #if defined(__CUDA_ARCH__)
    int tmp = 0;
    c = int(index % static_cast<int>(stride_[0]));

    unsigned int hw_mul, hw_shr, w_mul, w_shr, c_mul, c_shr;

    find_divisor(hw_mul, hw_shr, stride_[2]);
    find_divisor(w_mul, w_shr, stride_[1]);
    find_divisor(c_mul, c_shr, stride_[0]);

    fast_divmod(n, tmp, index, int(stride_[2]), hw_mul, hw_shr);
    fast_divmod(h, w, tmp, int(stride_[1]), w_mul, w_shr);
    fast_divmod(w, tmp, w, int(stride_[0]), c_mul, c_shr);
    #else

    n = int(index / stride_[2]);
    LongIndex residual = index % stride_[2];

    h = int(residual / stride_[1]);
    residual = (residual % stride_[1]);

    w = int(residual / stride_[0]);
    c = int(residual % stride_[0]);

    #endif
    return TensorCoord(n, h, w, c);
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const {
    return stride_;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride() {
    return stride_;
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    // it does not make sense if the extent is larger than stride
    // and we could not rely on the capacity calculation in such cases
    // we could move this checkers to debug code only
    if ((extent.c() > stride_[0])
        || (extent.w() * stride_[0] > stride_[1]) 
        || (extent.h() * stride_[1] > stride_[2])) {
      assert(0);
    }
    return extent.n() * stride_[2];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Mapping function for 4-D NCHW tensors.
class TensorNCHW {
public:
  /// Logical rank of tensor
  static int const kRank = 4;

  /// Rank of stride vector
  static int const kStrideRank = 3;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = Tensor4DCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank>;

private:
  //
  // Data members
  //

  /// Stride data member - [w, hw, chw]
  Stride stride_;

public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorNCHW(Stride const &stride = Stride(0)): stride_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static TensorNCHW packed(TensorCoord const &extent) {
    return TensorNCHW(
      make_Coord(
        extent.w(),
        extent.w() * extent.h(),
        extent.h() * extent.w() * extent.c()
      )
    );
  }

  /// Returns the offset of a coordinate in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return coord.w() + 
      LongIndex(stride_[0] * coord.h()) + 
      LongIndex(stride_[1] * coord.c()) + 
      LongIndex(stride_[2] * coord.n());
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const {
    return stride_;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride() {
    return stride_;
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return extent.n() * stride_[2];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Mapping function for 4-D NC/xHWx tensors.
template <int Interleave>
class TensorNCxHWx {
public:

  /// Interleaving quantity
  static int const kInterleave = Interleave;

  /// Logical rank of tensor
  static int const kRank = 4;

  /// Rank of stride vector
  static int const kStrideRank = 3;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = Tensor4DCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank>;

private:
  //
  // Data members
  //

  /// Stride data member - [Interleave x w, Interleave x wh, hwc]
  Stride stride_;

public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorNCxHWx(Stride const &stride = Stride(0)): stride_(stride) { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorNCxHWx(
    typename Stride::Index stride_w,    ///< number of elements between adjacent W coordinates
    typename Stride::Index stride_h,    ///< number of elements between adjacent H coordinates
    typename Stride::Index stride_n     ///< number of elements between adjacent N coordinates
  ):
    stride_(make_Coord(stride_w, stride_h, stride_n)) { }

  /// Constructor
  // Once convolutions implement 64b stride this ctor can be deleted
  CUTLASS_HOST_DEVICE
  TensorNCxHWx(Coord<kStrideRank, LongIndex> const &stride): 
    stride_(make_Coord(
      static_cast<typename Stride::Index>(stride[0]), 
      static_cast<typename Stride::Index>(stride[1]), 
      static_cast<typename Stride::Index>(stride[2]))
    ) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static TensorNCxHWx packed(TensorCoord const &extent) {
    return TensorNCxHWx(
      make_Coord(
        kInterleave * extent.w(),
        kInterleave * extent.w() * extent.h(),
        extent.h() * extent.w() * extent.c()
      )
    );
  }

  /// Returns the offset of a coordinate in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {

    Index c_minor = (coord.c() % kInterleave);
    Index c_major = (coord.c() / kInterleave);

    return c_minor + 
      LongIndex(kInterleave * coord.w()) + 
      LongIndex(stride_[0] * coord.h()) + 
      LongIndex(stride_[1] * c_major) + 
      LongIndex(stride_[2] * coord.n());
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const {
    return stride_;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride() {
    return stride_;
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return extent.n() * stride_[2];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Mapping function for 4-D CxRSKx tensors.
template <int Interleave>
class TensorCxRSKx {
public:

  /// Interleaving quantity
  static int const kInterleave = Interleave;

  /// Logical rank of tensor
  static int const kRank = 4;

  /// Rank of stride vector
  static int const kStrideRank = 3;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = Tensor4DCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank>;

private:
  //
  // Data members
  //

  /// Stride data member - [Interleave x n, Interleave x nw, Interleave x nwh]
  Stride stride_;

public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorCxRSKx(Stride const &stride = Stride(0)): stride_(stride) { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorCxRSKx(
    typename Stride::Index stride_w,    ///< number of elements between adjacent W coordinates
    typename Stride::Index stride_h,    ///< number of elements between adjacent H coordinates
    typename Stride::Index stride_n     ///< number of elements between adjacent N coordinates
  ):
    stride_(make_Coord(stride_w, stride_h, stride_n)) { }

  /// Constructor
  // Once convolutions implement 64b stride this ctor can be deleted
  CUTLASS_HOST_DEVICE
  TensorCxRSKx(Coord<kStrideRank, LongIndex> const &stride): 
    stride_(make_Coord(
      static_cast<typename Stride::Index>(stride[0]), 
      static_cast<typename Stride::Index>(stride[1]), 
      static_cast<typename Stride::Index>(stride[2]))
    ) { }


  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static TensorCxRSKx packed(TensorCoord const &extent) {
    return TensorCxRSKx(
      make_Coord(
        kInterleave * extent.n(),
        kInterleave * extent.n() * extent.w(),
        kInterleave * extent.n() * extent.w() * extent.h()
      )
    );
  }

  /// Returns the offset of a coordinate in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {

    Index c_minor = (coord.c() % kInterleave);
    Index c_major = (coord.c() / kInterleave);

    return c_minor + 
      LongIndex(kInterleave * coord.n()) + 
      LongIndex(stride_[0] * coord.w()) + 
      LongIndex(stride_[1] * coord.h()) + 
      LongIndex(stride_[2] * c_major);
  }

  /// Returns the offset of a pitchlinear coordinate in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord const &coord) const {
    return (coord.contiguous() % kInterleave) +
      LongIndex((coord.contiguous() / kInterleave) * stride_[2]) +
      LongIndex(coord.strided() * kInterleave);
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const {
    return stride_;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride() {
    return stride_;
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return (extent.c() / kInterleave * stride_[2]);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Mapping function for 5-D NDHWC tensors.
class TensorNDHWC {
public:
  /// Logical rank of tensor
  static int const kRank = 5;

  /// Rank of stride vector
  static int const kStrideRank = 4;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate (n, d, h, w, c)
  using TensorCoord = Tensor5DCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank>;

private:
  //
  // Data members
  //

  /// Stride data member - [c, wc, hwc, dhwc]
  Stride stride_;

public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorNDHWC(Stride const &stride = Stride(0)): stride_(stride) { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorNDHWC(
    typename Stride::Index c, 
    typename Stride::Index wc, 
    typename Stride::Index hwc, 
    typename Stride::Index dhwc): 
  stride_(make_Coord(c, wc, hwc, dhwc)) { }

  /// Constructor
  // Once convolutions implement 64b stride this ctor can be deleted
  CUTLASS_HOST_DEVICE
  TensorNDHWC(Coord<kStrideRank, LongIndex> const &stride): 
    stride_(make_Coord(
      static_cast<typename Stride::Index>(stride[0]), 
      static_cast<typename Stride::Index>(stride[1]), 
      static_cast<typename Stride::Index>(stride[2]),
      static_cast<typename Stride::Index>(stride[3]))
    ) { }

  /// Helper returns a layout to a tightly packed NHWC tensor.
  CUTLASS_HOST_DEVICE
  static TensorNDHWC packed(TensorCoord const &extent) {
    return TensorNDHWC(
      make_Coord(
        extent.c(), 
        extent.w() * extent.c(),
        extent.h() * extent.w() * extent.c(),
        extent.d() * extent.h() * extent.w() * extent.c()
      )
    );
  }
  
  /// Returns the offset of a coordinate (n, d, h, w, c) in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return coord.c() + 
      LongIndex(stride_[0] * coord.w()) + 
      LongIndex(stride_[1] * coord.h()) +
      LongIndex(stride_[2] * coord.d()) +
      LongIndex(stride_[3] * coord.n());
  }

  /// Returns the offset of a pitchlinear coordinate in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const {
    return coord.contiguous() + LongIndex(coord.strided() * stride_[3]);
  }
  
  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const {
    return stride_;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride() {
    return stride_;
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    // it does not make sense if the extent is larger than stride
    // and we could not rely on the capacity calculation in such cases
    // we could move this checkers to debug code only
    if ((extent.c() > stride_[0])
        || (extent.w() * stride_[0] > stride_[1]) 
        || (extent.h() * stride_[1] > stride_[2])
        || (extent.d() * stride_[2] > stride_[3])) {
      assert(0);
    }
    return extent.n() * stride_[3];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace layout
} // namespace cutlass
