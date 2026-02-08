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
    \brief Defines additional layout functions used in Permute GEMM example to simplify
    computing reference permutations of 4/5D tensors when source data is column-major.
*/
#pragma once
#include "cutlass/cutlass.h"
#include CUDA_STD_HEADER(cassert)
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/coord.h"
#include "cutlass/tensor_coord.h"

namespace cutlass {
namespace layout {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Mapping function for 4-D CWHN tensors.
class TensorCWHN {
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

  /// Stride data member - [n, hn, whn]
  Stride stride_;

public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorCWHN(Stride const &stride = Stride(0)): stride_(stride) { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorCWHN(
    typename Stride::Index stride_h,    ///< number of elements between adjacent N coordinates
    typename Stride::Index stride_w,    ///< number of elements between adjacent C coordinates
    typename Stride::Index stride_c     ///< number of elements between adjacent W coordinates
  ): 
    stride_(make_Coord(stride_h, stride_w, stride_c)) { }

  /// Constructor
  // Once convolutions implement 64b stride this ctor can be deleted
  CUTLASS_HOST_DEVICE
  TensorCWHN(Coord<kStrideRank, LongIndex> const &stride): 
    stride_(make_Coord(
      static_cast<typename Stride::Index>(stride[0]), 
      static_cast<typename Stride::Index>(stride[1]), 
      static_cast<typename Stride::Index>(stride[2]))
    ) { }

  /// Helper returns a layout to a tightly packed WCNH tensor.
  CUTLASS_HOST_DEVICE
  static TensorCWHN packed(TensorCoord const &extent) {
    return TensorCWHN(
      make_Coord(
        extent.n(), 
        extent.h() * extent.n(),
        extent.w() * extent.h() * extent.n()
      )
    );
  }
  
  /// Returns the offset of a coordinate (n, h, w, c) in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return coord.n() + 
      LongIndex(stride_[0] * coord.h()) + 
      LongIndex(stride_[1] * coord.w()) +
      LongIndex(stride_[2] * coord.c());
  }
  
  /// Returns the offset of a pitchlinear coordinate in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const {
    return coord.contiguous() + LongIndex(coord.strided() * stride_[2]);
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
    if ((extent.n() > stride_[0])
        || (extent.h() * stride_[0] > stride_[1]) 
        || (extent.w() * stride_[1] > stride_[2])) {
      assert(0);
    }
    return extent.c() * stride_[2];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Mapping function for 4-D NHCW tensors.
class TensorNHCW {
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

  /// Stride data member - [w, cw, hcw]
  Stride stride_;

public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorNHCW(Stride const &stride = Stride(0)): stride_(stride) { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorNHCW(
    typename Stride::Index stride_c,    ///< number of elements between adjacent C coordinates
    typename Stride::Index stride_h,    ///< number of elements between adjacent H coordinates
    typename Stride::Index stride_n     ///< number of elements between adjacent N coordinates
  ): 
    stride_(make_Coord(stride_c, stride_h, stride_n)) { }

  /// Constructor
  // Once convolutions implement 64b stride this ctor can be deleted
  CUTLASS_HOST_DEVICE
  TensorNHCW(Coord<kStrideRank, LongIndex> const &stride): 
    stride_(make_Coord(
      static_cast<typename Stride::Index>(stride[0]), 
      static_cast<typename Stride::Index>(stride[1]), 
      static_cast<typename Stride::Index>(stride[2]))
    ) { }

  /// Helper returns a layout to a tightly packed WCNH tensor.
  CUTLASS_HOST_DEVICE
  static TensorNHCW packed(TensorCoord const &extent) {
    return TensorNHCW(
      make_Coord(
        extent.w(), 
        extent.c() * extent.w(),
        extent.h() * extent.c() * extent.w()
      )
    );
  }
  
  /// Returns the offset of a coordinate (n, h, w, c) in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return coord.w() + 
      LongIndex(stride_[0] * coord.c()) + 
      LongIndex(stride_[1] * coord.h()) +
      LongIndex(stride_[2] * coord.n());
  }
  
  /// Returns the offset of a pitchlinear coordinate in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const {
    return coord.contiguous() + LongIndex(coord.strided() * stride_[2]);
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
    if ((extent.w() > stride_[0])
        || (extent.c() * stride_[0] > stride_[1]) 
        || (extent.h() * stride_[1] > stride_[2])) {
      assert(0);
    }
    return extent.n() * stride_[2];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Mapping function for 4-D NHCW tensors.
class TensorNCWH {
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

  /// Stride data member - [h, wh, cwh]
  Stride stride_;

public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorNCWH(Stride const &stride = Stride(0)): stride_(stride) { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorNCWH(
    typename Stride::Index stride_w,    ///< number of elements between adjacent C coordinates
    typename Stride::Index stride_c,    ///< number of elements between adjacent H coordinates
    typename Stride::Index stride_n     ///< number of elements between adjacent N coordinates
  ): 
    stride_(make_Coord(stride_w, stride_c, stride_n)) { }

  /// Constructor
  // Once convolutions implement 64b stride this ctor can be deleted
  CUTLASS_HOST_DEVICE
  TensorNCWH(Coord<kStrideRank, LongIndex> const &stride): 
    stride_(make_Coord(
      static_cast<typename Stride::Index>(stride[0]), 
      static_cast<typename Stride::Index>(stride[1]), 
      static_cast<typename Stride::Index>(stride[2]))
    ) { }

  /// Helper returns a layout to a tightly packed WCNH tensor.
  CUTLASS_HOST_DEVICE
  static TensorNCWH packed(TensorCoord const &extent) {
    return TensorNCWH(
      make_Coord(
        extent.h(), 
        extent.w() * extent.h(),
        extent.c() * extent.w() * extent.h()
      )
    );
  }
  
  /// Returns the offset of a coordinate (n, h, w, c) in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return coord.h() + 
      LongIndex(stride_[0] * coord.w()) + 
      LongIndex(stride_[1] * coord.c()) +
      LongIndex(stride_[2] * coord.n());
  }
  
  /// Returns the offset of a pitchlinear coordinate in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const {
    return coord.contiguous() + LongIndex(coord.strided() * stride_[2]);
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
    if ((extent.h() > stride_[0])
        || (extent.w() * stride_[0] > stride_[1]) 
        || (extent.c() * stride_[1] > stride_[2])) {
      assert(0);
    }
    return extent.n() * stride_[2];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Mapping function for 5-D CWHDN tensors.
class TensorCWHDN {
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

  /// Stride data member - [n, dn, hdn, whdn]
  Stride stride_;

public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorCWHDN(Stride const &stride = Stride(0)): stride_(stride) { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorCWHDN(
    typename Stride::Index n, 
    typename Stride::Index dn, 
    typename Stride::Index hdn, 
    typename Stride::Index whdn): 
  stride_(make_Coord(n, dn, hdn, whdn)) { }

  /// Constructor
  // Once convolutions implement 64b stride this ctor can be deleted
  CUTLASS_HOST_DEVICE
  TensorCWHDN(Coord<kStrideRank, LongIndex> const &stride): 
    stride_(make_Coord(
      static_cast<typename Stride::Index>(stride[0]), 
      static_cast<typename Stride::Index>(stride[1]), 
      static_cast<typename Stride::Index>(stride[2]),
      static_cast<typename Stride::Index>(stride[3]))
    ) { }

  /// Helper returns a layout to a tightly packed CWHDN tensor.
  CUTLASS_HOST_DEVICE
  static TensorCWHDN packed(TensorCoord const &extent) {
    return TensorCWHDN(
      make_Coord(
        extent.n(), 
        extent.d() * extent.n(),
        extent.h() * extent.d() * extent.n(),
        extent.w() * extent.h() * extent.d() * extent.n()
      )
    );
  }
  
  /// Returns the offset of a coordinate (n, d, h, w, c) in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return coord.n() + 
      LongIndex(stride_[0] * coord.d()) + 
      LongIndex(stride_[1] * coord.h()) +
      LongIndex(stride_[2] * coord.w()) +
      LongIndex(stride_[3] * coord.c());
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
    if ((extent.n() > stride_[0])
        || (extent.d() * stride_[0] > stride_[1]) 
        || (extent.h() * stride_[1] > stride_[2])
        || (extent.w() * stride_[2] > stride_[3])) {
      assert(0);
    }
    return extent.c() * stride_[3];
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace layout
} // namespace cutlass
