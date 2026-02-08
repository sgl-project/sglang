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
/*! \file
    \brief This file contains definitions and utility functions for describing convolution problem shapes.
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/conv/convolution.h"

#include "cute/container/array.hpp"

#if ! defined(__CUDACC_RTC__)
#include <initializer_list>
#endif


////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv {

////////////////////////////////////////////////////////////////////////////////////////////////////

// Implements the user facing argument for all CUTLASS 3.x convolutions in a rank agnostic fashion.
// All tensors are flat and by default treated as layout right (NDHWC, KTRSC, NZPQK)
// Supports asymmetric padding, traversal strides, dilations, and all conv algorithm types.
template <
  conv::Operator ConvOp_,
  int NumSpatialDimensions_
>
struct ConvProblemShape {
  //
  // Alias types for members
  //

  static constexpr int RankS = NumSpatialDimensions_;
  static constexpr int RankT = NumSpatialDimensions_ + 2;
  static constexpr conv::Operator ConvOp = ConvOp_;
  static constexpr int NumSpatialDimensions = NumSpatialDimensions_;
  using SpatialExtent = cute::array<int, RankS>;
  using TensorExtent  = cute::array<int, RankT>;
  using TensorStride  = cute::array<int64_t, RankT>;
  using ShapePadding = SpatialExtent;
  using TraversalStride = SpatialExtent;
  using ShapeDilation = SpatialExtent;
  using Corner = SpatialExtent;

  //
  // Members
  //
  cutlass::conv::Mode mode{};
  TensorExtent shape_A{};
  TensorStride stride_A{};
  TensorExtent shape_B{};
  TensorStride stride_B{};
  TensorExtent shape_C{};
  TensorStride stride_C{};

  // asymmetric padding, both upper and lower padding must be >= 0
  ShapePadding lower_padding{};
  ShapePadding upper_padding{};
  TraversalStride traversal_stride{};
  ShapeDilation dilation{};
  int groups = 1;

  //
  // Methods
  //

  ConvProblemShape() = default;

  // Constructor accepts user facing arguments and computes to stores the corners as its internal state
  ConvProblemShape(
      conv::Mode mode,                                                     // convolution/cross-correlation
      TensorExtent shape_act,                                              // [n,d,h,w,c]
      TensorStride stride_act,                                             // [n,d,h,w,c]
      TensorExtent shape_flt,                                              // [k,t,r,s,c]
      TensorStride stride_flt,                                             // [k,t,r,s,c]
      ShapePadding lower_padding,                                          // [pad_d, pad_h, pad_w]
      ShapePadding upper_padding,                                          // [pad_d, pad_h, pad_w]
      TraversalStride tstride,                                             // [stride_d, stride_h, stride_w]
      ShapeDilation dilation,                                              // [dilation_d, dilation_h, dilation_w]
      int groups)
      : mode(mode)
      , lower_padding(lower_padding)
      , upper_padding(upper_padding)
      , traversal_stride(tstride)
      , dilation(dilation)
      , groups(groups) {

    auto [shape_xformed_act, stride_xformed_act] = calculate_xformed_act(shape_act, shape_flt);
    set_shape_stride_ABC(shape_act, stride_act, shape_flt, stride_flt, shape_xformed_act, stride_xformed_act);
  }

  // Allow user input of xformed activation stride to support non-packed strides.
  ConvProblemShape(
      conv::Mode mode,                                                     // convolution/cross-correlation
      TensorExtent shape_act,                                              // [n,d,h,w,c]
      TensorStride stride_act,                                             // [n,d,h,w,c]
      TensorExtent shape_flt,                                              // [k,t,r,s,c]
      TensorStride stride_flt,                                             // [k,t,r,s,c]
      TensorStride stride_xformed_act,                                     // [n,z,p,q,k]
      ShapePadding lower_padding,                                          // [pad_d, pad_h, pad_w]
      ShapePadding upper_padding,                                          // [pad_d, pad_h, pad_w]
      TraversalStride tstride,                                             // [stride_d, stride_h, stride_w]
      ShapeDilation dilation,                                              // [dilation_d, dilation_h, dilation_w]
      int groups)
      : mode(mode)
      , lower_padding(lower_padding)
      , upper_padding(upper_padding)
      , traversal_stride(tstride)
      , dilation(dilation)
      , groups(groups) {

    CUTLASS_ASSERT(stride_act[RankT - 1] == 1);
    CUTLASS_ASSERT(stride_flt[RankT - 1] == 1);
    CUTLASS_ASSERT(stride_xformed_act[RankT - 1] == 1);

    auto stride_act_packed = packed_stride_right_major(shape_act);
    auto stride_flt_packed = packed_stride_right_major(shape_flt);
    auto [shape_xformed_act, stride_xformed_act_packed] = calculate_xformed_act(shape_act, shape_flt);

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < RankT - 1; ++i) {
      CUTLASS_ASSERT(stride_act[i] >= stride_act_packed[i]);
      CUTLASS_ASSERT(stride_flt[i] >= stride_flt_packed[i]);
      CUTLASS_ASSERT(stride_xformed_act[i] >= stride_xformed_act_packed[i]);
    }

    set_shape_stride_ABC(shape_act, stride_act, shape_flt, stride_flt, shape_xformed_act, stride_xformed_act);
  }

  // Constructor accepts user facing arguments and presume packed tensor strides in canonical (CWHDN) order.
  ConvProblemShape(
      conv::Mode mode,
      TensorExtent shape_act,
      TensorExtent shape_flt,
      ShapePadding lower_padding,
      ShapePadding upper_padding,
      TraversalStride tstride,
      ShapeDilation dilation,
      int groups)
      : ConvProblemShape(
        mode,
        shape_act,
        packed_stride_right_major(shape_act),
        shape_flt,
        packed_stride_right_major(shape_flt),
        lower_padding,
        upper_padding,
        tstride,
        dilation,
        groups) {
    }

#if ! defined(__CUDACC_RTC__)
  // Constructor accepts user facing arguments and computes to stores the corners as its internal state
  ConvProblemShape(
      conv::Mode                     mode,
      std::initializer_list<int>     shape_act_,
      std::initializer_list<int64_t> stride_act_,
      std::initializer_list<int>     shape_flt_,
      std::initializer_list<int64_t> stride_flt_,
      std::initializer_list<int>     lower_padding_,
      std::initializer_list<int>     upper_padding_,
      std::initializer_list<int>     traversal_stride_,
      std::initializer_list<int>     dilation_,
      int groups)
      : mode(mode)
      , groups(groups) {

    TensorExtent shape_act{};
    TensorStride stride_act{};
    TensorExtent shape_flt{};
    TensorStride stride_flt{};

    assert(shape_act_.size() == shape_act.size());
    assert(stride_act_.size() == stride_act.size());
    assert(shape_flt_.size() == shape_flt.size());
    assert(stride_flt_.size() == stride_flt.size());
    assert(lower_padding_.size() == lower_padding.size());
    assert(upper_padding_.size() == upper_padding.size());
    assert(traversal_stride_.size() == traversal_stride.size());
    assert(dilation_.size() == dilation.size());

    std::copy(shape_act_.begin(), shape_act_.end(), shape_act.begin());
    std::copy(stride_act_.begin(), stride_act_.end(), stride_act.begin());
    std::copy(shape_flt_.begin(), shape_flt_.end(), shape_flt.begin());
    std::copy(stride_flt_.begin(), stride_flt_.end(), stride_flt.begin());
    std::copy(lower_padding_.begin(), lower_padding_.end(), lower_padding.begin());
    std::copy(upper_padding_.begin(), upper_padding_.end(), upper_padding.begin());
    std::copy(traversal_stride_.begin(), traversal_stride_.end(), traversal_stride.begin());
    std::copy(dilation_.begin(), dilation_.end(), dilation.begin());

    auto [shape_xformed_act, stride_xformed_act] = calculate_xformed_act(shape_act, shape_flt);
    set_shape_stride_ABC(shape_act, stride_act, shape_flt, stride_flt, shape_xformed_act, stride_xformed_act);
  }

  // Allow user input of xformed activation stride to support non-packed strides.
  ConvProblemShape(
      conv::Mode                     mode,
      std::initializer_list<int>     shape_act_,
      std::initializer_list<int64_t> stride_act_,
      std::initializer_list<int>     shape_flt_,
      std::initializer_list<int64_t> stride_flt_,
      std::initializer_list<int64_t> stride_xformed_act_,
      std::initializer_list<int>     lower_padding_,
      std::initializer_list<int>     upper_padding_,
      std::initializer_list<int>     traversal_stride_,
      std::initializer_list<int>     dilation_,
      int groups)
      : mode(mode)
      , groups(groups) {
    TensorExtent shape_act{};
    TensorStride stride_act{};
    TensorExtent shape_flt{};
    TensorStride stride_flt{};
    TensorStride stride_xformed_act{};

    std::copy(shape_act_.begin(), shape_act_.end(), shape_act.begin());
    std::copy(stride_act_.begin(), stride_act_.end(), stride_act.begin());
    std::copy(shape_flt_.begin(), shape_flt_.end(), shape_flt.begin());
    std::copy(stride_flt_.begin(), stride_flt_.end(), stride_flt.begin());
    std::copy(stride_xformed_act_.begin(), stride_xformed_act_.end(), stride_xformed_act.begin());
    std::copy(lower_padding_.begin(), lower_padding_.end(), lower_padding.begin());
    std::copy(upper_padding_.begin(), upper_padding_.end(), upper_padding.begin());
    std::copy(traversal_stride_.begin(), traversal_stride_.end(), traversal_stride.begin());
    std::copy(dilation_.begin(), dilation_.end(), dilation.begin());

    CUTLASS_ASSERT(stride_act[RankT - 1] == 1);
    CUTLASS_ASSERT(stride_flt[RankT - 1] == 1);
    CUTLASS_ASSERT(stride_xformed_act[RankT - 1] == 1);

    auto stride_act_packed = packed_stride_right_major(shape_act);
    auto stride_flt_packed = packed_stride_right_major(shape_flt);
    auto [shape_xformed_act, stride_xformed_act_packed] = calculate_xformed_act(shape_act, shape_flt);

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < RankT - 1; ++i) {
      CUTLASS_ASSERT(stride_act[i] >= stride_act_packed[i]);
      CUTLASS_ASSERT(stride_flt[i] >= stride_flt_packed[i]);
      CUTLASS_ASSERT(stride_xformed_act[i] >= stride_xformed_act_packed[i]);
    }

    set_shape_stride_ABC(shape_act, stride_act, shape_flt, stride_flt, shape_xformed_act, stride_xformed_act);
  }

  // Constructor accepts user facing arguments and computes to stores the corners as its internal state
  ConvProblemShape(
      conv::Mode                     mode,
      std::initializer_list<int>     shape_act_,
      std::initializer_list<int>     shape_flt_,
      std::initializer_list<int>     lower_padding_,
      std::initializer_list<int>     upper_padding_,
      std::initializer_list<int>     traversal_stride_,
      std::initializer_list<int>     dilation_,
      int groups)
      : mode(mode)
      , groups(groups) {
    TensorExtent shape_act{};
    TensorStride stride_act{};
    TensorExtent shape_flt{};
    TensorStride stride_flt{};

    assert(shape_act_.size() == shape_act.size());
    assert(shape_flt_.size() == shape_flt.size());
    assert(lower_padding_.size() == lower_padding.size());
    assert(upper_padding_.size() == upper_padding.size());
    assert(traversal_stride_.size() == traversal_stride.size());
    assert(dilation_.size() == dilation.size());

    std::copy(shape_act_.begin(), shape_act_.end(), shape_act.begin());
    std::copy(shape_flt_.begin(), shape_flt_.end(), shape_flt.begin());
    std::copy(lower_padding_.begin(), lower_padding_.end(), lower_padding.begin());
    std::copy(upper_padding_.begin(), upper_padding_.end(), upper_padding.begin());
    std::copy(traversal_stride_.begin(), traversal_stride_.end(), traversal_stride.begin());
    std::copy(dilation_.begin(), dilation_.end(), dilation.begin());
    stride_act = packed_stride_right_major(shape_act);
    stride_flt = packed_stride_right_major(shape_flt);

    auto [shape_xformed_act, stride_xformed_act] = calculate_xformed_act(shape_act, shape_flt);
    set_shape_stride_ABC(shape_act, stride_act, shape_flt, stride_flt, shape_xformed_act, stride_xformed_act);
  }
#endif // not defined(__CUDACC_RTC__)

  // Set shape and stride of tensor A/B/C according to following table:
  // |              | Fprop  | Dgrad  | Wgrad |
  // | ------       | ------ | ------ | ------|
  // |   ShapeA     | NDHWC  | NZPQK  | NZPQK |
  // |   ShapeB     | KTRSC  | KTRSC  | NDHWC |
  // |   ShapeC     | NZPQK  | NDHWC  | KTRSC |
  //
  // Input comes from calculate_xformed_act, which does NOT depend on ConvOp.
  CUTLASS_HOST_DEVICE
  constexpr void
  set_shape_stride_ABC(
    TensorExtent shape_act,
    TensorStride stride_act,
    TensorExtent shape_flt,
    TensorStride stride_flt,
    TensorExtent shape_xformed_act,
    TensorStride stride_xformed_act) {
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    printf("*** set_shape_stride_ABC ***");
    printf("\n  shape_act: ");
    print(shape_act);
    printf("\n  stride_act: ");
    print(stride_act);
    printf("\n  shape_flt: ");
    print(shape_flt);
    printf("\n  stride_flt: ");
    print(stride_flt);
    printf("\n  shape_xformed_act: ");
    print(shape_xformed_act);
    printf("\n  stride_xformed_act: ");
    print(stride_xformed_act);
    if constexpr (ConvOp == cutlass::conv::Operator::kFprop) {
      printf("\n  ConvOp: Fprop");
    }
    if constexpr (ConvOp == cutlass::conv::Operator::kDgrad) {
      printf("\n  ConvOp: Dgrad");
    }
    if constexpr (ConvOp == cutlass::conv::Operator::kWgrad) {
      printf("\n  ConvOp: Wgrad");
    }
    printf("\n");
#endif

    if constexpr (ConvOp == cutlass::conv::Operator::kFprop) {
      shape_A = shape_act;
      stride_A = stride_act;
      shape_B = shape_flt;
      stride_B = stride_flt;
      shape_C = shape_xformed_act;
      stride_C = stride_xformed_act;
    }
    else if constexpr (ConvOp == cutlass::conv::Operator::kDgrad) {
      shape_A = shape_xformed_act;
      stride_A = stride_xformed_act;
      shape_B = shape_flt;
      stride_B = stride_flt;
      shape_C = shape_act;
      stride_C = stride_act;
    }
    else if constexpr (ConvOp == cutlass::conv::Operator::kWgrad) {
      shape_A = shape_xformed_act;
      stride_A = stride_xformed_act;
      shape_B = shape_act;
      stride_B = stride_act;
      shape_C = shape_flt;
      stride_C = stride_flt;
    }
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    printf("\n  shape_A: ");
    print(shape_A);
    printf("\n  stride_A: ");
    print(stride_A);
    printf("\n  shape_B: ");
    print(shape_B);
    printf("\n  stride_B: ");
    print(stride_B);
    printf("\n  shape_C: ");
    print(shape_C);
    printf("\n  stride_C: ");
    print(stride_C);
#endif
  }

  // Get A extents.
  // fprop: A extents array contains [N,D,H,W,C]. Turn that into ((W,H,D,N), (C))
  // dgrad: A extents array contains [N,Z,P,Q,K]. Turn that into ((Q,P,Z,N), (K))
  // wgrad: A extents array contains [N,Z,P,Q,K]. Turn that into ((K), (Q,P,Z,N))
  CUTLASS_HOST_DEVICE
  constexpr auto
  get_shape_A() const {
    using cute::make_shape;
    using cute::take;

    if constexpr (ConvOp == conv::Operator::kFprop ||
                  ConvOp == conv::Operator::kDgrad) {
      return make_shape(
        cute::reverse(take<0, RankT - 1>(shape_A)),
        shape_A[RankT - 1]);
    }
    // For wgrad kernel, we need to linearize NZPQ for tensor A
    else if constexpr (ConvOp == conv::Operator::kWgrad) {
      return make_shape(
        shape_A[RankT - 1],
        cute::product(take<0, RankT - 1>(shape_A)));
    }
  }

  // Get B extents.
  // fprop: B extents array contains [K,T,R,S,C]. Turn that into ((K), (C,S,R,T))
  // dgrad: B extents array contains [K,T,R,S,C]. Turn that into ((C), (K,S,R,T))
  // wgrad: B extents array contains [N,D,H,W,C]. Turn that into ((C), (W,H,D,N))
  CUTLASS_HOST_DEVICE
  constexpr auto
  get_shape_B() const {
    using cute::make_shape;
    using cute::reverse;
    using cute::take;

    if constexpr (ConvOp == conv::Operator::kFprop) {
      return make_shape(
        shape_B[0],
        reverse(take<1, RankT>(shape_B)));
    }
    else if constexpr (ConvOp == conv::Operator::kWgrad) {
      return make_shape(
        shape_B[RankT - 1],
        reverse(take<0, RankT - 1>(shape_B)));
    }
    else if constexpr (ConvOp == conv::Operator::kDgrad) {
      // shape_B: [K,T,R,S,C], return: [(C),(K,S,R,T)]
      return make_shape(
        shape_B[RankT - 1],
        cute::insert<0>(
          reverse(take<1, RankT - 1>(shape_B)),
          shape_B[0]));
    }
  }

  // Get C extents.
  // fprop: C extents array contains [N,Z,P,Q,K]. Turn that into ((Q,P,Z,N), (K))
  // dgrad: C extents array contains [N,D,H,W,C]. Turn that into ((W,H,D,N), (C))
  // wgrad: C extents array contains [K,T,R,S,C]. Turn that into ((K), (C,S,R,T))
  CUTLASS_HOST_DEVICE
  constexpr auto
  get_shape_C() const {
    using cute::make_shape;
    using cute::reverse;
    using cute::take;

    if constexpr (ConvOp == conv::Operator::kFprop ||
                  ConvOp == conv::Operator::kDgrad) {
      return make_shape(
        reverse(take<0, RankT - 1>(shape_C)),
        shape_C[RankT - 1]);
    }
    else if constexpr (ConvOp == conv::Operator::kWgrad) {
      return make_shape(
        shape_C[0],
        reverse(take<1, RankT>(shape_C)));
    }
  }

  // Static method that returns the canonical strides of tensors (layouts are right major and compact)
  CUTLASS_HOST_DEVICE
  static constexpr TensorStride
  packed_stride_right_major(TensorExtent const& extents) {
    TensorStride strides{};
    strides[RankT-1] = 1;
    cute::for_each(cute::make_rseq<RankT-1>{}, [&](auto i) {
      strides[i] = extents[i+1] * strides[i+1];
    });
    return strides;
  }

  // Static method that returns the packed logical size of any TensorExtent
  CUTLASS_HOST_DEVICE
  static constexpr size_t
  size(TensorExtent const& extents) {
    size_t size = 1;
    cute::for_each(cute::make_seq<RankT>{}, [&](auto i) {
      size *= extents[i];
    });
    return size;
  }

  CUTLASS_HOST_DEVICE
  constexpr size_t
  size_A() const {
    return shape_A[0] * stride_A[0];
  }

  CUTLASS_HOST_DEVICE
  constexpr size_t
  size_B() const {
    return shape_B[0] * stride_B[0];
  }

  CUTLASS_HOST_DEVICE
  constexpr size_t
  size_C() const {
    return shape_C[0] * stride_C[0];
  }

  // Equality operator
  CUTLASS_HOST_DEVICE
  bool operator==(ConvProblemShape<ConvOp, NumSpatialDimensions> const& rhs) const {
    using cute::for_each;
    using cute::make_seq;

    bool is_equal = true;

    // Compare all tensor extents
    for_each(make_seq<RankT>{}, [&](auto i) {
      is_equal = is_equal
          && (shape_A[i] == rhs.shape_A[i])
          && (shape_B[i] == rhs.shape_B[i]);
    });

    // Compare all spatial extents
    for_each(make_seq<RankS>{}, [&](auto i) {
      is_equal = is_equal
          && (lower_padding[i] == rhs.lower_padding[i])
          && (upper_padding[i] == rhs.upper_padding[i])
          && (traversal_stride[i] == rhs.traversal_stride[i])
          && (dilation[i] == rhs.dilation[i]);
    });

    return is_equal;
  }

  /// Inequality operator
  CUTLASS_HOST_DEVICE
  bool operator!=(ConvProblemShape<ConvOp, NumSpatialDimensions> const &rhs) const {
    return !(*this == rhs);
  }

private:
  CUTLASS_HOST_DEVICE
  constexpr auto
  calculate_xformed_act(TensorExtent shape_act, TensorExtent shape_flt) {
    TensorExtent shape_xformed_act{};
    // calculate n,z,p,q,k.
    // a helper lambda to compute a single spatial extent of the nzpqk tensor
    auto nzpqk_extent = [](int act_ext, int filter_ext, int pad_total, int dilation, int tstride) {
      return 1 + (act_ext + pad_total - ((filter_ext -1) * dilation + 1)) / tstride;
    };

    shape_xformed_act[0] = shape_act[0]; // Activation N extent
    cute::for_each(cute::make_seq<RankS>{}, [&](auto i) {
      shape_xformed_act[i+1] = nzpqk_extent(
          shape_act[i+1], shape_flt[i+1], upper_padding[i] + lower_padding[i], dilation[i], traversal_stride[i]);
      });
    shape_xformed_act[RankT-1] = shape_flt[0]; // Filter K extent

    TensorStride stride_xformed_act = packed_stride_right_major(shape_xformed_act);

    return cute::make_tuple(shape_xformed_act, stride_xformed_act);
  }
};

template<
  conv::Operator ConvOp,
  int SpatialDim
>
void print(ConvProblemShape<ConvOp, SpatialDim> const& problem) {
  printf("ConvProblemShape with %d spatial dimensions implementing cutlass::conv::Operator::%d\n",
      SpatialDim, int(ConvOp));
  printf("\tTensorA: ");
      cute::print(problem.shape_A); printf(":");
      cute::print(problem.stride_A); printf("\n");
  printf("\tTensorB: ");
      cute::print(problem.shape_B); printf(":");
      cute::print(problem.stride_B); printf("\n");
  printf("\tTensorC: ");
      cute::print(problem.shape_C); printf(":");
      cute::print(problem.stride_C); printf("\n");
  printf("\tLower padding:     "); print(problem.lower_padding);       printf("\n");
  printf("\tUpper padding:     "); print(problem.upper_padding);       printf("\n");
  printf("\tTraversal strides: "); print(problem.traversal_stride);    printf("\n");
  printf("\tDilation:          "); print(problem.dilation);            printf("\n");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv

////////////////////////////////////////////////////////////////////////////////////////////////////
