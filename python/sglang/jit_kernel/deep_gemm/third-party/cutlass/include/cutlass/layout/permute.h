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
    \brief Defines layout functions used by GEMM+permute path for common tensor or matrix formats.

    Like Layout functions, permute layout functions map logical coordinates to linear memory. They often require additional
    data to describe strides between elements.

    Permute layout functions must implement all members in the interface of NoPermute<> defined in this file. Address offset
    computation lies in operator() with private member variables  {col_permute_, row_permute_ and stride_} as new addresses after permute op.
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

// template<PermuteTag, typename Layout, bool Inverse>
// struct PermuteSelect {
//   // Try to give a reasonable error message to the user
//   static_assert(!platform::is_same<Permute, Permute>::value, // aka always_false<T>
//                 "You've tried to use a layout permutation for which the implementation is not availble. "
//                 "In order to provide an implementation for a particular combination of matrix layout "
//                 "and direction (direct/inverse), please specialize PermuteSelect trait.");
// };

// Base template for defining specializations of permutation inverses
template<typename Permute>
struct InversePermute
{
  // Try to give a reasonable error message to the user
  static_assert(!platform::is_same<Permute, Permute>::value, // aka always_false<T>
                "To apply permutation to a GEMM input operand (A or B), an inverse permutation for the desired "
                "permute class must be defined and enabled by specializing cutlass::layout::InversePermute trait.");
};

class PermuteBase {
public:
  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;
};

class NoPermute : public PermuteBase {
public:
  //
  // Methods
  //

  /// Constructor from matrix extent
  CUTLASS_HOST_DEVICE
  NoPermute(MatrixCoord extent, Index stride) { };

  /// Constructor from pitch-linear extent
  CUTLASS_HOST_DEVICE
  NoPermute(PitchLinearCoord extent, Index stride) { };

  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord coord) const { return 0; } // not correct but should never be called

  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const { return 0; } // not correct but should never be called
};

template<>
struct InversePermute<NoPermute> {
  using type = NoPermute;
};

/// Helper trait to detect if permute operation is a noop
template<typename Permute>
inline bool constexpr is_trivial_permute = platform::is_same<Permute, cutlass::layout::NoPermute>::value;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines permute layouts of various tensor formats.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//  Tensor4DPermute0213
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Permute layout function for 4-D permuted tensors with matrix (dimensions [M, N]) reshaped
/// as [M/D1, D1, D2, N/D2]. Then perform permute([0, 2, 1, 3]) on the corresponding tensor.
template <int D1, int D2>
class Tensor4DPermute0213RowMajor : public PermuteBase {
private:
  //
  // Data members
  //

  Index D3_;

  Index stride_;
  
public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermute0213RowMajor(MatrixCoord extent, Index stride) {

    assert(extent.row() % D1 == 0);
    assert(extent.column() % D2 == 0);

    D3_ = extent.column() / D2;

    stride_ = stride * D1 / D2;
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermute0213RowMajor(PitchLinearCoord extent, Index stride)
  : Tensor4DPermute0213RowMajor(MatrixCoord(extent.strided(), extent.contiguous()), stride) {}
  
  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord coord) const {

    // [i,j,k,l] -> [i,k,j,l]
    Index l = coord.column() % D3_;
    Index k = coord.column() / D3_;
    Index j = coord.row() % D1;
    Index i = coord.row() / D1;

    MatrixCoord permuted{k + i * D2, l + j * D3_};

    return LongIndex(permuted.row()) * LongIndex(stride_) + LongIndex(permuted.column());
  }

  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const { 
    return operator()(MatrixCoord(coord.strided(), coord.contiguous()));
  }
};

// Inverse for Tensor4DPermute0213 can be implemented by simply swapping D1 and D2
template <int D1, int D2>
class Tensor4DPermute0213RowMajorInverse : public Tensor4DPermute0213RowMajor<D2, D1> {
public:
  using Base = Tensor4DPermute0213RowMajor<D2, D1>;
  using Base::Base;
};

template<int D1, int D2>
struct InversePermute<Tensor4DPermute0213RowMajor<D1, D2>> {
  using type = Tensor4DPermute0213RowMajorInverse<D1, D2>;
};

template<int D1, int D2>
struct InversePermute<Tensor4DPermute0213RowMajorInverse<D1, D2>> {
  using type = Tensor4DPermute0213RowMajor<D1, D2>;
};

/// Permute layout function for 4-D permuted tensors with matrix (dimensions [M, N]) reshaped
/// as [M/D1, D1, D2, N/D2]. Then perform permute([0, 2, 1, 3]) on the corresponding tensor.
template <int D1, int D2>
class Tensor4DPermute0213ColumnMajor : public PermuteBase {
private:
  //
  // Data members
  //

  Index D0_;

  Index stride_;
  
public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermute0213ColumnMajor(MatrixCoord extent, Index stride) {

    assert(extent.row() % D1 == 0);
    assert(extent.column() % D2 == 0);

    D0_ = extent.row() / D1;

    stride_ = stride * D2 / D1;
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermute0213ColumnMajor(PitchLinearCoord extent, Index stride)
  : Tensor4DPermute0213ColumnMajor(MatrixCoord(extent.contiguous(), extent.strided()), stride) {}
  
  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord coord) const {

    // [i,j,k,l] -> [i,k,j,l]
    Index l = coord.column() / D2;
    Index k = coord.column() % D2;
    Index j = coord.row() / D0_;
    Index i = coord.row() % D0_;

    MatrixCoord permuted{i + k * D0_, j + l * D1};

    return LongIndex(permuted.row()) + LongIndex(permuted.column()) * LongIndex(stride_);
  }

  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const { 
    return operator()(MatrixCoord(coord.contiguous(), coord.strided()));
  }
};

// Inverse for Tensor4DPermute0213 can be implemented by simply swapping D1 and D2
template <int D1, int D2>
class Tensor4DPermute0213ColumnMajorInverse : public Tensor4DPermute0213ColumnMajor<D2, D1> {
public:
  using Base = Tensor4DPermute0213ColumnMajor<D2, D1>;
  using Base::Base;
};

template<int D1, int D2>
struct InversePermute<Tensor4DPermute0213ColumnMajor<D1, D2>> {
  using type = Tensor4DPermute0213ColumnMajorInverse<D1, D2>;
};

template<int D1, int D2>
struct InversePermute<Tensor4DPermute0213ColumnMajorInverse<D1, D2>> {
  using type = Tensor4DPermute0213ColumnMajor<D1, D2>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//  Tensor4DPermuteBMM0213
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Permute layout function for 4-D permuted tensors for BMM with BMM tensor (dimensions [B, M, N]) reshaped
/// as [B/D1, D1, M, N]. Then perform permute([0, 2, 1, 3]) on the corresponding whole BMM tensor.
template <int D1>
class Tensor4DPermuteBMM0213RowMajor : public PermuteBase {
private:
  //
  // Data members
  //

  Index D3_;

  Index stride_;

  Index batch_stride_;
  
public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermuteBMM0213RowMajor(MatrixCoord extent, Index stride) {

    Index D2 = extent.row();
    D3_ = extent.column();

    stride_ = stride * D1;
    batch_stride_ = D2 * stride_;
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermuteBMM0213RowMajor(PitchLinearCoord extent, Index stride)
  : Tensor4DPermuteBMM0213RowMajor(MatrixCoord(extent.strided(), extent.contiguous()), stride) {}
  
  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord coord) const {

    // The batch index for BMM
    Index BMM_batch_idx = blockIdx.z;
    
    // [i,j,k,l] -> [i,k,j,l]
    Index l = coord.column();
    Index k = coord.row();
    Index j = BMM_batch_idx % D1;
    Index i = BMM_batch_idx / D1;

    Index pbatch = i;
    MatrixCoord pcoord{k, l + j * D3_};

    return pbatch * LongIndex(batch_stride_) + pcoord.row() * LongIndex(stride_) + pcoord.column();
  }

  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const { 
    return operator()(MatrixCoord(coord.strided(), coord.contiguous()));
  }
};

template <int D1>
class Tensor4DPermuteBMM0213RowMajorInverse : public PermuteBase {
private:
  //
  // Data members
  //

  Index D3_;

  Index stride_;

  Index batch_stride_;
  
public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermuteBMM0213RowMajorInverse(MatrixCoord extent, Index stride) {

    assert(extent.column() % D1 == 0);

    Index D2 = extent.row();
    D3_ = extent.column() / D1;

    stride_ = stride / D1;

    batch_stride_ = D2 * stride_;
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermuteBMM0213RowMajorInverse(PitchLinearCoord extent, Index stride)
  : Tensor4DPermuteBMM0213RowMajorInverse(MatrixCoord(extent.strided(), extent.contiguous()), stride) {}
  
  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord coord) const {

    // The batch index for BMM
    Index BMM_batch_idx = blockIdx.z;
    
    // The following assumes grouping [(D0)->batch, (D2)->row, (D1,D3)->col]
    Index l = coord.column() % D3_;
    Index j = coord.column() / D3_;
    Index k = coord.row();
    Index i = BMM_batch_idx;

    // compute original [batch, row, col] index
    Index pbatch = j + i * D1;
    MatrixCoord pcoord{k, l};

    return pbatch * LongIndex(batch_stride_) + pcoord.row() * LongIndex(stride_) + pcoord.column();
  }

  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const { 
    return operator()(MatrixCoord(coord.strided(), coord.contiguous()));
  }
};

template<int D1>
struct InversePermute<Tensor4DPermuteBMM0213RowMajor<D1>> {
  using type = Tensor4DPermuteBMM0213RowMajorInverse<D1>;
};

template<int D1>
struct InversePermute<Tensor4DPermuteBMM0213RowMajorInverse<D1>> {
  using type = Tensor4DPermuteBMM0213RowMajor<D1>;
};

/// Permute layout function for 4-D permuted tensors for BMM with BMM tensor (dimensions [B, M, N]) reshaped
/// as [B/D1, D1, M, N]. Then perform permute([0, 3, 2, 1]) on the corresponding whole BMM tensor.
template <int D1>
class Tensor4DPermuteBMM0321ColumnMajor : public PermuteBase {
private:
  //
  // Data members
  //

  Index D2_;

  Index stride_;

  Index batch_stride_;
  
public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermuteBMM0321ColumnMajor(MatrixCoord extent, Index stride) {

    D2_ = extent.row();
    Index D3 = extent.column();

    stride_ = stride * D1;
    batch_stride_ = stride_ * D3;
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermuteBMM0321ColumnMajor(PitchLinearCoord extent, Index stride)
  : Tensor4DPermuteBMM0321ColumnMajor(MatrixCoord(extent.contiguous(), extent.strided()), stride) {}
  
  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord coord) const {

    Index BMM_batch_idx = blockIdx.z;
    
    // [i,j,k,l] -> [i,k,j,l]
    Index l = coord.column();
    Index k = coord.row();
    Index j = BMM_batch_idx % D1;
    Index i = BMM_batch_idx / D1;

    Index pbatch = i;
    MatrixCoord pcoord{k + j * D2_, l};

    return pbatch * LongIndex(batch_stride_) + pcoord.row() + pcoord.column() * LongIndex(stride_);
  }

  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const { 
    return operator()(MatrixCoord(coord.contiguous(), coord.strided()));
  }
};

template <int D1>
class Tensor4DPermuteBMM0321ColumnMajorInverse : public PermuteBase {
private:
  //
  // Data members
  //

  Index D2_;

  Index stride_;

  Index batch_stride_;
  
public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermuteBMM0321ColumnMajorInverse(MatrixCoord extent, Index stride) {

    assert(extent.row() % D1 == 0);

    D2_ = extent.row() / D1;
    Index D3 = extent.column();

    stride_ = stride / D1;
    batch_stride_ = stride_ * D3;
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermuteBMM0321ColumnMajorInverse(PitchLinearCoord extent, Index stride)
  : Tensor4DPermuteBMM0321ColumnMajorInverse(MatrixCoord(extent.contiguous(), extent.strided()), stride) {}
  
  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord coord) const {

    Index BMM_batch_idx = blockIdx.z;
    
    // The following assumes grouping [(D0)->batch, (D1,D2)->row, (D3)->col]
    Index l = coord.column();
    Index k = coord.row() % D2_;
    Index j = coord.row() / D2_;
    Index i = BMM_batch_idx;

    Index pbatch = i * D1 + j;
    MatrixCoord pcoord{k, l};

    return pbatch * LongIndex(batch_stride_) + pcoord.row() + pcoord.column() * LongIndex(stride_);
  }

  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const { 
    return operator()(MatrixCoord(coord.contiguous(), coord.strided()));
  }
};

template<int D1>
struct InversePermute<Tensor4DPermuteBMM0321ColumnMajor<D1>> {
  using type = Tensor4DPermuteBMM0321ColumnMajorInverse<D1>;
};

template<int D1>
struct InversePermute<Tensor4DPermuteBMM0321ColumnMajorInverse<D1>> {
  using type = Tensor4DPermuteBMM0321ColumnMajor<D1>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//  Tensor5DPermute20314
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Permute layout function for 5-D permuted tensors with output matrix (dimension as [M, N]) reshaped
/// as [M/T1, T1, T2, T3, N/T2/T3]. Then perform permute([2, 0, 3, 1, 4]) on the corresponding output tensor.
template <int T1, int T2, int T3>
class Tensor5DPermute20314RowMajor : public PermuteBase {
private:
  //
  // Data members
  //

  Index T0_;

  Index T4_;

  Index stride_;
  
public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor5DPermute20314RowMajor(MatrixCoord extent, Index stride) {

    assert(extent.row() % T1 == 0);
    assert(extent.column() % (T2 * T3) == 0);

    T0_ = extent.row() / T1;
    T4_ = extent.column() / (T2 * T3);

    /// Update stride_permute with stride
    stride_ = stride / T2 * T1; // stride in Elements
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor5DPermute20314RowMajor(PitchLinearCoord extent, Index stride)
  : Tensor5DPermute20314RowMajor(MatrixCoord(extent.strided(), extent.contiguous()), stride) {}
  
  
  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord coord) const {

    // Permute as torch.permute(X1, [2, 0, 3, 1, 4]) -> 5D Tensor indices as [i,j,k,l,m], the dimension of X 
    // is [T0, T1, T2, T3, T4], after permutation the dim of X1 is [T2, T0, T3, T1, T4].

    Index m = coord.column() % T4_;
    Index l = (coord.column() / T4_) % T3;
    Index k = (coord.column() / T4_) / T3;
    Index j = coord.row() % T1;
    Index i = coord.row() / T1;

    MatrixCoord permuted{i + k * T0_, m + j * T4_ + l * T1 * T4_};

    return LongIndex(permuted.row()) * LongIndex(stride_) + LongIndex(permuted.column());
  }

  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const { 
    return operator()(MatrixCoord(coord.strided(), coord.contiguous()));
  }
};

/// Inverse for Tensor5DPermute20314 (could also be given a proper name, e.g. Tensor5DPermute13024).
template <int T1, int T2, int T3>
class Tensor5DPermute20314RowMajorInverse : public PermuteBase {
private:
  //
  // Data members
  //

  Index T0_;

  Index T4_;

  // Permuted stride in units of elements
  Index stride_;
  
public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor5DPermute20314RowMajorInverse(MatrixCoord extent, Index stride) {

    assert(extent.row() % T2 == 0);
    assert(extent.column() % (T1 * T3) == 0);

    T0_ = extent.row() / T2;
    T4_ = extent.column() / (T1 * T3);

    stride_ = stride / T1 * T2;
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor5DPermute20314RowMajorInverse(PitchLinearCoord extent, Index stride)
  : Tensor5DPermute20314RowMajorInverse(MatrixCoord(extent.strided(), extent.contiguous()), stride) {}

  /// Computes the offset after the inverse of permute operation in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord coord) const {

    Index m = coord.column() % T4_;
    Index j = (coord.column() / T4_) % T1;
    Index l = (coord.column() / T4_) / T1;
    Index i = coord.row() % T0_;
    Index k = coord.row() / T0_;

    MatrixCoord permuted{j + i * T1, m + l * T4_ + k * T3 * T4_};

    return LongIndex(permuted.row()) * LongIndex(stride_) + LongIndex(permuted.column());
  }

  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const { 
    return operator()(MatrixCoord(coord.strided(), coord.contiguous()));
  }
};

template<int T1, int T2, int T3>
struct InversePermute<Tensor5DPermute20314RowMajor<T1, T2, T3>> {
  using type = Tensor5DPermute20314RowMajorInverse<T1, T2, T3>;
};

template<int T1, int T2, int T3>
struct InversePermute<Tensor5DPermute20314RowMajorInverse<T1, T2, T3>> {
  using type = Tensor5DPermute20314RowMajor<T1, T2, T3>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Tensor5DPermute02413
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Permute layout function for 5-D permuted tensors with matrix (dimensions [M, N]) reshaped
/// as [M/T1, T1, T2, T3, N/T2/T3]. Then perform permute([0, 2, 4, 1, 3]) on the corresponding tensor.
template <int T1, int T2, int T3>
class Tensor5DPermute02413ColumnMajor : public PermuteBase {
private:
  //
  // Data members
  //

  Index T0_;

  Index T4_;

  Index stride_;
  
public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor5DPermute02413ColumnMajor(MatrixCoord extent, Index stride) {

    assert(extent.row() % T1 == 0);
    assert(extent.column() % (T2 * T3) == 0);

    T0_ = extent.row() / T1;
    T4_ = extent.column() / (T2 * T3);

    /// Update stride_permute with stride
    stride_ = stride / T1 * T2; // stride in Elements
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor5DPermute02413ColumnMajor(PitchLinearCoord extent, Index stride)
  : Tensor5DPermute02413ColumnMajor(MatrixCoord(extent.contiguous(), extent.strided()), stride) {}
  
  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord coord) const {

    // Permute as torch.permute(X1, [2, 0, 3, 1, 4]) -> 5D Tensor indices as [i,j,k,l,m], the dimension of X 
    // is [T0, T1, T2, T3, T4], after permutation the dim of X1 is [T0, T2, T4, T1, T3].

    Index m = (coord.column() / T2) / T3;
    Index l = (coord.column() / T2) % T3;
    Index k = coord.column() % T2;
    Index j = coord.row() / T0_;
    Index i = coord.row() % T0_;

    MatrixCoord permuted{i + k * T0_, m + j * T4_ + l * T4_ * T1};

    return LongIndex(permuted.row()) + LongIndex(permuted.column()) * LongIndex(stride_);
  }

  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const { 
    return operator()(MatrixCoord(coord.contiguous(), coord.strided()));
  }
};

/// Inverse for Tensor5DPermute02413ColumnMajor
template <int T1, int T2, int T3>
class Tensor5DPermute02413ColumnMajorInverse : public PermuteBase {
private:
  //
  // Data members
  //

  Index T0_;

  Index T4_;

  // Permuted stride in units of elements
  Index stride_;
  
public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor5DPermute02413ColumnMajorInverse(MatrixCoord extent, Index stride) {

    assert(extent.row() % T2 == 0);
    assert(extent.column() % (T1 * T3) == 0);

    T0_ = extent.row() / T2;
    T4_ = extent.column() / (T1 * T3);

    stride_ = stride / T2 * T1;
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor5DPermute02413ColumnMajorInverse(PitchLinearCoord extent, Index stride)
  : Tensor5DPermute02413ColumnMajorInverse(MatrixCoord(extent.contiguous(), extent.strided()), stride) {}

  /// Computes the offset after the inverse of permute operation in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord coord) const {

    Index m = coord.column() % T4_;
    Index j = (coord.column() / T4_) % T1;
    Index l = (coord.column() / T4_) / T1;
    Index i = coord.row() % T0_;
    Index k = coord.row() / T0_;

    MatrixCoord permuted{i + j * T0_, k + l * T2 + m * T2 * T3};

    return LongIndex(permuted.row()) + LongIndex(permuted.column()) * LongIndex(stride_);
  }

  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const { 
    return operator()(MatrixCoord(coord.contiguous(), coord.strided()));
  }
};

template<int T1, int T2, int T3>
struct InversePermute<Tensor5DPermute02413ColumnMajor<T1, T2, T3>> {
  using type = Tensor5DPermute02413ColumnMajorInverse<T1, T2, T3>;
};

template<int T1, int T2, int T3>
struct InversePermute<Tensor5DPermute02413ColumnMajorInverse<T1, T2, T3>> {
  using type = Tensor5DPermute02413ColumnMajor<T1, T2, T3>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace layout
} // namespace cutlass
