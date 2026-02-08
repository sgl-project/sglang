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
    \brief Contains additional metadata about layout permute functions used in the example.
*/

#include "cutlass/tensor_coord.h"
#include "cutlass/layout/permute.h"

/// Additional permutation metadata to facilitate testing/printing 
template<typename PermuteLayout>
struct PermuteInfo;

/// Specialization for default case (no permute). Other specializations must follow this template.
template<>
struct PermuteInfo<cutlass::layout::NoPermute> {

  /// Whether this is a BMM or GEMM permutation (NoPermute can actually be either)
  static bool constexpr kBatched = false;

  /// Minimal divisor for row extent
  static int  constexpr kRowFactor = 1;

  /// Minimum divisor for column extent
  static int  constexpr kColumnFactor = 1;

  /// Minimum divisor for batch size dimension
  static int  constexpr kBatchFactor = 1;

  /// Tensor layout used in permutation operation
  using Layout = cutlass::layout::PackedVectorLayout;

  static std::string name() {
    return "NoPermute";
  }

  /// User-friendly description of the permute operation
  static std::string desc() {
    return "no permutation";
  }

  /// Infer original higher-rank tensor shape from GEMM/BMM matrix extents.
  /// For direct (output) permutations, must be a simple reshape of extent.
  /// For inverse (input) permutations, must return shape *before* permute operation.
  /// In case of NoPermute, simply use a linear (rank 1) view of the memory
  static Layout::TensorCoord original_shape(cutlass::MatrixCoord extent, int batch_count) {
    return Layout::TensorCoord(extent.row() * extent.column() * batch_count);
  }

  /// Compute the permuted higher-rank tensor shape from the original shape.
  static Layout::TensorCoord permute(Layout::TensorCoord const &s) {
    return s;
  }
};

template<int D1>
struct PermuteInfo<cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>> {

  static bool constexpr kBatched = true;
  static int  constexpr kRowFactor = 1;
  static int  constexpr kColumnFactor = 1;
  static int  constexpr kBatchFactor = D1;

  using Layout = cutlass::layout::TensorNHWC;

  static std::string name() {
    return "Tensor4DPermuteBMM0213<" + std::to_string(D1) + ">";
  }

  static std::string desc() {
    return "batched GEMM permutation [0, 2, 1, 3]";
  }

  static Layout::TensorCoord original_shape(cutlass::MatrixCoord extent, int batch_count) {
    int D0 = batch_count / D1;
    int D2 = extent.row();
    int D3 = extent.column();
    return {D0, D1, D2, D3};
  }

  static Layout::TensorCoord permute(Layout::TensorCoord const &s) {
    return {s[0], s[2], s[1], s[3]};
  }
};

template<int D1>
struct PermuteInfo<cutlass::layout::Tensor4DPermuteBMM0213RowMajorInverse<D1>> 
: public PermuteInfo<cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>> {

  static bool constexpr kBatched = true;
  static int  constexpr kRowFactor = 1;
  static int  constexpr kColumnFactor = D1;
  static int  constexpr kBatchFactor = 1;

  using Base = PermuteInfo<cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>>;
  using Layout = typename Base::Layout;

  static typename Layout::TensorCoord original_shape(cutlass::MatrixCoord extent, int batch_count) {
    int D0 = batch_count;
    int D2 = extent.row();
    int D3 = extent.column() / D1;
    return {D0, D1, D2, D3};
  }
};

template<int D1>
struct PermuteInfo<cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>> {
  
  static bool constexpr kBatched = true;
  static int  constexpr kRowFactor = 1;
  static int  constexpr kColumnFactor = 1;
  static int  constexpr kBatchFactor = D1;

  using Layout = cutlass::layout::TensorNHCW;

  static std::string name() {
    return "Tensor4DPermuteBMM0321<" + std::to_string(D1) + ">";
  }

  static std::string desc() {
    return "batched GEMM permutation [0, 3, 2, 1]";
  }

  static Layout::TensorCoord original_shape(cutlass::MatrixCoord extent, int batch_count) {
    int D0 = batch_count / D1;
    int D2 = extent.row();
    int D3 = extent.column();
    return {D0, D1, D2, D3};
  }

  static Layout::TensorCoord permute(Layout::TensorCoord const &s) {
    return {s[0], s[3], s[2], s[1]};
  }
};

template<int D1>
struct PermuteInfo<cutlass::layout::Tensor4DPermuteBMM0321ColumnMajorInverse<D1>> 
: public PermuteInfo<cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>> {
  
  static bool constexpr kBatched = true;
  static int  constexpr kRowFactor = D1;
  static int  constexpr kColumnFactor = 1;
  static int  constexpr kBatchFactor = 1;

  using Base = PermuteInfo<cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>>;
  using Layout = typename Base::Layout;

  static typename Layout::TensorCoord original_shape(cutlass::MatrixCoord extent, int batch_count) {
    int D0 = batch_count;
    int D2 = extent.row() / D1;
    int D3 = extent.column();
    return {D0, D1, D2, D3};
  }
};

template<int D1, int D2>
struct PermuteInfo<cutlass::layout::Tensor4DPermute0213RowMajor<D1, D2>> {

  static bool constexpr kBatched = false;
  static int  constexpr kRowFactor = D1;
  static int  constexpr kColumnFactor = D2;
  static int  constexpr kBatchFactor = 1;

  using Layout = cutlass::layout::TensorNHWC;

  static std::string name() {
    return "Tensor4DPermute0213<" + std::to_string(D1) + "," + std::to_string(D2) + ">";
  }

  static std::string desc() {
    return "normal GEMM permutation [0, 2, 1, 3]";
  }

  static Layout::TensorCoord original_shape(cutlass::MatrixCoord extent, int batch_count) {
    int D0 = extent.row() / D1;
    int D3 = extent.column() / D2;
    return {D0, D1, D2, D3};
  }

  static Layout::TensorCoord permute(Layout::TensorCoord const &s) {
    return {s[0], s[2], s[1], s[3]};
  }
};

template<int D1, int D2>
struct PermuteInfo<cutlass::layout::Tensor4DPermute0213RowMajorInverse<D1, D2>>
: public PermuteInfo<cutlass::layout::Tensor4DPermute0213RowMajor<D1, D2>> {

  static bool constexpr kBatched = false;
  static int  constexpr kRowFactor = D2;
  static int  constexpr kColumnFactor = D1;
  static int  constexpr kBatchFactor = 1;

  using Base = PermuteInfo<cutlass::layout::Tensor4DPermute0213RowMajor<D1, D2>>;
  using Layout = typename Base::Layout;

  static typename Layout::TensorCoord original_shape(cutlass::MatrixCoord extent, int batch_count) {
    int D0 = extent.row() / D2;
    int D3 = extent.column() / D1;
    return {D0, D1, D2, D3};
  }
};

template<int D1, int D2>
struct PermuteInfo<cutlass::layout::Tensor4DPermute0213ColumnMajor<D1, D2>>
: public PermuteInfo<cutlass::layout::Tensor4DPermute0213RowMajor<D1, D2>> {
  using Layout = cutlass::layout::TensorCWHN;
};

template<int D1, int D2>
struct PermuteInfo<cutlass::layout::Tensor4DPermute0213ColumnMajorInverse<D1, D2>>
: public PermuteInfo<cutlass::layout::Tensor4DPermute0213RowMajorInverse<D1, D2>> {
  using Layout = cutlass::layout::TensorCWHN;
};

template<int T1, int T2, int T3>
struct PermuteInfo<cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>> {

  static bool constexpr kBatched = false;
  static int  constexpr kRowFactor = T1;
  static int  constexpr kColumnFactor = T2 * T3;
  static int  constexpr kBatchFactor = 1;

  using Layout = cutlass::layout::TensorNDHWC;

  static std::string name() {
    return "Tensor5DPermute20314<" + std::to_string(T1) + "," + std::to_string(T2) + "," + std::to_string(T3) + ">";
  }

  static std::string desc() {
    return "normal GEMM permutation [2, 0, 3, 1, 4]";
  }

  static Layout::TensorCoord original_shape(cutlass::MatrixCoord extent, int batch_count)
  {
    int const T0 = extent.row() / T1;
    int const T4 = extent.column() / (T2 * T3);
    return {T0, T1, T2, T3, T4};
  }

  static Layout::TensorCoord permute(Layout::TensorCoord const &s)
  {
    return {s[2], s[0], s[3], s[1], s[4]};
  }
};

template<int T1, int T2, int T3>
struct PermuteInfo<cutlass::layout::Tensor5DPermute20314RowMajorInverse<T1, T2, T3>>
: public PermuteInfo<cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>> {

  static bool constexpr kBatched = false;
  static int  constexpr kRowFactor = T2;
  static int  constexpr kColumnFactor = T1 * T3;
  static int  constexpr kBatchFactor = 1;

  using Base = PermuteInfo<cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>>;
  using Layout = typename Base::Layout;

  static typename Layout::TensorCoord original_shape(cutlass::MatrixCoord extent, int batch_count) {
    int const T0 = extent.row() / T2;
    int const T4 = extent.column() / (T1 * T3);
    return {T0, T1, T2, T3, T4};
  }
};

template<int T1, int T2, int T3>
struct PermuteInfo<cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>> {

  static bool constexpr kBatched = false;
  static int  constexpr kRowFactor = T1;
  static int  constexpr kColumnFactor = T2 * T3;
  static int  constexpr kBatchFactor = 1;

  using Layout = cutlass::layout::TensorCWHDN;

  static std::string name() {
    return "Tensor5DPermute02413<" + std::to_string(T1) + "," + std::to_string(T2) + "," + std::to_string(T3) + ">";
  }

  static std::string desc() {
    return "normal GEMM permutation [0, 2, 4, 1, 3]";
  }

  using Coord = cutlass::Tensor5DCoord;

  static Layout::TensorCoord original_shape(cutlass::MatrixCoord extent, int batch_count)
  {
    int const T0 = extent.row() / T1;
    int const T4 = extent.column() / (T2 * T3);
    return {T0, T1, T2, T3, T4};
  }

  static Layout::TensorCoord permute(Layout::TensorCoord const &s)
  {
    return {s[0], s[2], s[4], s[1], s[3]};
  }
};

template<int T1, int T2, int T3>
struct PermuteInfo<cutlass::layout::Tensor5DPermute02413ColumnMajorInverse<T1, T2, T3>>
: public PermuteInfo<cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>> {

  static bool constexpr kBatched = false;
  static int  constexpr kRowFactor = T2;
  static int  constexpr kColumnFactor = T1 * T3;
  static int  constexpr kBatchFactor = 1;

  using Base = PermuteInfo<cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>>;
  using Layout = typename Base::Layout;

  static typename Layout::TensorCoord original_shape(cutlass::MatrixCoord extent, int batch_count) {
    int const T0 = extent.row() / T2;
    int const T4 = extent.column() / (T1 * T3);
    return {T0, T1, T2, T3, T4};
  }
};
