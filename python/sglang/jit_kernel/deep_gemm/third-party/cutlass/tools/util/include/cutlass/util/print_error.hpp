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

#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <type_traits>

#include <cute/util/type_traits.hpp>
#include <cute/tensor.hpp>

#include <cute/numeric/numeric_types.hpp>
#include <cute/numeric/complex.hpp>

#include <cutlass/layout/layout.h>

// The computed infinity norm does not include
// any NaN column absolute-value sums.
struct matrix_inf_norm_result {
  // Accumulate errors in double, as this is generally
  // the highest precision that the examples use.
  double inf_norm = 0.0;
  bool found_nan = false;
};

// In theory, cute::Tensor<ViewEngine<T*>, T> could be treated as a view type,
// and thus passed by value (as std::span or std::string_view would be).
// However, generic cute::Tensor are more like containers
// and thus are best passed by reference or const reference.
template <typename EngineType, typename LayoutType>
matrix_inf_norm_result
matrix_inf_norm(cute::Tensor<EngineType, LayoutType> const& host_matrix)
{
  using error_type = decltype(std::declval<matrix_inf_norm_result>().inf_norm);
  using element_type = typename EngineType::value_type;

  error_type inf_norm = 0.0;
  bool found_nan = false;

  // Computing the infinity norm requires that we be able
  // to treat the input as a matrix, with rows and columns.
  const int64_t num_rows = cute::size<0>(host_matrix);
  const int64_t num_cols = cute::size<1>(host_matrix);

  auto abs_fn = [] (element_type A_ij) {
    if constexpr (not std::is_unsigned_v<element_type>) {
      using std::abs;
      return abs(A_ij);
    }
    else {
      return A_ij;
    }
  };

  for (int64_t i = 0; i < num_rows; ++i) {
    error_type row_abs_sum = 0.0;
    for(int64_t j = 0; j < num_cols; ++j) {
      row_abs_sum += abs_fn(host_matrix(i, j));
    }
    if (std::isnan(row_abs_sum)) {
      found_nan = true;
    }
    else {
      inf_norm = row_abs_sum > inf_norm ? row_abs_sum : inf_norm;
    }
  }

  return {inf_norm, found_nan};
}

// Infinity norm of (X - Y).
template <typename EngineType, typename LayoutType>
matrix_inf_norm_result
matrix_diff_inf_norm(cute::Tensor<EngineType, LayoutType> const& X,
                     cute::Tensor<EngineType, LayoutType> const& Y)
{
  using error_type = decltype(std::declval<matrix_inf_norm_result>().inf_norm);
  using element_type = typename EngineType::value_type;

  auto abs_fn = [] (element_type A_ij) {
    if constexpr (not std::is_unsigned_v<element_type>) {
      using std::abs;
      return abs(A_ij);
    }
    else {
      return A_ij;
    }
  };

  assert(cute::size<0>(X) == cute::size<0>(Y));
  assert(cute::size<1>(X) == cute::size<1>(Y));

  // Computing the infinity norm requires that we be able
  // to treat the input as a matrix, with rows and columns.
  const int64_t num_rows = cute::size<0>(X);
  const int64_t num_cols = cute::size<1>(X);

  error_type inf_norm = 0.0;
  bool found_nan = false;

  for (int64_t i = 0; i < num_rows; ++i) {
    error_type row_abs_sum = 0.0;
    for (int64_t j = 0; j < num_cols; ++j) {
      row_abs_sum += error_type(abs_fn(element_type(X(i,j)) -
                                       element_type(Y(i,j))));
    }
    if (std::isnan(row_abs_sum)) {
      found_nan = true;
    }
    else {
      inf_norm = row_abs_sum > inf_norm ? row_abs_sum : inf_norm;
    }
  }

  return {inf_norm, found_nan};
}

template <typename EngineType_A, typename LayoutType_A,
          typename EngineType_B, typename LayoutType_B,
          typename EngineType_C, typename LayoutType_C,
          typename EngineType_C_ref, typename LayoutType_C_ref>
auto
print_matrix_multiply_mollified_relative_error(
  char const A_value_type_name[],
  cute::Tensor<EngineType_A, LayoutType_A> const& A,
  char const B_value_type_name[],
  cute::Tensor<EngineType_B, LayoutType_B> const& B,
  char const C_value_type_name[],
  cute::Tensor<EngineType_C, LayoutType_C> const& C,
  cute::Tensor<EngineType_C_ref, LayoutType_C_ref> const& C_ref)
{
  const auto [A_norm, A_has_nan] = matrix_inf_norm(A);
  const auto [B_norm, B_has_nan] = matrix_inf_norm(B);
  const auto [C_norm, C_has_nan] = matrix_inf_norm(C_ref);
  const auto [diff_norm, diff_has_nan] = matrix_diff_inf_norm(C, C_ref);

  const auto A_norm_times_B_norm = A_norm * B_norm;
  const auto relative_error = A_norm_times_B_norm == 0.0 ?
    diff_norm : (diff_norm / A_norm_times_B_norm);

  // For expected error bounds, please refer to the LAPACK Users' Guide,
  // in particular https://netlib.org/lapack/lug/node108.html .
  // Printing the infinity norm of C is a way to check
  // that both the function being tested (C)
  // and the reference implementation (C_ref)
  // don't just do nothing (or fill with zeros).
  using std::cout;
  using cute::shape;
  cout << "Matrix A: " << shape<0>(A) << "x" << shape<1>(A) << " of " << A_value_type_name << '\n'
      << "Matrix B: " << shape<0>(B) << "x" << shape<1>(B) << " of " << B_value_type_name << '\n'
      << "Matrix C: " << shape<0>(C) << "x" << shape<1>(C) << " of " << C_value_type_name << '\n'
      << std::scientific
      << "Infinity norm of A: " << A_norm << '\n'
      << "Infinity norm of B: " << B_norm << '\n'
      << "Infinity norm of C: " << C_norm << '\n'
      << "Infinity norm of (C - C_ref): " << diff_norm << '\n';

  if(A_norm_times_B_norm == 0.0) {
    cout << "Mollified relative error: " << relative_error << '\n';
  } else {
    cout << "Relative error: " << relative_error << '\n';
  }

  if (A_has_nan || B_has_nan || C_has_nan || diff_has_nan) {
    cout << "Did we encounter NaN in A? " << (A_has_nan ? "yes" : "no") << '\n'
        << "Did we encounter NaN in B? " << (B_has_nan ? "yes" : "no") << '\n'
        << "Did we encounter NaN in C? " << (C_has_nan ? "yes" : "no") << '\n'
        << "Did we encounter NaN in (C - C_ref)? " << (diff_has_nan ? "yes" : "no") << '\n';
  }
  return relative_error;
}

template <typename EngineType, typename LayoutType>
auto
print_matrix_multiply_mollified_relative_error(
  const char value_type_name[],
  const cute::Tensor<EngineType, LayoutType>& A,
  const cute::Tensor<EngineType, LayoutType>& B,
  const cute::Tensor<EngineType, LayoutType>& C_computed,
  const cute::Tensor<EngineType, LayoutType>& C_expected)
{
  return print_matrix_multiply_mollified_relative_error(value_type_name, A, value_type_name, B,
                                                 value_type_name, C_computed, C_expected);
}

// Take a CUTLASS HostTensor (or the like) as input,
// and return a const CuTe Tensor.
// This is useful for use with the above error printing functions.
// This implicitly "transposes" if the layout is RowMajor.
// Note that the HostTensor must be captured by nonconst reference
// in order for X.host_ref().data() to compile.
// (CUTLASS is a bit more container-y than CuTe.)
template<class CutlassHostTensorType>
auto host_matrix_to_const_cute_tensor(CutlassHostTensorType& X)
{
  // The tensors were created with post-transposed extents.
  const auto extents = X.extent();
  const auto shape = cute::Shape<int, int>{extents[0], extents[1]};
  // Both RowMajor and ColumnMajor only store one stride.
  const int LDX = X.stride(0);
  const auto strides = [&]() {
      using input_layout_type = typename std::decay_t<decltype(X)>::Layout;
      if constexpr (std::is_same_v<input_layout_type, cutlass::layout::ColumnMajor>) {
        return cute::Stride<int, int>{1, LDX};
      }
      else {
        static_assert(std::is_same_v<input_layout_type, cutlass::layout::RowMajor>);
        return cute::Stride<int, int>{LDX, 1};
      }
    }();
  const auto layout = cute::make_layout(shape, strides);
  auto X_data = X.host_ref().data();
  auto X_data_const = const_cast<std::add_const_t< decltype(X_data)> >(X_data);
  return cute::make_tensor(X_data_const, layout);
};


// Returns EXIT_SUCCESS if the 2-norm relative error is exactly zero, else returns EXIT_FAILURE.
// This makes the return value suitable as the return value of main().
template <typename T1, typename T2>
int
print_relative_error(
    std::size_t n,
    T1 const& data,
    T2 const& reference,
    bool print_verbose = false,
    bool print_error = true,
    double error_margin = 0.00001) {
  using std::abs; using std::sqrt;

  // Use either double or complex<double> for error computation
  using value_type = cute::remove_cvref_t<decltype(reference[0])>;
  using error_type = std::conditional_t<cute::is_complex<value_type>::value,
                                        cute::complex<double>,
                                        double>;

  if (print_verbose) {
    std::cout << "Idx:\t"<< "Val\t" << "RefVal\t" << "RelError" << std::endl;
  }

  double eps = 1e-200;

  double tot_error_sq = 0;
  double tot_norm_sq = 0;
  double tot_ind_rel_err = 0;
  double max_ind_rel_err = 0;
  double max_diff = 0;
  for (std::size_t i = 0; i < n; ++i) {
    error_type val = data[i];
    error_type ref = reference[i];

    double aref = abs(ref);
    double diff = abs(ref - val);
    double rel_error = diff / (aref + eps);

    // Individual relative error
    tot_ind_rel_err += rel_error;

    // Maximum relative error
    max_ind_rel_err  = std::max(max_ind_rel_err, rel_error);

    // Maximum delta in value error
    max_diff = std::max(max_diff, diff);

    // Total relative error
    tot_error_sq += diff * diff;
    tot_norm_sq  += aref * aref;

    if (print_verbose) {
      std::cout << i << ":\t" << val << "\t" << ref << "\t" << rel_error << std::endl;
    }
  }

  double ave_rel_err = tot_ind_rel_err / double(n);
  if (print_error) {
    printf("Average relative error: %.3e\n", ave_rel_err);
  }

  if (print_error) {
    printf("Maximum relative error: %.3e\n", max_ind_rel_err);
  }

  if (print_error) {
    printf("Maximum difference    : %.3e\n", max_diff);
  }

  double tot_rel_err = sqrt(tot_error_sq/(tot_norm_sq+eps));
  if (print_error) {
    printf("Vector relative error:  %.3e\n", tot_rel_err);
  }

  printf("Vector reference  norm: %.3e\n", sqrt(tot_norm_sq));

  return (tot_rel_err <= error_margin) ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Overload for cute::Tensor<>
template <class Engine, class Layout>
int
print_relative_error(
    cute::Tensor<Engine, Layout> data,
    cute::Tensor<Engine, Layout> reference,
    bool print_verbose = false,
    bool print_error = true,
    double error_margin = 0.00001) {
  assert(size(data) == size(reference));
  return print_relative_error(static_cast<std::size_t>(size(data)),
                              data, reference,
                              print_verbose, print_error, error_margin);
}
