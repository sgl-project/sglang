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

#pragma once
#include <cute/tensor.hpp>                    // CuTe tensor implementation
#include <cute/arch/copy_sm90_desc.hpp>

template <class AccType,
          class TensorA, class TensorB,
          class TensorC, class TensorD,
          class Alpha, class Beta>
void
reference_gemm(TensorA const& tensor_A, TensorB const& tensor_B,
               TensorC const& tensor_C, TensorD      & tensor_D,
               Alpha alpha, Beta beta)
{
  using namespace cute;
  for (int m = 0; m < size<0>(tensor_D); ++m) {
    for (int n = 0; n < size<1>(tensor_D); ++n) {
      AccType c = AccType(0.f);
      for (int k = 0; k < size<1>(tensor_A); ++k) {
        c += tensor_A(m,k) * tensor_B(n,k);
      }
      tensor_D(m,n) = alpha * c + beta * tensor_C(m,n);
    }
  }
}

template <class TensorA, class TensorB,
          class TensorC, class TensorD,
          class RefTensorD>
bool
compare_results(TensorA const& tensor_A, TensorB const& tensor_B,
                TensorC const& tensor_C, TensorD const& tensor_D,
                RefTensorD const& ref_tensor_D,
                bool print_diff = false)
{
  using namespace cute;
  auto norm_A = matrix_inf_norm(tensor_A);
  auto norm_B = matrix_inf_norm(tensor_B);
  auto norm_C = matrix_inf_norm(tensor_C);
  auto norm_D = matrix_inf_norm(tensor_D);
  auto norm_ref_D = matrix_inf_norm(ref_tensor_D);
  auto norm_diff = matrix_diff_inf_norm(tensor_D, ref_tensor_D);

  if (print_diff) {
    for (int m = 0; m < size<0>(tensor_D); ++m) {
      for (int n = 0; n < size<1>(tensor_D); ++n) {
        std::cout << m << "," << n << " : " << tensor_D(m,n) << " vs. " << ref_tensor_D(m,n) << std::endl;
      }
    }
  }

  std::cout << "norm (A)       : " << norm_A.inf_norm << std::endl;
  std::cout << "norm (B)       : " << norm_B.inf_norm << std::endl;
  std::cout << "norm (C)       : " << norm_C.inf_norm << std::endl;
  std::cout << "norm (D)       : " << norm_D.inf_norm << std::endl;
  std::cout << "norm (ref_D)   : " << norm_ref_D.inf_norm << std::endl;
  std::cout << "norm (D-ref_D) : " << norm_diff.inf_norm << std::endl;

  return (!norm_A.found_nan) && (!norm_B.found_nan) &&
         (!norm_C.found_nan) && (!norm_D.found_nan) && (!norm_ref_D.found_nan) &&                 // There are no NaNs
         (norm_A.inf_norm > 0.0) && (norm_B.inf_norm > 0.0) &&
         (norm_C.inf_norm > 0.0) && (norm_D.inf_norm > 0.0) && (norm_ref_D.inf_norm > 0.0) &&     // Values in tensors aren't zeros
         (norm_diff.inf_norm <= 0.0);                                                             // Diff (ref_D-D) == 0
}

template <class Tensor>
void
initialize_tensor(Tensor& tensor, cute::tuple<int, int> value_range = {-2, 2})
{
  using DataType = typename Tensor::element_type;
  auto [min, max] = value_range;
  for (int i = 0; i < cute::size(tensor); i++) {
    tensor(i) = DataType(int((max-min)*(rand() / double(RAND_MAX)) + min));
  }
}
