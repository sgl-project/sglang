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

#include <cute/config.hpp>           // CUTE_HOST_DEVICE

#include <cute/layout.hpp>
#include <cute/tensor_impl.hpp>

namespace cute
{

////////////////////////////////
// Layout 2D to Console table //
////////////////////////////////

template <class Layout>
CUTE_HOST_DEVICE
void
print_layout(Layout const& layout)  // (m,n) -> idx
{
  CUTE_STATIC_ASSERT_V(rank(layout) == Int<2>{});

  int idx_width = num_digits(cosize(layout)) + 2;
  const char* delim = "+-----------------------";

  print(layout); print("\n");

  // Column indices
  print("    ");
  for (int n = 0; n < size<1>(layout); ++n) { printf("  %*d ", idx_width-2, n); }
  printf("\n");

  // Print out A m-by-n
  for (int m = 0; m < size<0>(layout); ++m) {
    // Header
    print("    ");
    for (int n = 0; n < size<1>(layout); ++n) { printf("%.*s", idx_width+1, delim); }
    printf("+\n");
    // Values
    printf("%2d  ", m);  // Row indices
    for (int n = 0; n < size<1>(layout); ++n) { printf("| %*d ", idx_width-2, int(layout(m,n))); }
    printf("|\n");
  }
  // Footer
  print("    ");
  for (int n = 0; n < size<1>(layout); ++n) { printf("%.*s", idx_width+1, delim); }
  printf("+\n");
}

// Capture and cast smem_ptr_flag Layouts to offset-0 layouts
template <class SwizzleFn, int B, class Layout>
CUTE_HOST_DEVICE
void
print_layout(ComposedLayout<SwizzleFn,smem_ptr_flag_bits<B>,Layout> const& layout)
{
  print_layout(as_position_independent_swizzle_layout(layout));
}

////////////////////////////////
// Tensor 1D,2D,3D,4D Console //
////////////////////////////////

template <class Engine, class Layout>
CUTE_HOST_DEVICE
void
print_tensor(Tensor<Engine,Layout> const& tensor, bool print_type = true)
{
  if (print_type) {
    print(tensor); print(":\n");
  }

  if constexpr (Layout::rank == 1)
  {
    for (int m = 0; m < size(tensor); ++m) {
      pretty_print(tensor(m));
      printf("\n");
    }
  } else
  if constexpr (Layout::rank == 2)
  {
    for (int m = 0; m < size<0>(tensor); ++m) {
      for (int n = 0; n < size<1>(tensor); ++n) {
        pretty_print(tensor(m,n));
      }
      printf("\n");
    }
  } else
  if constexpr (Layout::rank == 3)
  {
    print_tensor(tensor(_,_,0), false);
    for (int k = 1; k < size<2>(tensor); ++k) {
      for (int i = 0; i < 5*size<1>(tensor); ++i) { print("-"); } print("\n");
      print_tensor(tensor(_,_,k), false);
    }
  } else
  if constexpr (Layout::rank == 4)
  {
    print_tensor(tensor(_,_,_,0), false);
    for (int p = 1; p < size<3>(tensor); ++p) {
      for (int i = 0; i < 5*size<1>(tensor); ++i) { print("="); } print("\n");
      print_tensor(tensor(_,_,_,p), false);
    }
  }
}

#if !defined(__CUDACC_RTC__)
template <class Engine, class Layout>
CUTE_HOST
std::ostream&
print_tensor_os(std::ostream& os, Tensor<Engine,Layout> const& tensor)
{
  int digits = 9;

  if constexpr (Layout::rank == 1)
  {
    for (int m = 0; m < size(tensor); ++m) {
      os << std::setw(digits) << tensor(m) << std::endl;
    }
  } else
  if constexpr (Layout::rank == 2)
  {
    for (int m = 0; m < size<0>(tensor); ++m) {
      for (int n = 0; n < size<1>(tensor); ++n) {
        os << std::setw(digits) << tensor(m,n);
      }
      os << std::endl;
    }
  } else
  if constexpr (Layout::rank == 3)
  {
    print_tensor_os(os, tensor(_,_,0));
    for (int k = 1; k < size<2>(tensor); ++k) {
      for (int i = 0; i < digits*size<1>(tensor); ++i) { os << "-"; } os << std::endl;
      print_tensor_os(os, tensor(_,_,k));
    }
  } else
  if constexpr (Layout::rank == 4)
  {
    print_tensor_os(os, tensor(_,_,_,0));
    for (int p = 1; p < size<3>(tensor); ++p) {
      for (int i = 0; i < digits*size<1>(tensor); ++i) { os << "="; } os << std::endl;
      print_tensor_os(os, tensor(_,_,_,p));
    }
  }

  return os;
}

template <class Engine, class Layout>
CUTE_HOST
std::ostream&
operator<<(std::ostream& os, Tensor<Engine,Layout> const& tensor)
{
  os << tensor.layout() << std::endl;
  return print_tensor_os(os, tensor);
}
#endif // !defined(__CUDACC_RTC__)

} // end namespace cute
