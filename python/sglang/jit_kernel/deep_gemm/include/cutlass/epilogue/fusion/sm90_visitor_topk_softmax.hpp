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
  \brief Visitor tree Top-K + Softmax fusion operation for sm90 TMA warp-specialized epilogue
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"

#include "cute/tensor.hpp"
#include "sm90_visitor_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Top-K + Softmax reduction across columns
// Performs a reduction of top-K values across N, and finally performs a softmax on them,
// and sets values not in the top-K to 0.
//
//   Assumptions:
//     1. CTA_N >= N (single tile across N, the mode which is reduced)
//     2. EPI_N >= N (single epilogue tile across N, because we can reduce and revisit one
//        epilogue tile at a time.)
//     3. Top-K value is either 2 or 4.
//

namespace detail {

// Implementations for add to sorted list and merging sorted lists,
// with fast paths for lists of size 2 and 4 (Top-2 and Top-4).
// Generic implementations may result in greater register use and branching,
// and should be avoided.
// Fast paths for Top-2 and Top-4 are written in inline PTX directly.

CUTLASS_DEVICE
Array<float, 2> top_2_reduce_scalar(Array<float, 2> a, float scalar) {
  Array<float, 2> out;
  asm volatile(
      "{\n"
      "  .reg .f32 mx;\n"
      "  .reg .pred p;\n"
      "  max.f32 mx, %3, %4;\n"
      "  setp.gtu.f32 p, %2, %4;\n"
      "  selp.f32 %1, mx, %2, p;\n"
      "  selp.f32 %0, %2, %4, p;\n"
      "}\n" : "=f"(out[0]), "=f"(out[1]) : "f"(a[0]), "f"(a[1]), "f"(scalar));
  return out;
}

CUTLASS_DEVICE
Array<float, 2> top_2_reduce(Array<float, 2> a, Array<float, 2> b) {
  Array<float, 2> out;
  asm volatile(
      "{\n"
      "  .reg .v2 .f32 mx;\n"
      "  .reg .pred p;\n"
      "  max.f32 mx.x, %3, %4;\n"           // max(a1, b0)
      "  max.f32 mx.y, %2, %5;\n"           // max(a0, b1)
      "  setp.gtu.f32 p, %2, %4;\n"         // a0 > b0
      "  selp.f32 %1, mx.x, mx.y, p;\n"     // a0 > b0 ? max(a1, b0) : max(a0, b1)
      "  selp.f32 %0, %2, %4, p;\n"         // a0 > b0 ? a0 : b0
      "}\n" : "=f"(out[0]), "=f"(out[1]) :
      "f"(a[0]), "f"(a[1]), "f"(b[0]), "f"(b[1]));
  return out;
}

CUTLASS_DEVICE
Array<float, 4> top_4_reduce_scalar(Array<float, 4> a, float scalar) {
  Array<float, 4> out;
  asm volatile(
      "{\n"
      "  .reg .f32 mx;\n"                   // max(a3, b)
      "  .reg .pred p0;\n"                  // a0 > b
      "  .reg .pred p1;\n"                  // a1 > b
      "  .reg .pred p2;\n"                  // a2 > b
      "  max.f32 mx, %7, %8;\n"             // max(a3, b)
      "  setp.gtu.f32 p0, %4, %8;\n"        // a0 > b
      "  setp.gtu.f32 p1, %5, %8;\n"        // a1 > b
      "  setp.gtu.f32 p2, %6, %8;\n"        // a2 > b
      "  selp.f32 %3, mx, %6, p2;\n"        // a2 > b ? max(a3, b) : a2
      "  selp.f32 %2, %6, %8, p2;\n"        // a1 = a2 > b ? a2 : b
      "  selp.f32 %2, %2, %5, p1;\n"        // a1 > b ? max(a2, b) : a1 == a1 > b ? a1 : old_a1
      "  selp.f32 %1, %5, %8, p1;\n"        // a0 = a1 > b ? a1 : b
      "  selp.f32 %1, %1, %4, p0;\n"        // a0 > b ? max(a1, b) : a0 == a0 > b ? a0 : old_a0
      "  selp.f32 %0, %4, %8, p0;\n"        // a0 = a0 > b ? a0 : b
      "}\n" :
      "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]) :
      "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]), "f"(scalar));
  return out;
}

CUTLASS_DEVICE
Array<float, 4> top_4_reduce(Array<float, 4> a, Array<float, 4> b) {
  Array<float, 4> out;
  asm volatile(
      "{\n"
      "  .reg .f32 mxa0b1;\n"                          // max(a0, b1)
      "  .reg .f32 mxa1b0;\n"                          // max(a1, b0)

      "  .reg .f32 mxa2b0;\n"                          // max(a2, b0)
      "  .reg .f32 mxa1b1;\n"                          // max(a1, b1)
      "  .reg .f32 mxa0b2;\n"                          // max(a1, b1)

      "  .reg .f32 mxa1b2;\n"                          // max(a1, b2)
      "  .reg .f32 mxa2b1;\n"                          // max(a2, b1)
      "  max.f32 mxa1b2, %5, %10;\n"
      "  max.f32 mxa2b1, %6, %9;\n"

      "  .reg .f32 mxa3b0;\n"                          // max(a1, b2)
      "  .reg .f32 mxa0b3;\n"                          // max(a2, b1)
      "  max.f32 mxa3b0, %7, %8;\n"
      "  max.f32 mxa0b3, %4, %11;\n"

      "  .reg .pred pa0b0;\n"                          // a0 > b0
      "  .reg .pred pa1b0;\n"                          // a1 > b0
      "  .reg .pred pa2b0;\n"                          // a2 > b0
      "  .reg .pred pa0b1;\n"                          // a0 > b1
      "  .reg .pred pa1b1;\n"                          // a1 > b1
      "  .reg .pred pa0b2;\n"                          // a0 > b2
      "  .reg .pred pb2a0;\n"                          // b1 > a0
      "  .reg .pred pb1a0;\n"                          // b1 > a0

      "  setp.gtu.f32 pa0b0, %4, %8;\n"                // a0 > b0
      "  setp.gtu.f32 pa1b0, %5, %8;\n"                // a1 > b0
      "  setp.gtu.f32 pa2b0, %6, %8;\n"                // a2 > b0
      "  setp.gtu.f32 pa0b1, %4, %9;\n"                // a0 > b1
      "  setp.gtu.f32 pa1b1, %5, %9;\n"                // a1 > b1
      "  setp.gtu.f32 pa0b2, %4, %10;\n"               // a0 > b2

      "  not.pred pb2a0, pa0b2;\n"
      "  not.pred pb1a0, pa0b1;\n"

      "  selp.f32 mxa1b0, %5, %8, pa1b0;\n"            // max(a1, b0)
      "  selp.f32 mxa0b1, %4, %9, pa0b1;\n"            // max(a0, b1)

      "  selp.f32 mxa1b1, %5, %9, pa1b1;\n"            // max(a1, b1)
      "  selp.f32 mxa2b0, %6, %8, pa2b0;\n"            // max(a2, b0)
      "  selp.f32 mxa0b2, %4, %10, pa0b2;\n"           // max(a0, b2)

      // a0
      "  selp.f32 %0, %4, %8, pa0b0;\n"                // a0 = a0 > b0 ? a0 : b0

      // a1
      "  selp.f32 %1, mxa1b0, mxa0b1, pa0b0;\n"        // a1 = a0 > b0 ? max(a1, b0) : max(a0, b1)

      // a2
      "  mov.f32 %2, mxa1b1;\n"                        // a2 = max(a1, b1) ** most likely case
      "  selp.f32 %2, mxa2b0, %2, pa1b0;\n"            // a0 > a1 > b0
      "  selp.f32 %2, mxa0b2, %2, pb1a0;\n"            // b0 > b1 > a0

      // a3
      "  mov.f32 %3, mxa1b2;\n"                        // a3 = max(a1, b2) ** one of the most likely cases
      "  selp.f32 %3, mxa2b1, %3, pa1b1;\n"            // a3 = a1 > b1 ? max(a2, b1) ** second most likely case
      "  selp.f32 %3, mxa3b0, %3, pa2b0;\n"            // a0 > a1 > a2 > b0
      "  selp.f32 %3, mxa0b3, %3, pb2a0;\n"            // b0 > b1 > b2 > a0
      "}\n" :
      "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]) :
      "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
      "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]));
  return out;
}

// Assumption: array elements are sorted in descending order
// (a[0] is the largest element in a[].)
template <typename Element, int N>
CUTLASS_DEVICE
void add_element_to_desc_sorted_array(cutlass::Array<Element, N>& a, Element b) {
  if constexpr (N == 2 && is_same_v<Element, float>) {
    a = top_2_reduce_scalar(a, b);
  }
  else if constexpr (N == 4 && is_same_v<Element, float>) {
    a = top_4_reduce_scalar(a, b);
  }
  else {
    // slower generic path with branching, slower, and can cause register spill
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < N; ++k) {
      if (a[k] < b) {
        // Shift down
        CUTLASS_PRAGMA_UNROLL
        for (int l = N - 1; l > k; --l) {
          a[l] = a[l-1];
        }
        a[k] = b;
        break;
      }
    }
  }
}

// Assumption: array elements are sorted in descending order
// (a[0] and b[0] are the largest elements in a[] and b[].)
template <typename Element, int N>
CUTLASS_DEVICE
void merge_desc_sorted_arrays(cutlass::Array<Element, N>& a, const cutlass::Array<Element, N>& b) {
  if constexpr (N == 2 && is_same_v<Element, float>) {
    a = top_2_reduce(a, b);
  }
  else if constexpr (N == 4 && is_same_v<Element, float>) {
    a = top_4_reduce(a, b);
  }
  else {
    // slower generic path with branching, slower, and can cause register spill
    int j = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < N; ++k) {
      if (a[k] < b[j]) {
        // Shift down
        CUTLASS_PRAGMA_UNROLL
        for (int l = N - 1; l > k; --l) {
          a[l] = a[l-1];
        }
        a[k] = b[j];
        ++j;
      }
    }
  }
}

// Assumption: array elements are sorted in descending order
// (a[0] is the largest element in a[].)
template <typename Element, int N>
CUTLASS_DEVICE
Element topk_logsumexp(cutlass::Array<Element, N> a) {
  // Do one less `exp`, because we know what its result will be.
  // Assume x is a set of `x_i`s, and `x_m` is the maximum of that set.
  // logsumexp(x) = log(sum(x_i)) = m + log(sum(x_i - m)) = m + log(1 + sum_{i != m}(x_i - x_m))
  // Compute m + log(1 + sum_{i != m}(x_i - x_m))
  Element sum = Element(1.0);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 1; i < N; ++i) {
    sum += fast_exp(a[i] - a[0]);
  }
  return a[0] + fast_log(sum);
}

CUTLASS_DEVICE
float fast_masked_softmax(float value, float minimum, float logsumexp) {
  float new_value;
  asm volatile(
      "{\n"
      "  .reg .pred p0;\n"
      // value >= minimum
      "  setp.geu.f32 p0, %1, %2;\n"

      "  .reg .f32 x_lse;\n"
      "  .reg .f32 %%f<11>;\n"
      "  .reg .b32 %%r<3>;\n"

      // x_lse = value - minimum
      "  sub.rn.f32  x_lse, %1, %3;\n"

      // exp(x_lse)
      // The following is derived from a ptx dump of expf.
      // exp requires a base conversion from exp2.
      "  fma.rn.f32 %%f1, x_lse, 0f3BBB989D, 0f3F000000;\n"
      "  cvt.sat.f32.f32 %%f2, %%f1;\n"
      "  fma.rm.f32 %%f3, %%f2, 0f437C0000, 0f4B400001;\n"
      "  add.f32 %%f4, %%f3, 0fCB40007F;\n"
      "  neg.f32 %%f5, %%f4;\n"
      "  fma.rn.f32 %%f6, x_lse, 0f3FB8AA3B, %%f5;\n"
      "  fma.rn.f32 %%f7, x_lse, 0f32A57060, %%f6;\n"
      "  mov.b32 %%r1, %%f3;\n"
      "  shl.b32 %%r2, %%r1, 23;\n"
      "  mov.b32 %%f8, %%r2;\n"
      "  ex2.approx.ftz.f32 %%f9, %%f7;\n"
      "  mul.f32 %%f10, %%f9, %%f8;\n"

      // Mask or softmax
      "  selp.f32 %0, %%f10, 0f00000000, p0;\n"
      "}\n" : "=f"(new_value) : "f"(value), "f"(minimum), "f"(logsumexp));
  return new_value;
}

template <typename Element>
CUTLASS_DEVICE
Element masked_softmax(Element value, Element minimum, Element logsumexp) {
  if constexpr (is_same_v<Element, float>) {
    // Inline PTX implementation
    // Significantly reduces register requirements
    return fast_masked_softmax(value, minimum, logsumexp);
  }
  else {
    return value < minimum ? Element(0.0) : fast_exp(value - logsumexp);
  }
}

} // namespace detail

template <
  int TopK,
  int FragmentSize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  int Alignment = 128 / sizeof_bits_v<ElementOutput>,
  bool UseButterflyReduce = true
>
struct Sm90TopKSoftmaxColReduction {
private:
  static_assert(is_same_v<ElementCompute, float>, "Fused Top-K + Softmax reduction requires FP32 accumulation.");
  static_assert(TopK == 2 || TopK == 4,
  "Fused Top-K + Softmax reduction only allows K=2 and K=4, because those cases have been performance-optimized. Other values of K can be enabled by removing this assertion, but they may come with serious performance implications."
  );
  static_assert(Alignment * sizeof_bits_v<ElementOutput> % 128 == 0, "sub-16B alignment not supported yet");

  // Reduction tensors
  //   We have two tensors for this EVT node: a reduction tensor and a tensor holding
  //   final reduction values (tCrSoftmax). The reason for this is that Top-K and Softmax
  //   require different reductions, but those luckily overlap. Top-K obviously needs at least
  //   two values (K >= 2), and softmax needs one value: logsumexp. Logsumexp is simply the log
  //   of sum of exponents over the set, and is equivalent to m + sum(exp(x_i - m)), where m is the
  //   maximum of all x_i elements. Since safe softmax for any element x_i is computed as
  //   softmax(x_i) = exp(x_i - m) / sum_j(exp(x_j - max))
  //   we can track logsumexp instead of tracking two variables (sum of exps and the max).
  //   In addition, subtracting logsumexp from any element and taking its exp is equivalent to
  //   computing its softmax.
  //
  //   The overlap between softmax and top-K is that we don't need to reduce logsumexp along the
  //   way at all, because any element not in the top-K is going to be masked out and set to 0.
  //   Therefore, we only reduce the top-K elements, and when done, compute their logsumexp and
  //   keep it, and the smallest element in the top-K for masking out non-top-K elements.
  //
  //   This means that our final reduction result will always be 2 elements, regardless of the value
  //   of K: minimum of top-K, and logsumexp.
  //
  //   For each reduction tensor, we define a new struct for readability.

  struct ReductionResult {
    ElementCompute min_;
    ElementCompute logsumexp_;

    CUTLASS_DEVICE
    ReductionResult() { }

    CUTLASS_DEVICE
    ReductionResult(ElementCompute min, ElementCompute logsumexp):
      logsumexp_(logsumexp), min_(min) { }

    // Warp shuffle broadcast
    CUTLASS_DEVICE
    void shuffle_up_sync(uint32_t delta, int lane_id) {
      static_assert(sizeof(ReductionResult) == sizeof(uint64_t));
      uint64_t r = reinterpret_cast<uint64_t&>(*this);
      r = __shfl_up_sync(0xFFFFFFFF, r, delta);
      *this = (lane_id - static_cast<int>(delta) >= 0) ? reinterpret_cast<ReductionResult&>(r) : *this;
    }
  };

  struct TopKResult {
    Array<ElementCompute, TopK> top_k_;

    CUTLASS_DEVICE
    TopKResult() {
      top_k_.fill(-cutlass::platform::numeric_limits<ElementCompute>::infinity());
    }

    // This is where we do the "final" reduction, where we compute
    // the logsumexp for softmax, keep the smallest value in top-K,
    // and discard the rest.
    CUTLASS_DEVICE
    ReductionResult reduce_final() const {
      return ReductionResult(top_k_[TopK - 1], topk_logsumexp(top_k_));
    }

    // Butterfly reduction
    CUTLASS_DEVICE
    void shuffle_xor_sync(int laneMask) {
      if constexpr (TopK == 2) {
        static_assert(sizeof(TopKResult) == sizeof(uint64_t));
        uint64_t top_k = reinterpret_cast<uint64_t&>(*this);
        top_k = __shfl_xor_sync(0xFFFFFFFF, top_k, laneMask);
        auto synced_v = reinterpret_cast<TopKResult&>(top_k);
        detail::merge_desc_sorted_arrays(top_k_, synced_v.top_k_);
      }
      else if constexpr (TopK == 4) {
        static_assert(sizeof(TopKResult) == 2 * sizeof(uint64_t));
        uint64_t* top_k_ptr = reinterpret_cast<uint64_t*>(this);
        uint64_t top_k_arr[2];
        top_k_arr[0] = top_k_ptr[0];
        top_k_arr[1] = top_k_ptr[1];
        top_k_arr[0] = __shfl_xor_sync(0xFFFFFFFF, top_k_arr[0], laneMask);
        top_k_arr[1] = __shfl_xor_sync(0xFFFFFFFF, top_k_arr[1], laneMask);
        auto synced_v = reinterpret_cast<TopKResult&>(top_k_arr);
        detail::merge_desc_sorted_arrays(top_k_, synced_v.top_k_);
      }
      else {
        TopKResult synced_v;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < TopK; ++i) {
          synced_v.top_k_[i] = __shfl_xor_sync(0xFFFFFFFF, top_k_[i], laneMask);
        }
        detail::merge_desc_sorted_arrays(top_k_, synced_v.top_k_);
      }
    }

    // Warp shuffle reduction
    CUTLASS_DEVICE
    void shuffle_down_sync(uint32_t delta) {
      if constexpr (TopK == 2) {
        static_assert(sizeof(TopKResult) == sizeof(uint64_t));
        uint64_t top_k = reinterpret_cast<uint64_t&>(*this);
        top_k = __shfl_down_sync(0xFFFFFFFF, top_k, delta);
        auto synced_v = reinterpret_cast<TopKResult&>(top_k);
        detail::merge_desc_sorted_arrays(top_k_, synced_v.top_k_);
      }
      else if constexpr (TopK == 4) {
        static_assert(sizeof(TopKResult) == 2 * sizeof(uint64_t));
        uint64_t* top_k_ptr = reinterpret_cast<uint64_t*>(this);
        uint64_t top_k_arr[2];
        top_k_arr[0] = top_k_ptr[0];
        top_k_arr[1] = top_k_ptr[1];
        top_k_arr[0] = __shfl_down_sync(0xFFFFFFFF, top_k_arr[0], delta);
        top_k_arr[1] = __shfl_down_sync(0xFFFFFFFF, top_k_arr[1], delta);
        auto synced_v = reinterpret_cast<TopKResult&>(top_k_arr);
        detail::merge_desc_sorted_arrays(top_k_, synced_v.top_k_);
      }
      else {
        TopKResult synced_v;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < TopK; ++i) {
          synced_v.top_k_[i] = __shfl_down_sync(0xFFFFFFFF, top_k_[i], delta);
        }
        detail::merge_desc_sorted_arrays(top_k_, synced_v.top_k_);
      }
    }
  };

public:
  struct SharedStorage { };

  struct Arguments { };

  struct Params { };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return {};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    auto [M, N, K, L] = problem_shape;
    auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};
    // Cross CTA reduction is not possible because there is no guarantee that all CTAs run
    // concurrently.
    // Cross epilogue tile reduction is possible, but re-visiting and applying reduction
    // to accumulators is only possible for the current epilogue tile.
    auto [epi_M, epi_N] = EpilogueTile{};
    return N <= tile_N && N <= epi_N && N >= TopK;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  Sm90TopKSoftmaxColReduction() { }

  CUTLASS_HOST_DEVICE
  Sm90TopKSoftmaxColReduction(Params const& params, SharedStorage const& shared_storage)
      : params(params) { }

  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class ArgsTuple>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(ArgsTuple&& args_tuple, Params const& params)
      : args_tuple(cute::forward<ArgsTuple>(args_tuple)),
        params(params) {}

    ArgsTuple args_tuple;
    Params const& params;

    template <typename ElementAccumulator, typename ElementInput>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {

      auto& [tCrTopK, tCrSoftmax, tCcCol, cCol,
              lane_layout_MN, lane_mn,
              residue_cCol, residue_tCcCol] = args_tuple;
      Tensor tCcCol_mn = tCcCol(_,_,_,epi_m,epi_n);

      using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};

      Array frg_I = convert_input(frg_input);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        auto thread_crd = tCcCol_mn(epi_v * FragmentSize + i);
        if (elem_less(thread_crd, residue_tCcCol)) {
          TopKResult& tCrCol_vmn = tCrTopK(epi_v * FragmentSize + i);
          detail::add_element_to_desc_sorted_array(tCrCol_vmn.top_k_, frg_I[i]);
        }
      }

      return frg_input;
    }

    template <class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void
    reduce(STensor&& smem_buffer, SyncFn const& sync_fn, int epi_m, int epi_n, bool is_last_iteration, VTensor visit_results) {

      auto& [tCrTopK, tCrSoftmax, tCcCol, cCol,
              lane_layout_MN, lane_mn,
              residue_cCol, residue_tCcCol] = args_tuple;

      // fully OOB CTA in partially OOB cluster
      if (not elem_less(cCol(_0{},_0{}), residue_cCol)) {
        return;
      }
      Tensor tCcCol_mn = tCcCol(_,_,_,epi_m,epi_n);

      // `tCrTopK` and `tCrSoftmax` have 0-strides along modes that correspond to N,
      // in order to reduce along modes in the `R2S` sublayout that correspond to N.
      // This means we should modify and warp-reduce them according to their co-domain instead of
      // their domain. Therefore we keep a filtered view of both and use them as necessary.
      auto tCrTopK_f = filter(tCrTopK);
      auto tCrSoftmax_f = filter(tCrSoftmax);

      // The pattern here is: reduce Top-K first, then compute logsumexp, keep it and the
      // last element of Top-K, use the latter to mask the visited results, and the former
      // to apply softmax.
      //
      // This gives us two options: reduce the Top-K with warp shuffles, have the reduced
      // lanes compute logsumexp and pair it with the last Top-K element, and broadcast
      // the result back using warp shuffles.
      //
      // Alternatively, we can do a butterfly reduction over Top-K, and have all lanes
      // compute their own logsumexp and skip the broadcast.
      if constexpr (UseButterflyReduce) {
        //
        // 1. Butterfly reduction
        //
        CUTLASS_PRAGMA_UNROLL
        for (int j = 1; j < size<1>(lane_layout_MN); j *= 2) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tCrTopK_f); ++i) {
            tCrTopK_f(i).shuffle_xor_sync(j);
          }
        }

        //
        // 2. Strip down reduced value and compute sum of exps
        //
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrSoftmax_f); ++i) {
          tCrSoftmax_f(i) = tCrTopK_f(i).reduce_final();
        }
      }
      else {
        //
        // 1. Warp shuffle reduction
        //
        CUTLASS_PRAGMA_UNROLL
        for (int reduction_cols = size<1>(lane_layout_MN) / 2; reduction_cols > 0; reduction_cols /= 2) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tCrTopK_f); ++i) {
            tCrTopK_f(i).shuffle_down_sync(lane_layout_MN(_0{},reduction_cols));
          }
        }

        //
        // 2. Strip down reduced value and compute sum of exps
        //
        bool is_reduced_lane = get<1>(lane_mn) == 0;
        if (is_reduced_lane) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tCrSoftmax_f); ++i) {
            tCrSoftmax_f(i) = tCrTopK_f(i).reduce_final();
          }
        }

        //
        // 3. Broadcast reduced values to all participants
        //
        CUTLASS_PRAGMA_UNROLL
        for (int broadcast_cols = 1; broadcast_cols <= size<1>(lane_layout_MN) / 2; broadcast_cols *= 2) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tCrSoftmax_f); ++i) {
            tCrSoftmax_f(i).shuffle_up_sync(lane_layout_MN(_0{},broadcast_cols), get<1>(lane_mn));
          }
        }
      }

      //
      // 4. Re-visit and apply top-K and softmax
      //
      CUTLASS_PRAGMA_UNROLL
      for (int epi_v = 0; epi_v < size(visit_results); ++epi_v) {
        auto& visit_frag = visit_results(epi_v);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          visit_frag[i] = detail::masked_softmax(
            visit_frag[i],
            tCrSoftmax(epi_v * FragmentSize + i).min_,
            tCrSoftmax(epi_v * FragmentSize + i).logsumexp_
          );
        }
      }

    }

    CUTLASS_DEVICE void
    end_loop(int epi_m, int epi_n) {
      auto& [tCrTopK, tCrSoftmax, tCcCol, cCol,
              lane_layout_MN, lane_mn,
              residue_cCol, residue_tCcCol] = args_tuple;

      // Reset reduced top-K values for next tile
      // This must be done because we only assume a single epilogue tile across N,
      // but not M.
      fill(tCrTopK, TopKResult());
    }

    CUTLASS_DEVICE void
    end() { }

  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    Layout ref_layout_MN = [&] () {
      auto mn_shape = shape(typename decltype(args.tiled_copy)::Tiler_MN{});
      if constexpr (ReferenceSrc) { return right_inverse(args.tiled_copy.get_layoutS_TV()).with_shape(mn_shape); }
      else                        { return right_inverse(args.tiled_copy.get_layoutD_TV()).with_shape(mn_shape); }
    }();                                                                                         // tile_mn -> tv_idx

    // Get the MN layout + coord of lanes to determine shuffle reduction iterations
    using _W = Int<decltype(args.tiled_copy)::TiledNumThr::value / NumThreadsPerWarp>;
    Layout tv2lane = Layout<Shape<Int<NumThreadsPerWarp>,_W,_1>,Stride<_1,_0,_0>>{};            //   tv_idx -> lane_idx
    Layout ref2lane = composition(tv2lane, ref_layout_MN);                                      //  tile_mn -> lane_idx
    Layout lane_layout_MN = make_layout(filter(get<0>(ref2lane)), filter(get<1>(ref2lane)));    //  lane_mn -> lane_idx
    Layout inv_lane_layout_MN = right_inverse(lane_layout_MN);                                  // lane_idx -> lane_mn
    int lane_idx = canonical_lane_idx();
    auto lane_mn = idx2crd(inv_lane_layout_MN(lane_idx), shape(lane_layout_MN));

    // Get the MN layout + coord of warps to determine smem reduction iterations
    Layout tv2warp = Layout<Shape<Int<NumThreadsPerWarp>,_W,_1>,Stride<_0,_1,_0>>{};            //   tv_idx -> warp_idx
    Layout ref2warp = composition(tv2warp, ref_layout_MN);                                      //  tile_mn -> warp_idx
    Layout warp_layout_MN = make_layout(filter(get<0>(ref2warp)), filter(get<1>(ref2warp)));    //  warp_mn -> warp_idx

    // Make sure there's only one warp across N so we can use warp shuffle intrinsics for reduction.
    static_assert(decltype(size<1>(warp_layout_MN))::value <= 1);

    // Reduction layout
    //   We're assuming all elements in a row (over which we're performing the reduction) are
    //   visited in the same corresponding epilogue tile, and this is what allows us to apply the
    //   top-K + softmax operation within `reduce()`, by re-visiting the accumulated results.
    //
    //   This presents a challenge, because the layout of the accumulated results is typically in
    //   in the register to shared memory shape, or: (R2S,R2S_M,R2S_N).
    //   This means that we still need to reduce this tensor along N.
    //
    //   The solution is simple: we need to flatten the layout, identify modes that correspond to
    //   N and set their strides to 0, in order to map fragment indices corresponding to the same
    //   row back to the same element in the tensor.
    //
    //   This requires some extra layout manipulation, which is as follows.

    // Create new accumulator layout with column broadcast
    auto [M, N, K] = args.tile_shape_mnk;
    auto thr_mma = args.tiled_mma.get_thread_slice(args.thread_idx);
    auto gColReduce = make_tensor<ElementCompute>(
        make_layout(make_shape(M, N), make_stride(_1{}, 0_c)));                                                // (M,N)
    auto tCrColReduce = make_tensor_like<ElementCompute>(                                       // (FrgV, MMA_M, MMA_N)
        thr_mma.partition_C(gColReduce).layout());

    // Tile the new accumulator tensor according to R2S
    ThrCopy thread_r2s = args.tiled_copy.get_slice(args.thread_idx);
    Tensor tRS_rSoftmax = thread_r2s.retile_S(tCrColReduce);                               // ((R2S,R2S_V),MMA_M,MMA_N)
    auto tCrC_layout = args.tCrC.layout();                                                         // (R2S,R2S_M,R2S_N)

    // Compose the new accumulator R2S layout with the expected tCrC layout to get final
    // reduction tensor layout.
    auto tCrSoftmax_layout = take<0, 3>(tRS_rSoftmax.layout()).compose(tCrC_layout); // (R2S,R2S_V) o (R2S,R2S_M,R2S_N)

    Tensor tCrTopK = make_tensor<TopKResult>(tCrSoftmax_layout);                                   // (R2S,R2S_M,R2S_N)
    Tensor tCrSoftmax = make_tensor<ReductionResult>(tCrSoftmax_layout);                           // (R2S,R2S_M,R2S_N)
    fill(tCrTopK, TopKResult());

    auto args_tuple = make_tuple(
        cute::move(tCrTopK), cute::move(tCrSoftmax), args.tCcD, args.cD,
        lane_layout_MN, lane_mn,
        args.residue_cD, args.residue_tCcD);
    return ConsumerStoreCallbacks<decltype(args_tuple)>(std::move(args_tuple), params);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
