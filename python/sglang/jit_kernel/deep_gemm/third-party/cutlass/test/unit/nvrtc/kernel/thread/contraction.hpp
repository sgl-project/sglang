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

#include "cute/tensor.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"


namespace nvrtc {
namespace thread {

template<
  typename ElementA, typename ElementB, typename ElementC,
  typename TileShape, typename ClusterShape,
  bool kTransA, bool kTransB,
  int RANK_M, int RANK_N, int RANK_K, int RANK_L
>
struct ContractionKernel {

using ElementScalar = float;
using ElementAccum = float;
using EpilogueThread = cutlass::epilogue::thread::LinearCombination<ElementC,
                                                                    1,
                                                                    ElementAccum,
                                                                    ElementScalar>;

static constexpr cute::GMMA::Major majorA = ! kTransA ? cute::GMMA::Major::MN : cute::GMMA::Major::K;
static constexpr cute::GMMA::Major majorB = ! kTransB ? cute::GMMA::Major::K : cute::GMMA::Major::MN;

/// Kernel config
typedef int64_t stride_type;
typedef int32_t extent_type;

static constexpr const stride_type* stride_null = nullptr;
static constexpr const extent_type* extent_null = nullptr;

template <int Rank, bool IsMajor, class Indexable>
static constexpr
auto
make_stride_tuple(Indexable const& t, int n, int64_t init_default = 0) {
  static_assert(Rank > 1);
  if constexpr (IsMajor) {
    return cute::transform(cute::make_seq<Rank>{}, [&](auto i) {
      if constexpr (i == 0) {
        return cute::Int<1>{};
      }
      else {
        return i < n ? t[i] : init_default;
      }
    });
  }
  else {
    return cute::make_int_tuple<Rank>(t, n, init_default);
  }
}

using StrideA = decltype(cute::make_stride(
  make_stride_tuple<RANK_M, majorA == cute::GMMA::Major::MN>(stride_null, 0, 0),
  make_stride_tuple<RANK_K, majorA == cute::GMMA::Major::K>(stride_null, 0, 0),
  cute::make_int_tuple<RANK_L>(stride_null, 0, 0)));

using StrideB = decltype(cute::make_stride(
  make_stride_tuple<RANK_N, majorB == cute::GMMA::Major::MN>(stride_null, 0, 0),
  make_stride_tuple<RANK_K, majorB == cute::GMMA::Major::K>(stride_null, 0, 0),
  cute::make_int_tuple<RANK_L>(stride_null, 0, 0)));

using StrideC = decltype(cute::make_stride(
  cute::make_int_tuple<RANK_M>(stride_null, 0, 0),
  cute::make_int_tuple<RANK_N>(stride_null, 0, 0),
  cute::make_int_tuple<RANK_L>(stride_null, 0, 0)));

using ProblemShape = decltype(cute::make_shape(
  cute::make_int_tuple<RANK_M>(extent_null, 0, 0),
  cute::make_int_tuple<RANK_N>(extent_null, 0, 0),
  cute::make_int_tuple<RANK_K>(extent_null, 0, 0),
  cute::make_int_tuple<RANK_L>(extent_null, 0, 0)));

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
  ElementA, StrideA, 16 / sizeof(ElementA),
  ElementB, StrideB, 16 / sizeof(ElementB),
  ElementAccum,
  TileShape, ClusterShape, cutlass::gemm::collective::StageCountAuto,
  cutlass::gemm::KernelTmaWarpSpecialized
>::CollectiveOp;

using EpilogueOutputOp = cutlass::epilogue::collective::DefaultEpilogue<ElementC, StrideC, StrideC, EpilogueThread, cutlass::gemm::EpilogueDefault>;
using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<EpilogueOutputOp>;
using Kernel = cutlass::gemm::kernel::GemmUniversal<
  ProblemShape,
  CollectiveOp,
  CollectiveEpilogue>;

};

} // namespace nvrtc
} // namespace thread
