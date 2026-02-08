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
  \brief Example of a GETT targeting Hopper tensor cores using the CUTLASS 3.x API.

  CUTLASS has long provided implementations of Generalized Matrix times Matrix (GEMM) kernels.
  However, a plethora of workloads compute on higher ranked tensors. Products of such tensors,
  called tensor contractions, can be executed as multiple batched GEMMs, however, they can be
  further accelerated with kernels that natively operate on these higher ranked tensors to
  perform Generalized Tensor times Tensor contractions (GETT). CuTe's hierarchical layouts
  and CUTLASS 3.0's unified micro-kernels make implementation of GETTs trivial. In this example,
  we show how CUTLASS 3.0, CuTe, and Hopper's TMA feature together can accelerate GETTs while
  making the process of authoring custom GETT kernels easier than ever before.

  The modes of a tensor that participate in a GETT can be fundamentally grouped into four
  semantic categories. The contraction modes (or K-modes) only appear in the A and B (left and right)
  inputs but not in the C output tensor. Row modes (or M-modes) only appear in the left
  input tensor (A) and the output tensor (C). Column modes (or N-modes) only appear in the
  right (B) input tensor and the output tensor (C). Batch modes (or L-modes) appear in all
  input and output tensors. If we fold the many modes of a tensor contraction into these four
  categories, it would allow us to represent the input and output tensors as rank-3 "matrices"
  that can be computed upon as if we were computing a batched GEMM!

  This is exactly what CuTe's hierarchical layout representation allows us to do! Instead of having
  simple integers as strides for these four modes, we can have nested strides for each of these
  semantic categories that themselves have multiple modes within them -- multi-mode strides!
  In CUTLASS 3.0, all one has to do to take advantage of this capability is to substitute the
  required multi-mode strides instead of the default ones provided by gemm::detail::TagToStrideX.

  In the following example, we illustrate how every Hopper GEMM in CUTLASS 3.0 is a GETT in disguise.
  We begin by defining the four modes detailed above as Row, Col (column), Red (reduction), and
  Bat (batch) strides, which we then nest for each of the in/out tensors to create our rank-3 stride
  tuples. Note that although we do not define the problem shape type explicitely, it too remains a
  rank-4 shape tuple just like any other batched GEMM, but instead with multi-mode shapes for each
  of the four corresponding multi-modes within it. After this, the same CollectiveMma and
  CollectiveBuilder we describe in examples 50 and 49 are used to create our kernel type. Nothing
  else changes from a user's point of view. Note that multi-mode strides do not affect our
  specializations in any way -- the lexical spelling of our kernels remains the same. The
  only difference between a CUTLASS 3 batched GEMM and GETT are the instaced CuTe Layouts.

  CollectiveBuilders rely on detecting the static-1 in the stride tuples to determine the major mode,
  which is what the example demonstrates. However, it is possible to have all modes be dynamic as well
  if the user assembles a CollectiveMma manually and ensures that the runtime strides are compatible
  with the static micro-kernel of the collective (TiledMma, TiledCopy, and smem layouts). On the other
  hand, a user can have more than one static stride too (which need not correspond to the major mode).

  In particular, this example demonstrates a GETT where the 0th M-mode (M0) in A and the 0th K-mode (K0)
  in B are major. All other combinations of major modes are supported, with the exception of mixed
  K-major scenarios where both A and B are K-major (e.g. K0 is major in A but K1 is major in B).
  NVIDIA Hopper architecture's TMA feature makes the predictaion required to implement these complicated
  kernels trivial, as it is all handled by TMA itself without requiring any programmer effort.

  Example executions, where the stride order defines the major-order (major on the left):
  51_hopper_gett --modeC=m,n,l --modeA=m,k,l --modeB=k,n,l --extents=m:4096,n:4096,k:4096
  51_hopper_gett --modeC=l,m,n --modeA=m,l,k --modeB=k,n,l --extents=m:128,n:128,k:128,l:64
  51_hopper_gett --modeC=m,a,b,p,q,n,l --modeA=m,l,b,k,a --modeB=k,n,p,q,l --extents=m:32,a:32,b:3,n:128,k:128,l:4,p:3,q:3
*/

#include "gett_kernel.cuh"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

#include "cutlass/util/gett_commandline.hpp"
#include "cutlass/util/reference/device/gett.hpp"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/print_error.hpp"

namespace example {

// Returns true if the left-most value in the tuple is statically known to be 1
template<class Stride>
constexpr bool
is_left_major() {
  // Account for stride types with and without batch mode and batch modes with static zero stride
  return cute::is_constant<1, decltype(cute::size<0,0>(Stride{}))>::value;
}

// Same as cute::make_int_tuple but inserts a major stride (Int<1>) for the leftmost mode if required
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

} // namespace example

//////////////////////////////////////////////////////////////////////////////

int
main(int argc, char const* argv[]) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  using namespace cute;

  if (argc != 5) {
    std::cout << "Number of command line args must be 4.\n";
    cutlass::GettCommandLine::print_usage();
    return 0;
  }

  //
  // Define the stride types for A, B, C, and D
  //

  // Stride for A (left input). If reduction mode is major, same must be major in B
  // For this example, M0 is major in A.
  using RowModeStridesA = cute::Stride<cute::Int<1>, int64_t, int64_t, int64_t>;
  using RedModeStridesA = cute::Stride<int64_t, int64_t, int64_t>;
  using BatModeStridesA = cute::Stride<int64_t, int64_t, int64_t, int64_t>;

  // Stride for B (right input). If reduction mode is major, same must be major in A
  // For this example, K0 is major in B.
  using ColModeStridesB = cute::Stride<int64_t, int64_t, int64_t, int64_t>;
  using RedModeStridesB = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  using BatModeStridesB = cute::Stride<int64_t, int64_t, int64_t, int64_t>;

  // Strides for output, which can all be dynamic.
  using RowModeStridesC = cute::Stride<int64_t, int64_t, int64_t, int64_t>;
  using ColModeStridesC = cute::Stride<int64_t, int64_t, int64_t, int64_t>;
  using BatModeStridesC = cute::Stride<int64_t, int64_t, int64_t, int64_t>;

  // Assmble our rank-3 multi-mode strides for the in/out tensors
  using StrideA = cute::Stride<RowModeStridesA, RedModeStridesA, BatModeStridesA>;
  using StrideB = cute::Stride<ColModeStridesB, RedModeStridesB, BatModeStridesB>;
  using StrideC = cute::Stride<RowModeStridesC, ColModeStridesC, BatModeStridesC>;

  // Note: C and D share strides here for simplicity.
  //       In general, they need not have the same layout.
  using StrideD = StrideC;

  //
  // Define element types for tensors and intermediate values
  //
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementEpilogue = float;

  // The following constexpr values set the max number of modes in each MNKL mode
  constexpr int MaxRank_M = cute::rank(RowModeStridesA{}); // Max row modes
  constexpr int MaxRank_N = cute::rank(ColModeStridesB{}); // Max column modes
  constexpr int MaxRank_K = cute::rank(RedModeStridesA{}); // Max contraction modes
  constexpr int MaxRank_L = cute::rank(BatModeStridesA{}); // Max batch modes
  static_assert(cute::rank(RowModeStridesA{}) == cute::rank(RowModeStridesC{}));
  static_assert(cute::rank(ColModeStridesB{}) == cute::rank(RowModeStridesC{}));
  static_assert(cute::rank(RedModeStridesA{}) == cute::rank(RedModeStridesB{}));
  static_assert(cute::rank(BatModeStridesA{}) == cute::rank(BatModeStridesC{}));
  static_assert(cute::rank(BatModeStridesB{}) == cute::rank(BatModeStridesC{}));

  // Parse command line to get modes, extents, and strides
  cutlass::GettCommandLine cmd;
  auto parsed_args = cmd.parse(argc, argv, true);

  auto& m = parsed_args.M;
  auto& ldAm = parsed_args.ldAm;
  auto& ldCm = parsed_args.ldCm;
  int rank_m = int(m.size());

  auto& n = parsed_args.N;
  auto& ldBn = parsed_args.ldBn;
  auto& ldCn = parsed_args.ldCn;
  int rank_n = int(n.size());

  auto& k = parsed_args.K;
  auto& ldAk = parsed_args.ldAk;
  auto& ldBk = parsed_args.ldBk;
  int rank_k = int(k.size());

  auto& l = parsed_args.L;
  auto& ldAl = parsed_args.ldAl;
  auto& ldBl = parsed_args.ldBl;
  auto& ldCl = parsed_args.ldCl;
  int rank_l = int(l.size());

  if ((rank_m > MaxRank_M) || (rank_n > MaxRank_N) || (rank_k > MaxRank_K) || (rank_l > MaxRank_L)) {
    std::cerr << "ERROR: Input has more modes than statically configured.";
    return 1;
  }

  // Check that the user input major stride match the static major strides.
  if (example::is_left_major<RowModeStridesA>() && (ldAm[0] != 1)) {
    std::cerr << "ERROR: A_M0 is expected to be major, but was not in the provided input!\n";
    return 1;
  }

  if (example::is_left_major<RedModeStridesA>() && (ldAk[0] != 1)) {
    std::cerr << "ERROR: A_K0 is expected to be major, but was not in the provided input!\n";
    return 1;
  }

  if (example::is_left_major<ColModeStridesB>() && (ldBn[0] != 1)) {
    std::cerr << "ERROR: B_N0 is expected to be major, but was not in the provided input!\n";
    return 1;
  }

  if (example::is_left_major<RedModeStridesB>() && (ldBk[0] != 1)) {
    std::cerr << "ERROR: B_K0 is expected to be major, but was not in the provided input!\n";
    return 1;
  }

  // Convert to `cute::Tuple`s and set up arguments
  auto M   = make_int_tuple<MaxRank_M>(m.data(), rank_m, 1);
  auto dAm = example::make_stride_tuple<MaxRank_M, example::is_left_major<RowModeStridesA>()>(ldAm.data(), rank_m);
  auto dCm = example::make_stride_tuple<MaxRank_M, example::is_left_major<RowModeStridesC>()>(ldCm.data(), rank_m);

  auto N   = make_int_tuple<MaxRank_N>(n.data(), rank_n, 1);
  auto dBn = example::make_stride_tuple<MaxRank_N, example::is_left_major<ColModeStridesB>()>(ldBn.data(), rank_n);
  auto dCn = example::make_stride_tuple<MaxRank_N, example::is_left_major<ColModeStridesC>()>(ldCn.data(), rank_n);

  auto K   = make_int_tuple<MaxRank_K>(k.data(), rank_k, 1);
  auto dAk = example::make_stride_tuple<MaxRank_K, example::is_left_major<RedModeStridesA>()>(ldAk.data(), rank_k);
  auto dBk = example::make_stride_tuple<MaxRank_K, example::is_left_major<RedModeStridesB>()>(ldBk.data(), rank_k);

  auto L   = make_int_tuple<MaxRank_L>(l.data(), rank_l, 1);
  auto dAl = make_int_tuple<MaxRank_L>(ldAl.data(), rank_l, 0);
  auto dBl = make_int_tuple<MaxRank_L>(ldBl.data(), rank_l, 0);
  auto dCl = make_int_tuple<MaxRank_L>(ldCl.data(), rank_l, 0);

  // Concat tuples to turn it into rank-4 problem shape and rank-3 strides, just like GEMM
  auto problem_shape = make_shape(M, N, K, L);
  StrideA stride_A   = make_stride(dAm, dAk, dAl);
  StrideB stride_B   = make_stride(dBn, dBk, dBl);
  StrideC stride_C   = make_stride(dCm, dCn, dCl);
  StrideD stride_D   = stride_C;

  auto alpha = ElementEpilogue(1.0f);
  auto beta  = ElementEpilogue(1.0f);

  //
  // Allocate and init tensors
  //
  auto M_size = std::accumulate(std::begin(m), std::end(m), 1, std::multiplies<>{});
  auto N_size = std::accumulate(std::begin(n), std::end(n), 1, std::multiplies<>{});
  auto K_size = std::accumulate(std::begin(k), std::end(k), 1, std::multiplies<>{});
  auto L_size = std::accumulate(std::begin(l), std::end(l), 1, std::multiplies<>{});

  thrust::host_vector<ElementA> h_A(M_size * K_size * L_size);
  thrust::host_vector<ElementB> h_B(N_size * K_size * L_size);
  thrust::host_vector<ElementC> h_C(M_size * N_size * L_size);
  thrust::host_vector<ElementD> h_D(M_size * N_size * L_size);

  // Note: the cast to int here is to avoid false-negative ref-checks which can
  // occur due to floating point arithmetic not being purely associative.
  for (auto& a : h_A) a = ElementA(int(4*(rand() / double(RAND_MAX)) - 1));
  for (auto& b : h_B) b = ElementB(int(4*(rand() / double(RAND_MAX)) - 1));
  for (auto& c : h_C) c = ElementC(int(4*(rand() / double(RAND_MAX)) - 1));
  for (auto& d : h_D) d = ElementD(-1);

  thrust::device_vector<ElementA> d_A = h_A;
  thrust::device_vector<ElementB> d_B = h_B;
  thrust::device_vector<ElementC> d_C = h_C;
  thrust::device_vector<ElementD> cutlass_result = h_D;
  thrust::device_vector<ElementD> reference_result = h_D;

  //
  // Compute GETT
  //
  auto status = example::gett_kernel(
    problem_shape,
    d_A.data().get(), stride_A,
    d_B.data().get(), stride_B,
    ElementAccumulator{},
    d_C.data().get(), stride_C,
    cutlass_result.data().get(), stride_D,
    alpha, beta);

  if (cutlass::Status::kSuccess != status) {
    std::cerr << "ERROR: GETT operator launch failed.\n";
    return 1;
  }

  auto cuda_err = cudaDeviceSynchronize();
  if (cudaSuccess != cuda_err) {
    std::cerr << "ERROR: GETT operator execution failed. with error :";
    std::cerr << cudaGetErrorString(cuda_err) << "\n";
    return 1;
  }

  //
  // Verify
  //

  cutlass::reference::device::gett(
    problem_shape,
    d_A.data().get(), stride_A,
    d_B.data().get(), stride_B,
    ElementAccumulator{},
    d_C.data().get(), stride_C,
    reference_result.data().get(), stride_D,
    alpha, beta);

  cuda_err = cudaDeviceSynchronize();
  if (cudaSuccess != cuda_err) {
    std::cerr << "ERROR: GETT reference execution failed. with error :";
    std::cerr << cudaGetErrorString(cuda_err) << "\n";
    return 1;
  }

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::device::BlockCompareEqual(
      reference_result.data().get(), cutlass_result.data().get(), cutlass_result.size());
  if (passed) {
    std::cout << "GETT verification passed.\n";
    return 0;
  }
  else {
    std::cerr << "ERROR: GETT verification failed! Printing detailed stats.\n";
    h_D = reference_result;
    thrust::host_vector<ElementD> h_cutlass_result = cutlass_result;
    print_relative_error(h_cutlass_result.size(), h_cutlass_result.data(), h_D.data());

    std::cout << "StrideA: "; print(stride_A); std::cout << '\n';
    std::cout << "StrideB: "; print(stride_B); std::cout << '\n';
    std::cout << "StrideC: "; print(stride_C); std::cout << '\n';
    std::cout << "StrideD: "; print(stride_D); std::cout << '\n';
    return 1;
  }
#else
  std::cerr << "Unsupported example. Please ensure CUTLASS_ARCH_MMA_SM90_SUPPORTED is defined.\n";
  return 0;
#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
}
