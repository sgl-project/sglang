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
#pragma once

#include <cute/config.hpp>
#include <cute/util/type_traits.hpp>

#include <cute/atom/mma_atom.hpp>

#include <cute/algorithm/axpby.hpp>
#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cute/tensor_impl.hpp>

namespace cute
{

//
// Cooperative Shared-Memory GEMMs
//

namespace detail {

// Slow fallback path:
template<typename ... Args,
         typename Alpha, typename TRC, typename RCLayout,
         typename Beta, class TSC, typename CLayout, typename SCLayout,
         typename CLoadTransformOp, typename CStoreTransformOp>
CUTE_HOST_DEVICE
void
epilogue_predication(ThrMMA<Args...>    const& thr_mma,
                     Alpha              const& alpha,
                     Tensor<TRC, RCLayout>   & tCrC,
                     Beta               const& beta,
                     Tensor<TSC, CLayout>    & sC,
                     Tensor<TSC, SCLayout>   & tCsC,
                     CLoadTransformOp   const& sC_load_op,  // transforms C values before use in GEMM
                     CStoreTransformOp  const& sC_store_op) // transforms results before they are stored to C
{
  using InputTypeC   = typename TSC::value_type;
  using ComputeTypeC = typename ThrMMA<Args...>::ValTypeC;
  CUTE_STATIC_ASSERT(CUTE_STL_NAMESPACE::is_same_v<ComputeTypeC, typename TRC::value_type>);

  // Create coordinate tensors for the problem
  Tensor cC   = make_identity_tensor(shape(sC));                     // (M,N) -> (m,n)
  // Repeat partitioning with thr_mma
  Tensor tCcC = thr_mma.partition_C(cC);                             // (MMA,MMA_M,MMA_N) -> (m,n)

  const bool isBetaZero = [&] () {
    if constexpr (is_complex<Beta>::value) {
      return beta.real() == Int<0>{} && beta.imag() == Int<0>{};
    }
    else {
      return beta == Int<0>{};
    }
    CUTE_GCC_UNREACHABLE;
  } ();

  // Custom axpby_if for now
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i)
  {
    if (elem_less(tCcC(i), shape(sC)))
    {
      tCsC(i) = sC_store_op(isBetaZero ? alpha * tCrC(i)
                                       : alpha * tCrC(i) +
                                          beta * static_cast<ComputeTypeC>(sC_load_op(tCsC(i))));
    }
  }
}

template<class ... Args, 
         class Alpha, class TRC, class RCLayout,
         class Beta, class TSC, class SCLayout,
         class CLoadTransformOp, class CStoreTransformOp,
         class SmemCopyLdOpC, class SmemCopyStOpC>
CUTE_HOST_DEVICE
void
epilogue_no_predication(uint32_t                   thread_idx,
                        ThrMMA<Args...>     const& thr_mma,
                        Alpha              const& alpha,
                        Tensor<TRC, RCLayout>   & tCrC,
                        Beta               const& beta,
                        Tensor<TSC, SCLayout>   & sC,
                        CLoadTransformOp   const& sC_load_op,  // transforms C values before use in GEMM
                        CStoreTransformOp  const& sC_store_op, // transforms results before they are stored to C
                        SmemCopyLdOpC      const& sC_copy_ld_op,
                        SmemCopyStOpC      const& sC_copy_st_op)
{
  using InputTypeC   = typename TSC::value_type;
  using ComputeTypeC = typename TRC::value_type;

  const bool isBetaZero = [&] () {
    if constexpr (is_complex<Beta>::value) {
      return beta.real() == Int<0>{} && beta.imag() == Int<0>{};
    }
    else {
      return beta == Int<0>{};
    }
    CUTE_GCC_UNREACHABLE;
  } ();

  Tensor tCrD = make_fragment_like(tCrC);
  Tensor tCrDi = make_fragment_like<InputTypeC>(tCrD);

  if(!isBetaZero) {
    auto smem_tiled_copy_C = make_tiled_copy_C(Copy_Atom<SmemCopyLdOpC, InputTypeC>{}, thr_mma);
    auto smem_thr_copy_C   = smem_tiled_copy_C.get_thread_slice(thread_idx);
    Tensor tCsC            = smem_thr_copy_C.partition_S(sC);
    Tensor tCrDi_copy_view = smem_thr_copy_C.retile_D(tCrDi);
    CUTE_STATIC_ASSERT_V(size<1>(tCsC) == size<1>(tCrDi_copy_view));             // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsC) == size<2>(tCrDi_copy_view));             // CPY_N
    copy(smem_tiled_copy_C, tCsC, tCrDi_copy_view);

    // Transform C on/after load
    cute::transform(tCrDi, tCrD, sC_load_op);
  }
  // C = alpha * (A * B) + beta * C
  axpby(alpha, tCrC, beta, tCrD);
  // Transform C before/on store
  cute::transform(tCrD, tCrDi, sC_store_op);

  auto smem_tiled_copy_C = make_tiled_copy_C(Copy_Atom<SmemCopyStOpC, InputTypeC>{}, thr_mma);
  auto smem_thr_copy_C   = smem_tiled_copy_C.get_thread_slice(thread_idx);
  Tensor tCsC            = smem_thr_copy_C.partition_D(sC);
  Tensor tCrDi_copy_view = smem_thr_copy_C.retile_S(tCrDi);
  CUTE_STATIC_ASSERT_V(size<1>(tCsC) == size<1>(tCrDi_copy_view));             // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tCsC) == size<2>(tCrDi_copy_view));             // CPY_N
  copy(smem_tiled_copy_C, tCrDi_copy_view, tCsC);
}

// Predicated Cooperative GEMM
template <class... Args,
          class TA, class ALayout, class TB, class BLayout,
          class TC, class RCLayout,
          class ALoadTransformOp, class BLoadTransformOp>
CUTE_HOST_DEVICE
void
cooperative_gemm_predication(ThrMMA<Args...>     const& thr_mma,
                             Tensor<TA, ALayout> const& sA,
                             Tensor<TB, BLayout> const& sB,
                             Tensor<TC, RCLayout>     & tCrC,
                             ALoadTransformOp    const& sA_load_op,  // transforms A values before use in GEMM
                             BLoadTransformOp    const& sB_load_op)  // transforms B values before use in GEMM
{
  using InputTypeA        = typename TA::value_type;
  using InputTypeB        = typename TB::value_type;
  using InputTypeC        = typename TC::value_type;
  using ComputeTypeA = typename ThrMMA<Args...>::ValTypeA;
  using ComputeTypeB = typename ThrMMA<Args...>::ValTypeB;
  using ComputeTypeC = typename ThrMMA<Args...>::ValTypeC;

  //
  // MMA Partitioning
  //

  // Partition the sA, sB, and sC tiles across the threads for the MMA
  Tensor tCsA = thr_mma.partition_A(sA);                            // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);                            // (MMA,MMA_N,MMA_K)

  // Create register tensors for the MMA to operate on
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                      // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                      // (MMA,MMA_N,MMA_K)
  //
  // PREDICATION
  //

  // Create coordinate tensors for the problem
  Tensor cA = make_identity_tensor(shape(sA));                      // (M,K) -> (m,k)
  Tensor cB = make_identity_tensor(shape(sB));                      // (N,K) -> (n,k)

  // Repeat partitioning with thr_mma
  Tensor tCcA = thr_mma.partition_A(cA);                            // (MMA,MMA_M,MMA_K) -> (m,k)
  Tensor tCcB = thr_mma.partition_B(cB);                            // (MMA,MMA_N,MMA_K) -> (n,k)

  // Allocate the preds for MMA- and MMA_MN-modes
  Tensor tCpA = make_tensor<bool>(make_shape(size<0>(tCsA), size<1>(tCsA)));
  Tensor tCpB = make_tensor<bool>(make_shape(size<0>(tCsB), size<1>(tCsB)));
  // Populate the predicates on M and N
  CUTE_UNROLL
  for (int i = 0; i < size(tCpA); ++i) {
    tCpA(i) = elem_less(get<0>(tCcA(_,_,Int<0>{})(i)), shape<0>(sA));
  }
  CUTE_UNROLL
  for (int i = 0; i < size(tCpB); ++i) {
    tCpB(i) = elem_less(get<0>(tCcB(_,_,Int<0>{})(i)), shape<0>(sB));
  }
  //
  // PREFETCH k_block = 0
  //   Condition the k-predication on (static) k_block == K_BLOCK_MAX-1, the last k_block
  //   Assumes the MMA-tiling in K is trivial
  //

  constexpr int K_BLOCK_MAX = size<2>(tCrA);

  CUTE_UNROLL
  for (int m = 0; m < size<1>(tCrA); ++m) {     // Copy MMA_M
    CUTE_UNROLL
    for (int i = 0; i < size<0>(tCrA); ++i) {   // Copy MMA_I
      tCrA(i,m,0) = (tCpA(i,m) && (0 < K_BLOCK_MAX-1 || elem_less(get<1>(tCcA(i,m,0)), shape<1>(sA)))) ? static_cast<ComputeTypeA>(sA_load_op(tCsA(i,m,0))) : ComputeTypeA{};
    }
  }
  CUTE_UNROLL
  for (int n = 0; n < size<1>(tCrB); ++n) {     // Copy MMA_N
    CUTE_UNROLL
    for (int i = 0; i < size<0>(tCrB); ++i) {   // Copy MMA_I
      tCrB(i,n,0) = (tCpB(i,n) && (0 < K_BLOCK_MAX-1 || elem_less(get<1>(tCcB(i,n,0)), shape<1>(sB)))) ? static_cast<ComputeTypeB>(sB_load_op(tCsB(i,n,0))) : ComputeTypeB{};
    }
  }
  //
  // MAINLOOP
  //

  CUTE_UNROLL
  for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
  {
    if (k_block < K_BLOCK_MAX-1)   // static-if not the last k_block
    {
      int k_next = k_block + 1;    // Load k_next block

      //   Condition the k-predication on (static) k_block == K_BLOCK_MAX-1, the last k_block
      //   Assumes the MMA-tiling in K is trivial

      CUTE_UNROLL
      for (int m = 0; m < size<1>(tCrA); ++m) {       // Copy MMA_M
        CUTE_UNROLL
        for (int i = 0; i < size<0>(tCrA); ++i) {     // Copy MMA_I
          tCrA(i,m,k_next) = (tCpA(i,m) && (k_next < K_BLOCK_MAX-1 || elem_less(get<1>(tCcA(i,m,k_next)), shape<1>(sA)))) ? static_cast<ComputeTypeA>(sA_load_op(tCsA(i,m,k_next))) : ComputeTypeA{};
        }
      }
      CUTE_UNROLL
      for (int n = 0; n < size<1>(tCrB); ++n) {       // Copy MMA_N
        CUTE_UNROLL
        for (int i = 0; i < size<0>(tCrB); ++i) {     // Copy MMA_I
          tCrB(i,n,k_next) = (tCpB(i,n) && (k_next < K_BLOCK_MAX-1 || elem_less(get<1>(tCcB(i,n,k_next)), shape<1>(sB)))) ? static_cast<ComputeTypeB>(sB_load_op(tCsB(i,n,k_next))) : ComputeTypeB{};
        }
      }
    }
    // GEMM on k_block in registers
    gemm(thr_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
  }
}

// Unpredicated Cooperative GEMM
template <class... Args,
          class TA, class ALayout, class TB, class BLayout,
          class TC, class CLayout,
          class ALoadTransformOp, class BLoadTransformOp,
          class SmemCopyOpA, class SmemCopyOpB>
CUTE_HOST_DEVICE
void
cooperative_gemm_no_predication(uint32_t                   thread_idx,
                                ThrMMA<Args...>     const& thr_mma,
                                Tensor<TA, ALayout> const& sA,
                                Tensor<TB, BLayout> const& sB,
                                Tensor<TC, CLayout>      & tCrC,
                                ALoadTransformOp    const& sA_load_op,  // transforms A values before use in GEMM
                                BLoadTransformOp    const& sB_load_op,  // transforms B values before use in GEMM
                                SmemCopyOpA         const& sA_copy_op,
                                SmemCopyOpB         const& sB_copy_op)
{
  using InputTypeA        = typename TA::value_type;
  using InputTypeB        = typename TB::value_type;
  using InputTypeC        = typename TC::value_type;
  using ComputeTypeA = typename ThrMMA<Args...>::ValTypeA;
  using ComputeTypeB = typename ThrMMA<Args...>::ValTypeB;
  using ComputeTypeC = typename ThrMMA<Args...>::ValTypeC;


  //
  // MMA Partitioning
  //

  // Create register tensors for the MMA to operate on
  Tensor tCrA  = thr_mma.partition_fragment_A(sA);                    // (MMA,MMA_M,MMA_K)
  Tensor tCrAi = make_fragment_like<InputTypeA>(tCrA);
  Tensor tCrB  = thr_mma.partition_fragment_B(sB);                    // (MMA,MMA_N,MMA_K)
  Tensor tCrBi = make_fragment_like<InputTypeB>(tCrB);

  using CopyOpAType = SmemCopyOpA;
  using CopyOpBType = SmemCopyOpB;

  auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<CopyOpAType, InputTypeA>{}, thr_mma);
  auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
  Tensor tCsA            = smem_thr_copy_A.partition_S(sA);
  Tensor tCrAi_copy_view = smem_thr_copy_A.retile_D(tCrAi);
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrAi_copy_view));             // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrAi_copy_view));             // CPY_K

  auto smem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<CopyOpBType, InputTypeB>{}, thr_mma);
  auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
  Tensor tCsB            = smem_thr_copy_B.partition_S(sB);
  Tensor tCrBi_copy_view = smem_thr_copy_B.retile_D(tCrBi);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrBi_copy_view));            // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrBi_copy_view));            // CPY_K
  //
  // PREFETCH
  //

  copy(smem_tiled_copy_A, tCsA(_,_,Int<0>{}), tCrAi_copy_view(_,_,Int<0>{}));
  copy(smem_tiled_copy_B, tCsB(_,_,Int<0>{}), tCrBi_copy_view(_,_,Int<0>{}));
  //
  // MAINLOOP
  //

  constexpr int K_BLOCK_MAX = size<2>(tCrA);

  CUTE_UNROLL
  for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
  {
    // static-if load the next k_block. No k-predication required on these loads.
    if (k_block < K_BLOCK_MAX-1)
    {
      // Load the next k_block
      int k_next = k_block + 1;       // statically unrolled
      copy(smem_tiled_copy_A, tCsA(_,_,k_next), tCrAi_copy_view(_,_,k_next));
      copy(smem_tiled_copy_B, tCsB(_,_,k_next), tCrBi_copy_view(_,_,k_next));
    }

    // Transform A and B, relying on the compiler to remove in case of identity ops
    cute::transform(tCrAi(_,_,k_block), tCrA(_,_,k_block), sA_load_op);
    cute::transform(tCrBi(_,_,k_block), tCrB(_,_,k_block), sB_load_op);

    // GEMM on k_block in registers
    gemm(thr_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
  }
}

} // end namespace detail

// C passed as a shared memory tensor
// Epilogue included
template <class... Args,
          class Alpha, class TA, class ALayout, class TB, class BLayout,
          class Beta,  class TC, class CLayout,
          class ALoadTransformOp = cute::identity, class BLoadTransformOp  = cute::identity,
          class CLoadTransformOp = cute::identity, class CStoreTransformOp = cute::identity,
          class SmemCopyOpA = DefaultCopy, class SmemCopyOpB = DefaultCopy,
          class SmemCopyLdOpC = DefaultCopy, class SmemCopyStOpC = DefaultCopy>
CUTE_HOST_DEVICE
void
cooperative_gemm(uint32_t                   thread_idx,
                 TiledMMA<Args...>   const& tiled_mma,
                 Alpha               const& alpha,
                 Tensor<TA, ALayout> const& sA,
                 Tensor<TB, BLayout> const& sB,
                 Beta                const& beta,
                 Tensor<TC, CLayout>      & sC,
                 ALoadTransformOp    const& sA_load_op    = {}, // transforms A values before use in GEMM
                 BLoadTransformOp    const& sB_load_op    = {}, // transforms B values before use in GEMM
                 CLoadTransformOp    const& sC_load_op    = {}, // transforms C values before use in GEMM
                 CStoreTransformOp   const& sC_store_op   = {}, // transforms results before they are stored to C
                 SmemCopyOpA         const& sA_copy_op    = {},
                 SmemCopyOpB         const& sB_copy_op    = {},
                 SmemCopyLdOpC       const& sC_copy_ld_op = {},
                 SmemCopyStOpC       const& sC_copy_st_op = {})
{
  CUTE_STATIC_ASSERT_V(rank(sA) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(sB) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(sC) == Int<2>{});

  CUTE_STATIC_ASSERT_V(size<0>(sA) == size<0>(sC));  // AM == CM
  CUTE_STATIC_ASSERT_V(size<0>(sB) == size<1>(sC));  // BN == CN
  CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));  // AK == BK

  using InputTypeA        = typename TA::value_type;
  using InputTypeB        = typename TB::value_type;
  using InputTypeC        = typename TC::value_type;
  using ComputeTypeA = typename TiledMMA<Args...>::ValTypeA;
  using ComputeTypeB = typename TiledMMA<Args...>::ValTypeB;
  using ComputeTypeC = typename TiledMMA<Args...>::ValTypeC;

  auto compat = evenly_divides(make_shape(size<0>(sA), size<0>(sB), size<1>(sA)),
                               tile_shape(TiledMMA<Args...>{}));

  // ThrMMA
  auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
  Tensor tCsC  = thr_mma.partition_C(sC);                             // (MMA,MMA_M,MMA_N) :: InputTypeC
  Tensor tCrC  = thr_mma.make_fragment_C(tCsC);                       // (MMA,MMA_M,MMA_N) :: ComputeTypeC

  // Clear accumulators
  clear(tCrC);
  if constexpr (is_constant<true, decltype(compat)>::value) {
    detail::cooperative_gemm_no_predication(
        thread_idx, thr_mma, sA, sB, tCrC, sA_load_op, sB_load_op, sA_copy_op, sB_copy_op
    );
    detail::epilogue_no_predication(
        thread_idx, thr_mma,alpha, tCrC, beta, sC, sC_load_op, sC_store_op, sC_copy_ld_op, sC_copy_st_op
    );
  } else {
    detail::cooperative_gemm_predication(
        thr_mma, sA, sB, tCrC, sA_load_op, sB_load_op
    );
    detail::epilogue_predication(
        thr_mma, alpha, tCrC, beta, sC, tCsC, sC_load_op, sC_store_op
    );
  }
}

// C already partitioned into registers on input
// It can be passed non-empty
// Epilogue not included
template <class... Args,
          class TA, class ALayout, class TB, class BLayout,
          class TC, class CLayout,
          class ALoadTransformOp = cute::identity, class BLoadTransformOp  = cute::identity,
          class SmemCopyOpA = DefaultCopy, class SmemCopyOpB = DefaultCopy>
CUTE_HOST_DEVICE
void
cooperative_gemm(uint32_t                   thread_idx,
                 TiledMMA<Args...>   const& tiled_mma,
                 Tensor<TA, ALayout> const& sA,
                 Tensor<TB, BLayout> const& sB,
                 Tensor<TC, CLayout>      & tCrC,
                 ALoadTransformOp    const& sA_load_op  = {}, // transforms A values before use in GEMM
                 BLoadTransformOp    const& sB_load_op  = {}, // transforms B values before use in GEMM
                 SmemCopyOpA         const& sA_copy_op  = {},
                 SmemCopyOpB         const& sB_copy_op  = {})
{
  CUTE_STATIC_ASSERT_V(rank(sA) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(sB) == Int<2>{});

  CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));  // AK == BK

  using InputTypeA        = typename TA::value_type;
  using InputTypeB        = typename TB::value_type;
  using InputTypeC        = typename TC::value_type;
  using ComputeTypeA = typename TiledMMA<Args...>::ValTypeA;
  using ComputeTypeB = typename TiledMMA<Args...>::ValTypeB;
  using ComputeTypeC = typename TiledMMA<Args...>::ValTypeC;

  // Check if input C fragment is compatible with thr_mma and problem size
  using ref_c_frag = decltype(partition_shape_C(tiled_mma, make_shape(size<0>(sA), size<0>(sB))));
  CUTE_STATIC_ASSERT_V(compatible(shape(ref_c_frag{}), shape(tCrC)));

  auto compat = evenly_divides(make_shape(size<0>(sA), size<0>(sB), size<1>(sA)),
                               tile_shape(TiledMMA<Args...>{}));

  // ThrMMA
  auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

  if constexpr (is_constant<true, decltype(compat)>::value) {
    detail::cooperative_gemm_no_predication(
        thread_idx, thr_mma, sA, sB, tCrC, sA_load_op, sB_load_op, sA_copy_op, sB_copy_op
    );
  } else {
    detail::cooperative_gemm_predication(
        thr_mma, sA, sB, tCrC, sA_load_op, sB_load_op
    );
  }
}

// Accept mutable temporaries
template <class... Args,
          class Alpha, class TA, class ALayout, class TB, class BLayout,
          class Beta,  class TC, class CLayout,
          class ALoadTransformOp = cute::identity, class BLoadTransformOp  = cute::identity,
          class CLoadTransformOp = cute::identity, class CStoreTransformOp = cute::identity,
          class SmemCopyOpA = DefaultCopy, class SmemCopyOpB = DefaultCopy,
          class SmemCopyLdOpC = DefaultCopy, class SmemCopyStOpC = DefaultCopy>
CUTE_HOST_DEVICE
void
cooperative_gemm(uint32_t thread_idx,
                 TiledMMA<Args...>   const& tiled_mma,
                 Alpha               const& alpha,
                 Tensor<TA, ALayout> const& sA,
                 Tensor<TB, BLayout> const& sB,
                 Beta                const& beta,
                 Tensor<TC, CLayout>     && sC,
                 ALoadTransformOp    const& sA_load_op    = {}, // transforms A values before use in GEMM
                 BLoadTransformOp    const& sB_load_op    = {}, // transforms B values before use in GEMM
                 CLoadTransformOp    const& sC_load_op    = {}, // transforms C values before use in GEMM
                 CStoreTransformOp   const& sC_store_op   = {}, // transforms results before they are stored to C
                 SmemCopyOpA         const& sA_copy_op    = {},
                 SmemCopyOpB         const& sB_copy_op    = {},
                 SmemCopyLdOpC       const& sC_copy_ld_op = {},
                 SmemCopyStOpC       const& sC_copy_st_op = {})
{
  cooperative_gemm(thread_idx, tiled_mma, alpha, sA, sB, beta, sC,
                   sA_load_op, sB_load_op, sC_load_op, sC_store_op,
                   sA_copy_op, sB_copy_op, sC_copy_ld_op, sC_copy_st_op);
}

// Legacy overload of cute::gemm for backwards-compatibility
template <class... Args,
          class Alpha, class TA, class ALayout, class TB, class BLayout,
          class Beta,  class TC, class CLayout,
          class ALoadTransformOp = cute::identity, class BLoadTransformOp  = cute::identity,
          class CLoadTransformOp = cute::identity, class CStoreTransformOp = cute::identity>
CUTE_HOST_DEVICE
void
gemm(ThrMMA<Args...>     const& thr_mma,
     Alpha               const& alpha,
     Tensor<TA, ALayout> const& sA,
     Tensor<TB, BLayout> const& sB,
     Beta                const& beta,
     Tensor<TC, CLayout>      & sC,
     ALoadTransformOp    const& sA_load_op  = {}, // transforms A values before use in GEMM
     BLoadTransformOp    const& sB_load_op  = {}, // transforms B values before use in GEMM
     CLoadTransformOp    const& sC_load_op  = {}, // transforms C values before use in GEMM
     CStoreTransformOp   const& sC_store_op = {}) // transforms results before they are stored to C
{
  CUTE_STATIC_ASSERT_V(rank(sA) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(sB) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(sC) == Int<2>{});

  CUTE_STATIC_ASSERT_V(size<0>(sA) == size<0>(sC));  // AM == CM
  CUTE_STATIC_ASSERT_V(size<0>(sB) == size<1>(sC));  // BN == CN
  CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));  // AK == BK

  Tensor tCsC  = thr_mma.partition_C(sC);                           // (MMA,MMA_M,MMA_N)
  Tensor tCrC  = thr_mma.make_fragment_C(tCsC);                     // (MMA,MMA_M,MMA_N)

  // Goes directly to the slow path to avoid getting thread_idx from thr_mma
  detail::cooperative_gemm_predication(
    thr_mma, sA, sB, sC, sA_load_op, sB_load_op
  );

  detail::epilogue_predication(
      thr_mma, alpha, tCrC, beta, sC, tCsC, sC_load_op, sC_store_op
  );
}

} // end namespace cute
