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

#include <cute/tensor_impl.hpp>  // cute::Tensor
#include <cute/pointer.hpp>      // cute::is_rmem
#include <cute/arch/mma.hpp>     // cute::UniversalFMA
#include <cute/arch/util.hpp>    // cute::detail::explode

namespace cute
{

/**
 * concept MMA_Traits
 * {
 *   using ValTypeD =  // Logical D-value type
 *   using ValTypeA =  // Logical A-value type
 *   using ValTypeB =  // Logical B-value type
 *   using ValTypeC =  // Logical C-value type    (NOTE: Not used? Assumed == ValTypeD)
 *
 *   using FrgTypeA =  // A-type consumed by MMA  (if ommitted, same as ValTypeA)
 *   using FrgTypeB =  // B_type consumed by MMA  (if ommitted, same as ValTypeB)
 *   using FrgTypeC =  // C_type consumed by MMA  (if ommitted, same as ValTypeC)
 *
 *   using Shape_MNK =    // Logical MxNxK shape of the MMA
 *
 *   using ThrID     =    // Logical thread id (tid) -> tidx
 *
 *   using ALayout =      // (Logical thread id (tid), Logical value id (vid)) -> Flat MK-coord
 *   using BLayout =      // (Logical thread id (tid), Logical value id (vid)) -> Flat NK-coord
 *   using CLayout =      // (Logical thread id (tid), Logical value id (vid)) -> Flat MN-coord
 * };
 */

template <class MMAOperation, class... MMAOpArgs>
struct MMA_Traits
{
  static_assert(sizeof(MMAOperation) == 0, "MMA_Traits not implemented for this MMA_Operation.");
};

template <class D, class A, class B, class C>
struct MMA_Traits<UniversalFMA<D,A,B,C>>
{
  using ValTypeD = D;
  using ValTypeA = A;
  using ValTypeB = B;
  using ValTypeC = C;

  // Logical shape of the MMA
  using Shape_MNK = Shape<_1,_1,_1>;

  // Logical thread id (tid) -> tidx
  using ThrID   = Layout<_1>;

  // (Logical thread id (tid), Logical value id (vid)) -> coord

  // (tid,vid) -> (m,k)
  using ALayout = Layout<Shape<_1,_1>>;
  // (tid,vid) -> (n,k)
  using BLayout = Layout<Shape<_1,_1>>;
  // (tid,vid) -> (m,n)
  using CLayout = Layout<Shape<_1,_1>>;
};

// Extract an MMA_Op from an MMA_Traits
template <class MMA_Traits>
struct MMA_Op {};

template <class MMA_Op_Arg, class... Args>
struct MMA_Op<MMA_Traits<MMA_Op_Arg, Args...>> {
  using type = MMA_Op_Arg;
};

//
// Generic mma_unpack for any MMA_Traits
//

template <class AnyMMATraits,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr
void
mma_unpack(AnyMMATraits        const& traits,
           Tensor<TD, DLayout>      & D,
           Tensor<TA, ALayout> const& A,
           Tensor<TB, BLayout> const& B,
           Tensor<TC, CLayout> const& C)
{
  static_assert(is_rmem<TD>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TA>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TB>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TC>::value, "Expected registers in MMA_Atom::call");

  // Register value types from the MMA_Operation register arrays
  using MMA_Op   = typename MMA_Op<AnyMMATraits>::type;
  using RegTypeD = typename remove_extent<typename MMA_Op::DRegisters>::type;
  using RegTypeA = typename remove_extent<typename MMA_Op::ARegisters>::type;
  using RegTypeB = typename remove_extent<typename MMA_Op::BRegisters>::type;
  using RegTypeC = typename remove_extent<typename MMA_Op::CRegisters>::type;

  Tensor rA = recast<RegTypeA>(A);
  Tensor rB = recast<RegTypeB>(B);
  Tensor rD = recast<RegTypeD>(D);
  Tensor rC = recast<RegTypeC>(C);

  constexpr int RegNumD = extent<typename MMA_Op::DRegisters>::value;
  constexpr int RegNumA = extent<typename MMA_Op::ARegisters>::value;
  constexpr int RegNumB = extent<typename MMA_Op::BRegisters>::value;
  constexpr int RegNumC = extent<typename MMA_Op::CRegisters>::value;

  CUTE_STATIC_ASSERT_V(size(rA) == Int<RegNumA>{});
  CUTE_STATIC_ASSERT_V(size(rB) == Int<RegNumB>{});
  CUTE_STATIC_ASSERT_V(size(rD) == Int<RegNumD>{});
  CUTE_STATIC_ASSERT_V(size(rC) == Int<RegNumC>{});

  detail::explode(MMA_Op::fma,
                  rD, make_int_sequence<RegNumD>{},
                  rA, make_int_sequence<RegNumA>{},
                  rB, make_int_sequence<RegNumB>{},
                  rC, make_int_sequence<RegNumC>{});
}

// Accept mutable temporaries
template <class AnyMMATraits,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr
void
mma_unpack(AnyMMATraits        const& traits,
           Tensor<TD, DLayout>     && D,
           Tensor<TA, ALayout> const& A,
           Tensor<TB, BLayout> const& B,
           Tensor<TC, CLayout> const& C)
{
  mma_unpack(traits, D, A, B, C);
}

namespace detail {

template <class X, class = void>
struct FrgTypeA_or_Default { using type = typename X::ValTypeA; };
template <class X>
struct FrgTypeA_or_Default<X,void_t<typename X::FrgTypeA>> { using type = typename X::FrgTypeA; };

template <class X, class = void>
struct FrgTypeB_or_Default { using type = typename X::ValTypeB; };
template <class X>
struct FrgTypeB_or_Default<X,void_t<typename X::FrgTypeB>> { using type = typename X::FrgTypeB; };

template <class X, class = void>
struct FrgTypeC_or_Default { using type = typename X::ValTypeC; };
template <class X>
struct FrgTypeC_or_Default<X,void_t<typename X::FrgTypeC>> { using type = typename X::FrgTypeC; };

} // end namespace detail

} // namespace cute
