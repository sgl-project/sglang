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

#include <cute/config.hpp>    // CUTE_HOST_DEVICE

#include <cutlass/complex.h>  // cutlass::complexm, cutlass::real, cutlass::imag, cutlass::is_complex

namespace cute
{

using cutlass::complex;
using cutlass::is_complex;
using cutlass::RealType;
using cutlass::real;
using cutlass::imag;
using cutlass::conj;

template <class T>
static constexpr auto is_complex_v = is_complex<T>::value;

/// Fused multiply-add for complex numbers
template <class D, class A, class B, class C>
CUTE_HOST_DEVICE constexpr
void
fma(complex<D>      & d,
    complex<A> const& a,
    complex<B> const& b,
    complex<C> const& c)
{
  fma(d.real(),  a.real(), b.real(), c.real());
  fma(d.imag(),  a.real(), b.imag(), c.imag());
  fma(d.real(), -a.imag(), b.imag(), d.real());
  fma(d.imag(),  a.imag(), b.real(), d.imag());
}

/// Fused multiply-add for triplets
template <class A, class B, class C>
CUTE_HOST_DEVICE constexpr
void
fma(complex<A> const& a,
    complex<B> const& b,
    complex<C>      & c)
{
  return fma(c, a, b, c);
}

} // end namespace cute
