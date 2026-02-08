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

/* \file 
  \brief Boost-style constant definitions for floating-point types.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cutlass/complex.h"

///////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace constants {

///////////////////////////////////////////////////////////////////////////////////

//
// Primary templates
//

/// Returns 1, the multiplicative identity element
template <typename T> CUTLASS_HOST_DEVICE T one();

/// Returns 0, the additive identity element
template <typename T> CUTLASS_HOST_DEVICE T zero();

/// Returns 2
template <typename T> CUTLASS_HOST_DEVICE T two();

/// Returns pi, approximately 3.141
template <typename T> CUTLASS_HOST_DEVICE T pi();

/// Returns 2 * pi
template <typename T> CUTLASS_HOST_DEVICE T two_pi();

/// Returns pi / 2
template <typename T> CUTLASS_HOST_DEVICE T half_pi();

/// Returns sqrt(pi)
template <typename T> CUTLASS_HOST_DEVICE T root_pi();

/// Returns sqrt(pi / 2)
template <typename T> CUTLASS_HOST_DEVICE T root_half_pi();

/// Returns sqrt(2 * pi)
template <typename T> CUTLASS_HOST_DEVICE T root_two_pi();

/// Returns sqrt(ln(4))
template <typename T> CUTLASS_HOST_DEVICE T root_ln_four();

/// Returns e, approximately 2.718...
template <typename T> CUTLASS_HOST_DEVICE T e();

/// Returns (1/2)
template <typename T> CUTLASS_HOST_DEVICE T half();

/// Returns sqrt(2), approximately 1.414...
template <typename T> CUTLASS_HOST_DEVICE T root_two();

/// Returns sqrt(2)/2, approximately 0.707...
template <typename T> CUTLASS_HOST_DEVICE T half_root_two();

/// Returns ln(2), approximately 0.693...
template <typename T> CUTLASS_HOST_DEVICE T ln_two();

/// Returns ln(ln(2)), approximately -0.3665...
template <typename T> CUTLASS_HOST_DEVICE T ln_ln_two();

/// Returns 1/3, approximately 0.333...
template <typename T> CUTLASS_HOST_DEVICE T third();

/// Returns 2/3, approximately 0.666...
template <typename T> CUTLASS_HOST_DEVICE T twothirds();

/// Returns pi - 3, approximately 0.1416...
template <typename T> CUTLASS_HOST_DEVICE T pi_minus_three();

/// Returns 4 - pi, approximately 0.858...
template <typename T> CUTLASS_HOST_DEVICE T four_minus_pi();


/////////////////////////////////////////////////////////////////////////////////////

// Specialization for double

/// Returns 1, the multiplicative identity element  (specialization for double)
template <> CUTLASS_HOST_DEVICE double one<double>() {
  uint64_t bits = 0x3ff0000000000000ull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns 1, the multiplicative identity element  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> one< complex<double> >() {
  return complex<double>(one<double>(), double());
}

/// Returns 0, the additive identity element  (specialization for double)
template <> CUTLASS_HOST_DEVICE double zero<double>() {
  uint64_t bits = 0x0ull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns 0, the additive identity element  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> zero< complex<double> >() {
  return complex<double>(zero<double>(), double());
}

/// Returns 2  (specialization for double)
template <> CUTLASS_HOST_DEVICE double two<double>() {
  uint64_t bits = 0x4000000000000000ull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns 2  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> two< complex<double> >() {
  return complex<double>(two<double>(), double());
}

/// Returns pi, approximately 3.141  (specialization for double)
template <> CUTLASS_HOST_DEVICE double pi<double>() {
  uint64_t bits = 0x400921fb54442d18ull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns pi, approximately 3.141  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> pi< complex<double> >() {
  return complex<double>(pi<double>(), double());
}

/// Returns 2 * pi  (specialization for double)
template <> CUTLASS_HOST_DEVICE double two_pi<double>() {
  uint64_t bits = 0x401921fb54442d18ull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns 2 * pi  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> two_pi< complex<double> >() {
  return complex<double>(two_pi<double>(), double());
}

/// Returns pi / 2  (specialization for double)
template <> CUTLASS_HOST_DEVICE double half_pi<double>() {
  uint64_t bits = 0x3ff921fb54442d18ull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns pi / 2  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> half_pi< complex<double> >() {
  return complex<double>(half_pi<double>(), double());
}

/// Returns sqrt(pi)  (specialization for double)
template <> CUTLASS_HOST_DEVICE double root_pi<double>() {
  uint64_t bits = 0x3ffc5bf891b4ef6aull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns sqrt(pi)  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> root_pi< complex<double> >() {
  return complex<double>(root_pi<double>(), double());
}

/// Returns sqrt(pi / 2)  (specialization for double)
template <> CUTLASS_HOST_DEVICE double root_half_pi<double>() {
  uint64_t bits = 0x3ff40d931ff62705ull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns sqrt(pi / 2)  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> root_half_pi< complex<double> >() {
  return complex<double>(root_half_pi<double>(), double());
}

/// Returns sqrt(2 * pi)  (specialization for double)
template <> CUTLASS_HOST_DEVICE double root_two_pi<double>() {
  uint64_t bits = 0x40040d931ff62705ull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns sqrt(2 * pi)  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> root_two_pi< complex<double> >() {
  return complex<double>(root_two_pi<double>(), double());
}

/// Returns sqrt(ln(4))  (specialization for double)
template <> CUTLASS_HOST_DEVICE double root_ln_four<double>() {
  uint64_t bits = 0x3ff2d6abe44afc43ull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns sqrt(ln(4))  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> root_ln_four< complex<double> >() {
  return complex<double>(root_ln_four<double>(), double());
}

/// Returns e, approximately 2.718...  (specialization for double)
template <> CUTLASS_HOST_DEVICE double e<double>() {
  uint64_t bits = 0x4005bf0a8b145769ull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns e, approximately 2.718...  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> e< complex<double> >() {
  return complex<double>(e<double>(), double());
}

/// Returns (1/2)  (specialization for double)
template <> CUTLASS_HOST_DEVICE double half<double>() {
  uint64_t bits = 0x3fe0000000000000ull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns (1/2)  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> half< complex<double> >() {
  return complex<double>(half<double>(), double());
}

/// Returns sqrt(2), approximately 1.414...  (specialization for double)
template <> CUTLASS_HOST_DEVICE double root_two<double>() {
  uint64_t bits = 0x3ff6a09e667f3bcdull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns sqrt(2), approximately 1.414...  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> root_two< complex<double> >() {
  return complex<double>(root_two<double>(), double());
}

/// Returns sqrt(2)/2, approximately 0.707...  (specialization for double)
template <> CUTLASS_HOST_DEVICE double half_root_two<double>() {
  uint64_t bits = 0x3fe6a09e667f3bcdull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns sqrt(2)/2, approximately 0.707...  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> half_root_two< complex<double> >() {
  return complex<double>(half_root_two<double>(), double());
}

/// Returns ln(2), approximately 0.693...  (specialization for double)
template <> CUTLASS_HOST_DEVICE double ln_two<double>() {
  uint64_t bits = 0x3fe62e42fefa39efull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns ln(2), approximately 0.693...  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> ln_two< complex<double> >() {
  return complex<double>(ln_two<double>(), double());
}

/// Returns ln(ln(2)), approximately -0.3665...  (specialization for double)
template <> CUTLASS_HOST_DEVICE double ln_ln_two<double>() {
  uint64_t bits = 0xbfd774f29bdd6b9full;
  return reinterpret_cast<double const &>(bits);
}

/// Returns ln(ln(2)), approximately -0.3665...  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> ln_ln_two< complex<double> >() {
  return complex<double>(ln_ln_two<double>(), double());
}

/// Returns 1/3, approximately 0.333...  (specialization for double)
template <> CUTLASS_HOST_DEVICE double third<double>() {
  uint64_t bits = 0x3fd5555555555555ull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns 1/3, approximately 0.333...  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> third< complex<double> >() {
  return complex<double>(third<double>(), double());
}

/// Returns 2/3, approximately 0.666...  (specialization for double)
template <> CUTLASS_HOST_DEVICE double twothirds<double>() {
  uint64_t bits = 0x3fe5555555555555ull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns 2/3, approximately 0.666...  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> twothirds< complex<double> >() {
  return complex<double>(twothirds<double>(), double());
}

/// Returns pi - 3, approximately 0.1416...  (specialization for double)
template <> CUTLASS_HOST_DEVICE double pi_minus_three<double>() {
  uint64_t bits = 0x3fc21fb54442d180ull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns pi - 3, approximately 0.1416...  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> pi_minus_three< complex<double> >() {
  return complex<double>(pi_minus_three<double>(), double());
}

/// Returns 4 - pi, approximately 0.858...  (specialization for double)
template <> CUTLASS_HOST_DEVICE double four_minus_pi<double>() {
  uint64_t bits = 0x3feb7812aeef4ba0ull;
  return reinterpret_cast<double const &>(bits);
}

/// Returns 4 - pi, approximately 0.858...  (specialization for complex<double>)
template <> CUTLASS_HOST_DEVICE complex<double> four_minus_pi< complex<double> >() {
  return complex<double>(four_minus_pi<double>(), double());
}

/////////////////////////////////////////////////////////////////////////////////////

// Specialization for float

/// Returns 1, the multiplicative identity element  (specialization for float)
template <> CUTLASS_HOST_DEVICE float one<float>() {
  uint32_t bits = 0x3f800000u;
  return reinterpret_cast<float const &>(bits);
}

/// Returns 1, the multiplicative identity element  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> one< complex<float> >() {
  return complex<float>(one<float>(), float());
}

/// Returns 0, the additive identity element  (specialization for float)
template <> CUTLASS_HOST_DEVICE float zero<float>() {
  uint32_t bits = 0x0u;
  return reinterpret_cast<float const &>(bits);
}

/// Returns 0, the additive identity element  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> zero< complex<float> >() {
  return complex<float>(zero<float>(), float());
}

/// Returns 2  (specialization for float)
template <> CUTLASS_HOST_DEVICE float two<float>() {
  uint32_t bits = 0x40000000u;
  return reinterpret_cast<float const &>(bits);
}

/// Returns 2  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> two< complex<float> >() {
  return complex<float>(two<float>(), float());
}

/// Returns pi, approximately 3.141  (specialization for float)
template <> CUTLASS_HOST_DEVICE float pi<float>() {
  uint32_t bits = 0x40490fdbu;
  return reinterpret_cast<float const &>(bits);
}

/// Returns pi, approximately 3.141  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> pi< complex<float> >() {
  return complex<float>(pi<float>(), float());
}

/// Returns 2 * pi  (specialization for float)
template <> CUTLASS_HOST_DEVICE float two_pi<float>() {
  uint32_t bits = 0x40c90fdbu;
  return reinterpret_cast<float const &>(bits);
}

/// Returns 2 * pi  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> two_pi< complex<float> >() {
  return complex<float>(two_pi<float>(), float());
}

/// Returns pi / 2  (specialization for float)
template <> CUTLASS_HOST_DEVICE float half_pi<float>() {
  uint32_t bits = 0x3fc90fdbu;
  return reinterpret_cast<float const &>(bits);
}

/// Returns pi / 2  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> half_pi< complex<float> >() {
  return complex<float>(half_pi<float>(), float());
}

/// Returns sqrt(pi)  (specialization for float)
template <> CUTLASS_HOST_DEVICE float root_pi<float>() {
  uint32_t bits = 0x3fe2dfc5u;
  return reinterpret_cast<float const &>(bits);
}

/// Returns sqrt(pi)  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> root_pi< complex<float> >() {
  return complex<float>(root_pi<float>(), float());
}

/// Returns sqrt(pi / 2)  (specialization for float)
template <> CUTLASS_HOST_DEVICE float root_half_pi<float>() {
  uint32_t bits = 0x3fa06c99u;
  return reinterpret_cast<float const &>(bits);
}

/// Returns sqrt(pi / 2)  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> root_half_pi< complex<float> >() {
  return complex<float>(root_half_pi<float>(), float());
}

/// Returns sqrt(2 * pi)  (specialization for float)
template <> CUTLASS_HOST_DEVICE float root_two_pi<float>() {
  uint32_t bits = 0x40206c99u;
  return reinterpret_cast<float const &>(bits);
}

/// Returns sqrt(2 * pi)  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> root_two_pi< complex<float> >() {
  return complex<float>(root_two_pi<float>(), float());
}

/// Returns sqrt(ln(4))  (specialization for float)
template <> CUTLASS_HOST_DEVICE float root_ln_four<float>() {
  uint32_t bits = 0x3f96b55fu;
  return reinterpret_cast<float const &>(bits);
}

/// Returns sqrt(ln(4))  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> root_ln_four< complex<float> >() {
  return complex<float>(root_ln_four<float>(), float());
}

/// Returns e, approximately 2.718...  (specialization for float)
template <> CUTLASS_HOST_DEVICE float e<float>() {
  uint32_t bits = 0x402df854u;
  return reinterpret_cast<float const &>(bits);
}

/// Returns e, approximately 2.718...  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> e< complex<float> >() {
  return complex<float>(e<float>(), float());
}

/// Returns (1/2)  (specialization for float)
template <> CUTLASS_HOST_DEVICE float half<float>() {
  uint32_t bits = 0x3f000000u;
  return reinterpret_cast<float const &>(bits);
}

/// Returns (1/2)  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> half< complex<float> >() {
  return complex<float>(half<float>(), float());
}

/// Returns sqrt(2), approximately 1.414...  (specialization for float)
template <> CUTLASS_HOST_DEVICE float root_two<float>() {
  uint32_t bits = 0x3fb504f3u;
  return reinterpret_cast<float const &>(bits);
}

/// Returns sqrt(2), approximately 1.414...  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> root_two< complex<float> >() {
  return complex<float>(root_two<float>(), float());
}

/// Returns sqrt(2)/2, approximately 0.707...  (specialization for float)
template <> CUTLASS_HOST_DEVICE float half_root_two<float>() {
  uint32_t bits = 0x3f3504f3u;
  return reinterpret_cast<float const &>(bits);
}

/// Returns sqrt(2)/2, approximately 0.707...  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> half_root_two< complex<float> >() {
  return complex<float>(half_root_two<float>(), float());
}

/// Returns ln(2), approximately 0.693...  (specialization for float)
template <> CUTLASS_HOST_DEVICE float ln_two<float>() {
  uint32_t bits = 0x3f317218u;
  return reinterpret_cast<float const &>(bits);
}

/// Returns ln(2), approximately 0.693...  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> ln_two< complex<float> >() {
  return complex<float>(ln_two<float>(), float());
}

/// Returns ln(ln(2)), approximately -0.3665...  (specialization for float)
template <> CUTLASS_HOST_DEVICE float ln_ln_two<float>() {
  uint32_t bits = 0xbebba795u;
  return reinterpret_cast<float const &>(bits);
}

/// Returns ln(ln(2)), approximately -0.3665...  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> ln_ln_two< complex<float> >() {
  return complex<float>(ln_ln_two<float>(), float());
}

/// Returns 1/3, approximately 0.333...  (specialization for float)
template <> CUTLASS_HOST_DEVICE float third<float>() {
  uint32_t bits = 0x3eaaaaabu;
  return reinterpret_cast<float const &>(bits);
}

/// Returns 1/3, approximately 0.333...  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> third< complex<float> >() {
  return complex<float>(third<float>(), float());
}

/// Returns 2/3, approximately 0.666...  (specialization for float)
template <> CUTLASS_HOST_DEVICE float twothirds<float>() {
  uint32_t bits = 0x3f2aaaabu;
  return reinterpret_cast<float const &>(bits);
}

/// Returns 2/3, approximately 0.666...  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> twothirds< complex<float> >() {
  return complex<float>(twothirds<float>(), float());
}

/// Returns pi - 3, approximately 0.1416...  (specialization for float)
template <> CUTLASS_HOST_DEVICE float pi_minus_three<float>() {
  uint32_t bits = 0x3e10fdaau;
  return reinterpret_cast<float const &>(bits);
}

/// Returns pi - 3, approximately 0.1416...  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> pi_minus_three< complex<float> >() {
  return complex<float>(pi_minus_three<float>(), float());
}

/// Returns 4 - pi, approximately 0.858...  (specialization for float)
template <> CUTLASS_HOST_DEVICE float four_minus_pi<float>() {
  uint32_t bits = 0x3f5bc095u;
  return reinterpret_cast<float const &>(bits);
}

/// Returns 4 - pi, approximately 0.858...  (specialization for complex<float>)
template <> CUTLASS_HOST_DEVICE complex<float> four_minus_pi< complex<float> >() {
  return complex<float>(four_minus_pi<float>(), float());
}

/////////////////////////////////////////////////////////////////////////////////////

// Specialization for tfloat32_t

/// Returns 1, the multiplicative identity element  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t one<tfloat32_t>() {
  uint32_t bits = 0x3f801000u;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns 1, the multiplicative identity element  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> one< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(one<tfloat32_t>(), tfloat32_t());
}

/// Returns 0, the additive identity element  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t zero<tfloat32_t>() {
  uint32_t bits = 0x1000u;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns 0, the additive identity element  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> zero< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(zero<tfloat32_t>(), tfloat32_t());
}

/// Returns 2  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t two<tfloat32_t>() {
  uint32_t bits = 0x40001000u;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns 2  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> two< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(two<tfloat32_t>(), tfloat32_t());
}

/// Returns pi, approximately 3.141  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t pi<tfloat32_t>() {
  uint32_t bits = 0x40491fdbu;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns pi, approximately 3.141  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> pi< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(pi<tfloat32_t>(), tfloat32_t());
}

/// Returns 2 * pi  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t two_pi<tfloat32_t>() {
  uint32_t bits = 0x40c91fdbu;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns 2 * pi  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> two_pi< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(two_pi<tfloat32_t>(), tfloat32_t());
}

/// Returns pi / 2  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t half_pi<tfloat32_t>() {
  uint32_t bits = 0x3fc91fdbu;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns pi / 2  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> half_pi< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(half_pi<tfloat32_t>(), tfloat32_t());
}

/// Returns sqrt(pi)  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t root_pi<tfloat32_t>() {
  uint32_t bits = 0x3fe2efc5u;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns sqrt(pi)  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> root_pi< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(root_pi<tfloat32_t>(), tfloat32_t());
}

/// Returns sqrt(pi / 2)  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t root_half_pi<tfloat32_t>() {
  uint32_t bits = 0x3fa07c99u;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns sqrt(pi / 2)  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> root_half_pi< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(root_half_pi<tfloat32_t>(), tfloat32_t());
}

/// Returns sqrt(2 * pi)  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t root_two_pi<tfloat32_t>() {
  uint32_t bits = 0x40207c99u;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns sqrt(2 * pi)  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> root_two_pi< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(root_two_pi<tfloat32_t>(), tfloat32_t());
}

/// Returns sqrt(ln(4))  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t root_ln_four<tfloat32_t>() {
  uint32_t bits = 0x3f96c55fu;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns sqrt(ln(4))  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> root_ln_four< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(root_ln_four<tfloat32_t>(), tfloat32_t());
}

/// Returns e, approximately 2.718...  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t e<tfloat32_t>() {
  uint32_t bits = 0x402e0854u;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns e, approximately 2.718...  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> e< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(e<tfloat32_t>(), tfloat32_t());
}

/// Returns (1/2)  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t half<tfloat32_t>() {
  uint32_t bits = 0x3f001000u;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns (1/2)  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> half< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(half<tfloat32_t>(), tfloat32_t());
}

/// Returns sqrt(2), approximately 1.414...  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t root_two<tfloat32_t>() {
  uint32_t bits = 0x3fb514f3u;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns sqrt(2), approximately 1.414...  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> root_two< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(root_two<tfloat32_t>(), tfloat32_t());
}

/// Returns sqrt(2)/2, approximately 0.707...  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t half_root_two<tfloat32_t>() {
  uint32_t bits = 0x3f3514f3u;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns sqrt(2)/2, approximately 0.707...  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> half_root_two< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(half_root_two<tfloat32_t>(), tfloat32_t());
}

/// Returns ln(2), approximately 0.693...  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t ln_two<tfloat32_t>() {
  uint32_t bits = 0x3f318218u;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns ln(2), approximately 0.693...  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> ln_two< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(ln_two<tfloat32_t>(), tfloat32_t());
}

/// Returns ln(ln(2)), approximately -0.3665...  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t ln_ln_two<tfloat32_t>() {
  uint32_t bits = 0xbebbb795u;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns ln(ln(2)), approximately -0.3665...  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> ln_ln_two< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(ln_ln_two<tfloat32_t>(), tfloat32_t());
}

/// Returns 1/3, approximately 0.333...  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t third<tfloat32_t>() {
  uint32_t bits = 0x3eaabaabu;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns 1/3, approximately 0.333...  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> third< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(third<tfloat32_t>(), tfloat32_t());
}

/// Returns 2/3, approximately 0.666...  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t twothirds<tfloat32_t>() {
  uint32_t bits = 0x3f2abaabu;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns 2/3, approximately 0.666...  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> twothirds< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(twothirds<tfloat32_t>(), tfloat32_t());
}

/// Returns pi - 3, approximately 0.1416...  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t pi_minus_three<tfloat32_t>() {
  uint32_t bits = 0x3e110daau;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns pi - 3, approximately 0.1416...  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> pi_minus_three< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(pi_minus_three<tfloat32_t>(), tfloat32_t());
}

/// Returns 4 - pi, approximately 0.858...  (specialization for tfloat32_t)
template <> CUTLASS_HOST_DEVICE tfloat32_t four_minus_pi<tfloat32_t>() {
  uint32_t bits = 0x3f5bd095u;
  return reinterpret_cast<tfloat32_t const &>(bits);
}

/// Returns 4 - pi, approximately 0.858...  (specialization for complex<tfloat32_t>)
template <> CUTLASS_HOST_DEVICE complex<tfloat32_t> four_minus_pi< complex<tfloat32_t> >() {
  return complex<tfloat32_t>(four_minus_pi<tfloat32_t>(), tfloat32_t());
}

/////////////////////////////////////////////////////////////////////////////////////

// Specialization for half_t

/// Returns 1, the multiplicative identity element  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t one<half_t>() {
  uint16_t bits = 0x3c00u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns 1, the multiplicative identity element  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> one< complex<half_t> >() {
  return complex<half_t>(one<half_t>(), half_t());
}

/// Returns 0, the additive identity element  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t zero<half_t>() {
  uint16_t bits = 0x0u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns 0, the additive identity element  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> zero< complex<half_t> >() {
  return complex<half_t>(zero<half_t>(), half_t());
}

/// Returns 2  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t two<half_t>() {
  uint16_t bits = 0x4000u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns 2  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> two< complex<half_t> >() {
  return complex<half_t>(two<half_t>(), half_t());
}

/// Returns pi, approximately 3.141  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t pi<half_t>() {
  uint16_t bits = 0x4248u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns pi, approximately 3.141  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> pi< complex<half_t> >() {
  return complex<half_t>(pi<half_t>(), half_t());
}

/// Returns 2 * pi  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t two_pi<half_t>() {
  uint16_t bits = 0x4648u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns 2 * pi  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> two_pi< complex<half_t> >() {
  return complex<half_t>(two_pi<half_t>(), half_t());
}

/// Returns pi / 2  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t half_pi<half_t>() {
  uint16_t bits = 0x3e48u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns pi / 2  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> half_pi< complex<half_t> >() {
  return complex<half_t>(half_pi<half_t>(), half_t());
}

/// Returns sqrt(pi)  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t root_pi<half_t>() {
  uint16_t bits = 0x3f17u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns sqrt(pi)  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> root_pi< complex<half_t> >() {
  return complex<half_t>(root_pi<half_t>(), half_t());
}

/// Returns sqrt(pi / 2)  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t root_half_pi<half_t>() {
  uint16_t bits = 0x3d03u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns sqrt(pi / 2)  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> root_half_pi< complex<half_t> >() {
  return complex<half_t>(root_half_pi<half_t>(), half_t());
}

/// Returns sqrt(2 * pi)  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t root_two_pi<half_t>() {
  uint16_t bits = 0x4103u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns sqrt(2 * pi)  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> root_two_pi< complex<half_t> >() {
  return complex<half_t>(root_two_pi<half_t>(), half_t());
}

/// Returns sqrt(ln(4))  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t root_ln_four<half_t>() {
  uint16_t bits = 0x3cb6u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns sqrt(ln(4))  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> root_ln_four< complex<half_t> >() {
  return complex<half_t>(root_ln_four<half_t>(), half_t());
}

/// Returns e, approximately 2.718...  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t e<half_t>() {
  uint16_t bits = 0x4170u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns e, approximately 2.718...  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> e< complex<half_t> >() {
  return complex<half_t>(e<half_t>(), half_t());
}

/// Returns (1/2)  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t half<half_t>() {
  uint16_t bits = 0x3800u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns (1/2)  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> half< complex<half_t> >() {
  return complex<half_t>(half<half_t>(), half_t());
}

/// Returns sqrt(2), approximately 1.414...  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t root_two<half_t>() {
  uint16_t bits = 0x3da8u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns sqrt(2), approximately 1.414...  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> root_two< complex<half_t> >() {
  return complex<half_t>(root_two<half_t>(), half_t());
}

/// Returns sqrt(2)/2, approximately 0.707...  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t half_root_two<half_t>() {
  uint16_t bits = 0x39a8u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns sqrt(2)/2, approximately 0.707...  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> half_root_two< complex<half_t> >() {
  return complex<half_t>(half_root_two<half_t>(), half_t());
}

/// Returns ln(2), approximately 0.693...  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t ln_two<half_t>() {
  uint16_t bits = 0x398cu;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns ln(2), approximately 0.693...  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> ln_two< complex<half_t> >() {
  return complex<half_t>(ln_two<half_t>(), half_t());
}

/// Returns ln(ln(2)), approximately -0.3665...  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t ln_ln_two<half_t>() {
  uint16_t bits = 0xb5ddu;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns ln(ln(2)), approximately -0.3665...  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> ln_ln_two< complex<half_t> >() {
  return complex<half_t>(ln_ln_two<half_t>(), half_t());
}

/// Returns 1/3, approximately 0.333...  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t third<half_t>() {
  uint16_t bits = 0x3555u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns 1/3, approximately 0.333...  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> third< complex<half_t> >() {
  return complex<half_t>(third<half_t>(), half_t());
}

/// Returns 2/3, approximately 0.666...  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t twothirds<half_t>() {
  uint16_t bits = 0x3955u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns 2/3, approximately 0.666...  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> twothirds< complex<half_t> >() {
  return complex<half_t>(twothirds<half_t>(), half_t());
}

/// Returns pi - 3, approximately 0.1416...  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t pi_minus_three<half_t>() {
  uint16_t bits = 0x3088u;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns pi - 3, approximately 0.1416...  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> pi_minus_three< complex<half_t> >() {
  return complex<half_t>(pi_minus_three<half_t>(), half_t());
}

/// Returns 4 - pi, approximately 0.858...  (specialization for half_t)
template <> CUTLASS_HOST_DEVICE half_t four_minus_pi<half_t>() {
  uint16_t bits = 0x3adeu;
  return reinterpret_cast<half_t const &>(bits);
}

/// Returns 4 - pi, approximately 0.858...  (specialization for complex<half_t>)
template <> CUTLASS_HOST_DEVICE complex<half_t> four_minus_pi< complex<half_t> >() {
  return complex<half_t>(four_minus_pi<half_t>(), half_t());
}

/////////////////////////////////////////////////////////////////////////////////////

// Specialization for bfloat16_t

/// Returns 1, the multiplicative identity element  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t one<bfloat16_t>() {
  uint16_t bits = 0x3f80u;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns 1, the multiplicative identity element  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> one< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(one<bfloat16_t>(), bfloat16_t());
}

/// Returns 0, the additive identity element  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t zero<bfloat16_t>() {
  uint16_t bits = 0x0u;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns 0, the additive identity element  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> zero< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(zero<bfloat16_t>(), bfloat16_t());
}

/// Returns 2  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t two<bfloat16_t>() {
  uint16_t bits = 0x4000u;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns 2  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> two< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(two<bfloat16_t>(), bfloat16_t());
}

/// Returns pi, approximately 3.141  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t pi<bfloat16_t>() {
  uint16_t bits = 0x4049u;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns pi, approximately 3.141  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> pi< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(pi<bfloat16_t>(), bfloat16_t());
}

/// Returns 2 * pi  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t two_pi<bfloat16_t>() {
  uint16_t bits = 0x40c9u;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns 2 * pi  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> two_pi< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(two_pi<bfloat16_t>(), bfloat16_t());
}

/// Returns pi / 2  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t half_pi<bfloat16_t>() {
  uint16_t bits = 0x3fc9u;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns pi / 2  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> half_pi< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(half_pi<bfloat16_t>(), bfloat16_t());
}

/// Returns sqrt(pi)  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t root_pi<bfloat16_t>() {
  uint16_t bits = 0x3fe3u;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns sqrt(pi)  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> root_pi< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(root_pi<bfloat16_t>(), bfloat16_t());
}

/// Returns sqrt(pi / 2)  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t root_half_pi<bfloat16_t>() {
  uint16_t bits = 0x3fa0u;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns sqrt(pi / 2)  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> root_half_pi< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(root_half_pi<bfloat16_t>(), bfloat16_t());
}

/// Returns sqrt(2 * pi)  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t root_two_pi<bfloat16_t>() {
  uint16_t bits = 0x4020u;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns sqrt(2 * pi)  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> root_two_pi< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(root_two_pi<bfloat16_t>(), bfloat16_t());
}

/// Returns sqrt(ln(4))  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t root_ln_four<bfloat16_t>() {
  uint16_t bits = 0x3f97u;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns sqrt(ln(4))  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> root_ln_four< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(root_ln_four<bfloat16_t>(), bfloat16_t());
}

/// Returns e, approximately 2.718...  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t e<bfloat16_t>() {
  uint16_t bits = 0x402eu;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns e, approximately 2.718...  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> e< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(e<bfloat16_t>(), bfloat16_t());
}

/// Returns (1/2)  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t half<bfloat16_t>() {
  uint16_t bits = 0x3f00u;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns (1/2)  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> half< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(half<bfloat16_t>(), bfloat16_t());
}

/// Returns sqrt(2), approximately 1.414...  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t root_two<bfloat16_t>() {
  uint16_t bits = 0x3fb5u;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns sqrt(2), approximately 1.414...  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> root_two< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(root_two<bfloat16_t>(), bfloat16_t());
}

/// Returns sqrt(2)/2, approximately 0.707...  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t half_root_two<bfloat16_t>() {
  uint16_t bits = 0x3f35u;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns sqrt(2)/2, approximately 0.707...  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> half_root_two< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(half_root_two<bfloat16_t>(), bfloat16_t());
}

/// Returns ln(2), approximately 0.693...  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t ln_two<bfloat16_t>() {
  uint16_t bits = 0x3f31u;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns ln(2), approximately 0.693...  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> ln_two< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(ln_two<bfloat16_t>(), bfloat16_t());
}

/// Returns ln(ln(2)), approximately -0.3665...  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t ln_ln_two<bfloat16_t>() {
  uint16_t bits = 0xbebcu;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns ln(ln(2)), approximately -0.3665...  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> ln_ln_two< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(ln_ln_two<bfloat16_t>(), bfloat16_t());
}

/// Returns 1/3, approximately 0.333...  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t third<bfloat16_t>() {
  uint16_t bits = 0x3eabu;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns 1/3, approximately 0.333...  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> third< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(third<bfloat16_t>(), bfloat16_t());
}

/// Returns 2/3, approximately 0.666...  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t twothirds<bfloat16_t>() {
  uint16_t bits = 0x3f2bu;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns 2/3, approximately 0.666...  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> twothirds< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(twothirds<bfloat16_t>(), bfloat16_t());
}

/// Returns pi - 3, approximately 0.1416...  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t pi_minus_three<bfloat16_t>() {
  uint16_t bits = 0x3e11u;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns pi - 3, approximately 0.1416...  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> pi_minus_three< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(pi_minus_three<bfloat16_t>(), bfloat16_t());
}

/// Returns 4 - pi, approximately 0.858...  (specialization for bfloat16_t)
template <> CUTLASS_HOST_DEVICE bfloat16_t four_minus_pi<bfloat16_t>() {
  uint16_t bits = 0x3f5cu;
  return reinterpret_cast<bfloat16_t const &>(bits);
}

/// Returns 4 - pi, approximately 0.858...  (specialization for complex<bfloat16_t>)
template <> CUTLASS_HOST_DEVICE complex<bfloat16_t> four_minus_pi< complex<bfloat16_t> >() {
  return complex<bfloat16_t>(four_minus_pi<bfloat16_t>(), bfloat16_t());
}
///////////////////////////////////////////////////////////////////////////////////

} // namespace constants
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////
