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
/*! \file
    \brief Unit tests for thread-level GEMM
*/

#include "../../common/cutlass_unit_test.h"

#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace epilogue {
namespace thread {

using FunctorPlanarComplexF32F32 = cutlass::epilogue::thread::LinearCombinationPlanarComplex<
  float, 
  4,
  float, 
  float>;

__global__ void epilogue_thread_functor_planar_complex_f32_f32(
  float *output_ptr, 
  float const *accum_ptr, 
  float const *source_ptr,
  typename FunctorPlanarComplexF32F32::Params params) {

  FunctorPlanarComplexF32F32 linear_combination_op(params);

  auto accum = *reinterpret_cast<cutlass::ArrayPlanarComplex<float , 4> const *>(accum_ptr);
  auto source = *reinterpret_cast<cutlass::ArrayPlanarComplex<float, 4> const *>(source_ptr);

  *reinterpret_cast<cutlass::ArrayPlanarComplex<float, 4>*>(output_ptr) = linear_combination_op(accum, source);
}

}
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Epilogue_thread_linear_combination_planar_complex, f32) {

  using Element = float;
  using ElementOutput = float;
  int const kCount = 4;

  using Functor = cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    ElementOutput, 
    kCount,
    Element, 
    Element>;

  cutlass::complex<Element> alpha(Element(2), Element(1));
  cutlass::complex<Element> beta(Element(1), Element(-1));

  typename Functor::Params params(alpha, beta);

  Functor linear_combination_op(params);

  cutlass::ArrayPlanarComplex<ElementOutput, kCount> source;
  cutlass::ArrayPlanarComplex<Element, kCount> accum;

  // Define arbitrary inputs
  for (int i = 0; i < kCount; ++i) {
    accum.real[i] = Element(i * 2);
    accum.imag[i] = Element((i * 3 % 6) - 3);
    source.real[i] = ElementOutput((i * 7 % 9) - 4);
    source.imag[i] = ElementOutput(((i * 5 + 2) % 9) - 4);
  }

  cutlass::ArrayPlanarComplex<ElementOutput, kCount> destination = linear_combination_op(accum, source);

  // Verify each result
  for (int i = 0; i < kCount; ++i) {
    
    cutlass::complex<Element> expected = alpha * cutlass::complex<Element>(accum.real[i], accum.imag[i]) + 
      beta * cutlass::complex<Element>(Element(source.real[i]), Element(source.imag[i]));

    cutlass::complex<ElementOutput> got(destination.real[i], destination.imag[i]);
    
    EXPECT_TRUE(ElementOutput(expected.real()) == got.real());
    EXPECT_TRUE(ElementOutput(expected.imag()) == got.imag());
    EXPECT_TRUE(expected.real() != Element(0) || expected.imag() != Element(0));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace epilogue {
namespace thread {

using FunctorPlanarComplexF16F32 = cutlass::epilogue::thread::LinearCombinationPlanarComplex<
  cutlass::half_t, 
  4,
  float, 
  float>;

__global__ void epilogue_thread_functor_planar_complex_f16_f32(
  cutlass::half_t *output_ptr, 
  float const *accum_ptr, 
  cutlass::half_t const *source_ptr,
  typename FunctorPlanarComplexF16F32::Params params,
  int N) {

  FunctorPlanarComplexF16F32 linear_combination_op(params);

  
  auto accum = *reinterpret_cast<cutlass::ArrayPlanarComplex<float , 4> const *>(accum_ptr);   
  auto source = *reinterpret_cast<cutlass::ArrayPlanarComplex<cutlass::half_t , 4> const *>(source_ptr);

  #pragma unroll 1
  for (int n = 0; n < N; ++n) {
    source = linear_combination_op(accum, source);
  }

  *reinterpret_cast<cutlass::ArrayPlanarComplex<cutlass::half_t , 4>*>(output_ptr) = source;
}

}
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Epilogue_thread_linear_combination_planar_complex, f16_f32) {

  using Element = float;
  using ElementOutput = cutlass::half_t;
  int const kCount = 4;

  using Functor = cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    ElementOutput, 
    kCount,
    Element, 
    Element>;

  cutlass::complex<Element> alpha(Element(2), Element(1));
  cutlass::complex<Element> beta(Element(1), Element(-1));

  typename Functor::Params params(alpha, beta);

  Functor linear_combination_op(params);

  cutlass::ArrayPlanarComplex<ElementOutput, kCount> source;
  cutlass::ArrayPlanarComplex<Element, kCount> accum;

  // Define arbitrary inputs
  for (int i = 0; i < kCount; ++i) {
    accum.real[i] = Element(i * 2);
    accum.imag[i] = Element((i * 3 % 6) - 3);
    source.real[i] = ElementOutput((i * 7 % 9) - 4);
    source.imag[i] = ElementOutput(((i * 5 + 2) % 9) - 4);
  }

  cutlass::ArrayPlanarComplex<ElementOutput, kCount> destination{ linear_combination_op(accum, source) };

  // Verify each result
  for (int i = 0; i < kCount; ++i) {
    
    cutlass::complex<Element> expected = alpha * cutlass::complex<Element>(accum.real[i], accum.imag[i]) + 
      beta * cutlass::complex<Element>(Element(source.real[i]), Element(source.imag[i]));

    cutlass::complex<ElementOutput> got(destination.real[i], destination.imag[i]);
    
    EXPECT_TRUE(ElementOutput(expected.real()) == got.real());
    EXPECT_TRUE(ElementOutput(expected.imag()) == got.imag());
    EXPECT_TRUE(expected.real() != Element(0) || expected.imag() != Element(0));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace epilogue {
namespace thread {

using FunctorPlanarComplexF16F16 = cutlass::epilogue::thread::LinearCombinationPlanarComplex<
  cutlass::half_t, 
  4,
  cutlass::half_t, 
  cutlass::half_t>;

__global__ void epilogue_thread_functor_planar_complex_f16_f16(
  cutlass::half_t *output_ptr, 
  cutlass::half_t const *accum_ptr, 
  cutlass::half_t const *source_ptr,
  typename FunctorPlanarComplexF16F16::Params params,
  int N) {

  FunctorPlanarComplexF16F16 linear_combination_op(params);

  auto accum = *reinterpret_cast<cutlass::ArrayPlanarComplex<cutlass::half_t , 4> const *>(accum_ptr);
  auto source = *reinterpret_cast<cutlass::ArrayPlanarComplex<cutlass::half_t , 4> const *>(source_ptr);

  #pragma unroll 1
  for (int n = 0; n < N; ++n) {
    source = linear_combination_op(accum, source);
  }

  *reinterpret_cast<cutlass::ArrayPlanarComplex<cutlass::half_t , 4>*>(output_ptr) = source;
}

}
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Epilogue_thread_linear_combination_planar_complex, f16_f16) {

  using Element = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  int const kCount = 8;

  using Functor = cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    ElementOutput, 
    kCount,
    Element, 
    Element>;

  cutlass::complex<Element> alpha(Element(2), Element(1));
  cutlass::complex<Element> beta(Element(1), Element(-1));

  typename Functor::Params params(alpha, beta);

  Functor linear_combination_op(params);

  cutlass::ArrayPlanarComplex<ElementOutput, kCount> source;
  cutlass::ArrayPlanarComplex<Element, kCount> accum;

  // Define arbitrary inputs
  for (int i = 0; i < kCount; ++i) {
    accum.real[i] = Element(i * 2);
    accum.imag[i] = Element((i * 3 % 6) - 3);
    source.real[i] = ElementOutput((i * 7 % 9) - 4);
    source.imag[i] = ElementOutput(((i * 5 + 2) % 9) - 4);
  }

  cutlass::ArrayPlanarComplex<ElementOutput, kCount> destination = linear_combination_op(accum, source);

  // Verify each result
  for (int i = 0; i < kCount; ++i) {
    
    cutlass::complex<Element> expected = alpha * cutlass::complex<Element>(accum.real[i], accum.imag[i]) + 
      beta * cutlass::complex<Element>(Element(source.real[i]), Element(source.imag[i]));

    cutlass::complex<ElementOutput> got(destination.real[i], destination.imag[i]);
    
    EXPECT_TRUE(ElementOutput(expected.real()) == got.real());
    EXPECT_TRUE(ElementOutput(expected.imag()) == got.imag());
    EXPECT_TRUE(expected.real() != Element(0) || expected.imag() != Element(0));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
