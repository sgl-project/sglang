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
    \brief CUTLASS host-device template for complex numbers supporting all CUTLASS numeric types.
*/

#include <complex>
#include "cutlass/cutlass.h"
#include CUDA_STD_HEADER(complex)

#include "../common/cutlass_unit_test.h"

#include "cutlass/complex.h"
#include "cutlass/constants.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tfloat32.h"
#include <type_traits>

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(complex, f64_to_f32_conversion) {

  cutlass::complex<double> source = {1.5, -1.25};

  cutlass::complex<float> dest = cutlass::complex<float>(source); // explicit conversion

  EXPECT_TRUE(source.real() == 1.5 && source.imag() == -1.25 &&
    dest.real() == 1.5f && dest.imag() == -1.25f);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(complex, f32_to_f64_conversion) {

  cutlass::complex<float> source = {-1.5f, 1.25f};

  cutlass::complex<double> dest = source;  // implicit conversion

  EXPECT_TRUE(source.real() == -1.5f && source.imag() == 1.25f &&
    dest.real() == -1.5 && dest.imag() == 1.25);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(complex, s32_to_f64_conversion) {

  cutlass::complex<int> source = {-2, 1};

  cutlass::complex<double> dest = source;  // implicit conversion

  EXPECT_TRUE(source.real() == -2 && source.imag() == 1 &&
    dest.real() == -2 && dest.imag() == 1);
}


/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(complex, f16_to_f32_conversion) {

  cutlass::complex<cutlass::half_t> source = {1.5_hf, -1.25_hf};

  cutlass::complex<float> dest = cutlass::complex<float>(source); // explicit conversion

  EXPECT_TRUE(source.real() == 1.5_hf && source.imag() == -1.25_hf &&
    dest.real() == 1.5f && dest.imag() == -1.25f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(complex, exp_f32) {
  cutlass::complex<float> Z[] = {
    {1, 1},
    {2   ,  cutlass::constants::pi<float>()/2.0f   },
    {0.5f,  cutlass::constants::pi<float>()        },
    {0.25f,  cutlass::constants::pi<float>()*3/4.0f },
    {0, 0},
  };

  cutlass::complex<double> Expected[] = {
    {1.4686939399158851, 2.2873552871788423},
    {4.524491950137825e-16, 7.38905609893065},
    {-1.6487212707001282, 2.019101226849069e-16},
    {-0.9079430793557842, 0.9079430793557843},
    {1, 0}
  };

  double tolerance = 0.00001;

  for (int i = 0; cutlass::real(Z[i]) != 0.0f; ++i) {
    double e_r = cutlass::real(Expected[i]);
    double e_i = cutlass::real(Expected[i]);

    cutlass::complex<float> got = cutlass::exp(Z[i]);
    float g_r = cutlass::real(got);
    float g_i = cutlass::real(got);

    EXPECT_TRUE(
      std::abs(g_r - e_r) < tolerance && std::abs(g_i - e_i) < tolerance
    ) << "Expected(" << Expected[i] << "), Got(" << got << ")";
  }
}

TEST(complex, absolute_value_real_and_imag) {
  {
    cutlass::complex z_d{3.0, 4.0};

    auto abs_d = cutlass::abs(z_d);
    static_assert(std::is_same_v<decltype(abs_d), double>);
    EXPECT_EQ(abs_d, 5.0);

    auto real_d = cutlass::real(z_d);
    static_assert(std::is_same_v<decltype(real_d), double>);
    EXPECT_EQ(real_d, 3.0);

    auto imag_d = cutlass::imag(z_d);
    static_assert(std::is_same_v<decltype(imag_d), double>);
    EXPECT_EQ(imag_d, 4.0);
  }

  {
    cutlass::complex z_f{3.0f, 4.0f};

    auto abs_f = cutlass::abs(z_f);
    static_assert(std::is_same_v<decltype(abs_f), float>);
    EXPECT_EQ(abs_f, 5.0f);

    auto real_f = cutlass::real(z_f);
    static_assert(std::is_same_v<decltype(real_f), float>);
    EXPECT_EQ(real_f, 3.0f);

    auto imag_f = cutlass::imag(z_f);
    static_assert(std::is_same_v<decltype(imag_f), float>);
    EXPECT_EQ(imag_f, 4.0f);
  }

  {
    cutlass::complex z_tf32{cutlass::tfloat32_t{3.0f}, cutlass::tfloat32_t{4.0f}};
    auto abs_tf32 = cutlass::abs(z_tf32);
    static_assert(std::is_same_v<decltype(abs_tf32), cutlass::tfloat32_t>);
    EXPECT_EQ(abs_tf32, cutlass::tfloat32_t{5.0f});

    auto real_tf32 = cutlass::real(z_tf32);
    static_assert(std::is_same_v<decltype(real_tf32), cutlass::tfloat32_t>);
    EXPECT_EQ(real_tf32, cutlass::tfloat32_t{3.0f});

    auto imag_tf32 = cutlass::imag(z_tf32);
    static_assert(std::is_same_v<decltype(imag_tf32), cutlass::tfloat32_t>);
    EXPECT_EQ(imag_tf32, cutlass::tfloat32_t{4.0f});
  }

  {
    cutlass::complex z_i{3, 4};

    // sqrt(int) isn't a valid overload, so cutlass::abs isn't tested.
    auto real_i = cutlass::real(z_i);
    static_assert(std::is_same_v<decltype(real_i), int>);
    EXPECT_EQ(real_i, 3);

    auto imag_i = cutlass::imag(z_i);
    static_assert(std::is_same_v<decltype(imag_i), int>);
    EXPECT_EQ(imag_i, 4);
  }

  {
    double x_d{3.0};

    auto real_d = cutlass::real(x_d);
    static_assert(std::is_same_v<decltype(real_d), double>);
    EXPECT_EQ(real_d, 3.0);

    auto imag_d = cutlass::imag(x_d);
    static_assert(std::is_same_v<decltype(imag_d), double>);
    EXPECT_EQ(imag_d, 0.0);
  }

  {
    float x_f{3.0f};

    auto real_f = cutlass::real(x_f);
    static_assert(std::is_same_v<decltype(real_f), float>);
    EXPECT_EQ(real_f, 3.0f);

    auto imag_f = cutlass::imag(x_f);
    static_assert(std::is_same_v<decltype(imag_f), float>);
    EXPECT_EQ(imag_f, 0.0f);
  }

  {
    cutlass::tfloat32_t x_tf32{3.0f};

    auto real_tf32 = cutlass::real(x_tf32);
    static_assert(std::is_same_v<decltype(real_tf32), cutlass::tfloat32_t>);
    EXPECT_EQ(real_tf32, cutlass::tfloat32_t{3.0f});

    auto imag_tf32 = cutlass::imag(x_tf32);
    static_assert(std::is_same_v<decltype(imag_tf32), cutlass::tfloat32_t>);
    EXPECT_EQ(imag_tf32, cutlass::tfloat32_t{0.0f});
  }

  {
    int x_i{3};

    auto real_i = cutlass::real(x_i);
    static_assert(std::is_same_v<decltype(real_i), int>);
    EXPECT_EQ(real_i, 3);

    auto imag_i = cutlass::imag(x_i);
    static_assert(std::is_same_v<decltype(imag_i), int>);
    EXPECT_EQ(imag_i, 0);
  }
}

// FakeReal and FakeComplex test whether cutlass::real and
// cutlass::imag correctly handle user-defined non-complex
// and complex number types.
namespace test {

// These classes have no conversions to or from arithmetic types, so
// that the test can ensure that the implementation does not silently
// convert to, say, float or int.
class FakeReal {
public:
  // cutlass::imag must be able to value-construct its noncomplex input.
  FakeReal() = default;

  static CUTLASS_HOST_DEVICE FakeReal make_FakeReal(int val) {
    return FakeReal{val};
  }

  friend CUTLASS_HOST_DEVICE bool operator==(FakeReal lhs, FakeReal rhs) {
    return lhs.value_ == rhs.value_;
  }

  friend CUTLASS_HOST_DEVICE FakeReal operator-(FakeReal const& x) {
    return make_FakeReal(-x.value_);
  }

private:
  CUTLASS_HOST_DEVICE FakeReal(int val) : value_(val) {}
  int value_ = 0;
};

class FakeComplex {
public:
  static CUTLASS_HOST_DEVICE FakeComplex
  make_FakeComplex(FakeReal re, FakeReal im) {
    return FakeComplex{re, im};
  }

  // Existence of member functions real and imag tell
  // CUTLASS that FakeComplex is a complex number type.
  CUTLASS_HOST_DEVICE FakeReal real() const { return real_; }
  CUTLASS_HOST_DEVICE FakeReal imag() const { return imag_; }

  friend CUTLASS_HOST_DEVICE bool operator==(FakeComplex lhs, FakeComplex rhs) {
    return lhs.real_ == rhs.real_ && lhs.imag_ == rhs.imag_;
  }

private:
  CUTLASS_HOST_DEVICE FakeComplex(FakeReal re, FakeReal im)
    : real_(re), imag_(im)
  {}

  FakeReal real_{};
  FakeReal imag_{};
};

CUTLASS_HOST_DEVICE FakeComplex conj(FakeComplex const& z) {
  return FakeComplex::make_FakeComplex(z.real(), -z.imag());
}

// Variant of FakeComplex that has a hidden friend conj instead of a
// nonmember conj defined outside the class.
class FakeComplexWithHiddenFriendConj {
public:
  static CUTLASS_HOST_DEVICE FakeComplexWithHiddenFriendConj
  make_FakeComplexWithHiddenFriendConj(FakeReal re, FakeReal im) {
    return FakeComplexWithHiddenFriendConj{re, im};
  }

  CUTLASS_HOST_DEVICE FakeReal real() const { return real_; }
  CUTLASS_HOST_DEVICE FakeReal imag() const { return imag_; }

  friend CUTLASS_HOST_DEVICE bool
  operator==(FakeComplexWithHiddenFriendConj lhs,
    FakeComplexWithHiddenFriendConj rhs)
  {
    return lhs.real_ == rhs.real_ && lhs.imag_ == rhs.imag_;
  }

  friend CUTLASS_HOST_DEVICE FakeComplexWithHiddenFriendConj
  conj(FakeComplexWithHiddenFriendConj const& z) {
    return FakeComplexWithHiddenFriendConj::make_FakeComplexWithHiddenFriendConj(z.real(), -z.imag());
  }

private:
  CUTLASS_HOST_DEVICE
  FakeComplexWithHiddenFriendConj(FakeReal re, FakeReal im)
    : real_(re), imag_(im)
  {}

  FakeReal real_{};
  FakeReal imag_{};
};

} // namespace test

TEST(complex, real_and_imag_with_custom_types) {
  using test::FakeReal;
  using test::FakeComplex;

  {
    FakeReal x = FakeReal::make_FakeReal(42);
    auto x_r = cutlass::real(x);
    static_assert(std::is_same_v<decltype(x_r), FakeReal>);
    EXPECT_EQ(x_r, FakeReal::make_FakeReal(42));
    auto x_i = cutlass::imag(x);
    static_assert(std::is_same_v<decltype(x_i), FakeReal>);
    EXPECT_EQ(x_i, FakeReal::make_FakeReal(0));
  }
  {
    FakeComplex z = FakeComplex::make_FakeComplex(
      FakeReal::make_FakeReal(3), FakeReal::make_FakeReal(4));
    auto z_r = cutlass::real(z);
    static_assert(std::is_same_v<decltype(z_r), FakeReal>);
    EXPECT_EQ(z_r, FakeReal::make_FakeReal(3));
    auto z_i = cutlass::imag(z);
    static_assert(std::is_same_v<decltype(z_i), FakeReal>);
    EXPECT_EQ(z_i, FakeReal::make_FakeReal(4));
  }
}

namespace test {

template<class T>
void conj_tester(T z, T z_c_expected, const char type_name[]) {
  // Use cutlass::conj just like std::swap (the "std::swap two-step").
  using cutlass::conj;
  auto z_c = conj(z);
  static_assert(std::is_same_v<decltype(z_c), T>);
  constexpr bool is_cuComplex = std::is_same_v<T, cuDoubleComplex> ||
    std::is_same_v<T, cuFloatComplex>;
  if constexpr (is_cuComplex) {
    EXPECT_EQ(z_c.x, z_c_expected.x);
    EXPECT_EQ(z_c.y, z_c_expected.y) << "conj failed for type " << type_name;
  }
  else {
    EXPECT_EQ(z_c, z_c_expected) << "conj failed for type " << type_name;
  }

  auto z_c2 = cutlass::conjugate<T>{}(z);
  static_assert(std::is_same_v<decltype(z_c2), T>);
  if constexpr (is_cuComplex) {
    // cuFloatComplex and cuDoubleComplex don't report conj(z) as
    // being well-formed, probably because they are type aliases of
    // some kind.  cutlass::conj works fine, though!
    static_assert(! cutlass::platform::is_arithmetic_v<T> &&
                  (cutlass::detail::has_unqualified_conj_v<T> ||
                   cutlass::detail::has_cutlass_conj_v<T>));
    
    EXPECT_EQ(z_c2.x, z_c_expected.x);
    EXPECT_EQ(z_c2.y, z_c_expected.y)
      << "conjugate failed for type " << type_name;
  }
  else {
    EXPECT_EQ(z_c2, z_c_expected) << "conjugate failed for type " << type_name;
  }
}

} // namespace test

TEST(complex, conj_with_standard_arithmetic_types) {
  {
    double x = 42.0;
    double x_c_expected = 42.0;
    test::conj_tester(x, x_c_expected, "double");
  }
  {
    float x = 42.0f;
    float x_c_expected = 42.0f;
    test::conj_tester(x, x_c_expected, "float");
  }
  {
    int x = 42;
    int x_c_expected = 42;
    test::conj_tester(x, x_c_expected, "int");
  }
}

TEST(complex, conj_with_cutlass_complex_types) {
  {
    cutlass::complex<double> z{3.0, 4.0};
    cutlass::complex<double> z_c_expected{3.0, -4.0};
    test::conj_tester(z, z_c_expected, "cutlass::complex<double>");
  }
  {
    cutlass::complex<float> z{3.0f, 4.0f};
    cutlass::complex<float> z_c_expected{3.0f, -4.0f};
    test::conj_tester(z, z_c_expected, "cutlass::complex<float>");
  }
  {
    cutlass::complex<cutlass::tfloat32_t> z{
      cutlass::tfloat32_t{3.0f}, cutlass::tfloat32_t{4.0f}};
    cutlass::complex<cutlass::tfloat32_t> z_c_expected{
      cutlass::tfloat32_t{3.0f}, cutlass::tfloat32_t{-4.0f}};
    test::conj_tester(z, z_c_expected, "cutlass::complex<cutlass::tfloat32_t>");
  }
}

TEST(complex, conj_with_noncomplex_type_not_in_cutlass_namespace) {
  test::FakeReal x = test::FakeReal::make_FakeReal(42);
  test::FakeReal x_c_expected = test::FakeReal::make_FakeReal(42);
  test::conj_tester(x, x_c_expected, "test::FakeReal");
}

TEST(complex, conj_with_noncomplex_type_in_cutlass_namespace) {
  cutlass::tfloat32_t x{42.0f};
  cutlass::tfloat32_t x_c_expected{42.0f};
  test::conj_tester(x, x_c_expected, "cutlass::tfloat32_t");
}

TEST(complex, conj_with_complex_types_not_in_cutlass_namespace) {
  using test::FakeReal;

  // conj defined as nonmember outside the class
  {
    test::FakeComplex z = test::FakeComplex::make_FakeComplex(
      FakeReal::make_FakeReal(3), FakeReal::make_FakeReal(4));
    test::FakeComplex z_c_expected = test::FakeComplex::make_FakeComplex(
      FakeReal::make_FakeReal(3), FakeReal::make_FakeReal(-4));
    test::conj_tester(z, z_c_expected, "test::FakeComplex");
  }
  // conj defined as hidden friend
  {
    test::FakeComplexWithHiddenFriendConj z =
      test::FakeComplexWithHiddenFriendConj::make_FakeComplexWithHiddenFriendConj(
        FakeReal::make_FakeReal(3),
        FakeReal::make_FakeReal(4));
    test::FakeComplexWithHiddenFriendConj z_c_expected =
      test::FakeComplexWithHiddenFriendConj::make_FakeComplexWithHiddenFriendConj(
        FakeReal::make_FakeReal(3),
        FakeReal::make_FakeReal(-4));
    test::conj_tester(z, z_c_expected, "test::FakeComplexWithHiddenFriendConj");
  }
}

TEST(complex, conj_with_cuda_std_complex_types) {
  {
    cuda::std::complex<double> z{3.0, 4.0};
    cuda::std::complex<double> z_c_expected{3.0, -4.0};
    test::conj_tester(z, z_c_expected, "cuda::std::complex<double>");
  }
  {
    cuda::std::complex<float> z{3.0f, 4.0f};
    cuda::std::complex<float> z_c_expected{3.0f, -4.0f};
    test::conj_tester(z, z_c_expected, "cuda::std::complex<float>");
  }
}

TEST(complex, conj_with_cuComplex_types) {
  {
    cuDoubleComplex z = make_cuDoubleComplex(3.0, 4.0);
    cuDoubleComplex z_c_expected = make_cuDoubleComplex(3.0, -4.0);
    test::conj_tester(z, z_c_expected, "cuDoubleComplex");
  }
  {
    cuFloatComplex z = make_cuFloatComplex(3.0f, 4.0f);
    cuFloatComplex z_c_expected = make_cuFloatComplex(3.0f, -4.0f);
    test::conj_tester(z, z_c_expected, "cuFloatComplex");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {

  /// Thorough testing for basic complex math operators. Uses std::complex as a reference.
  template <typename T, int N, int M>
  struct ComplexOperators {
    ComplexOperators() {
      for (int ar = -N; ar <= N; ++ar) {
        for (int ai = -N; ai <= N; ++ai) {
          for (int br = -N; br <= N; ++br) {
            for (int bi = -N; bi <= N; ++bi) {

              cutlass::complex<T> Ae(T(ar) / T(M), T(ai) / T(M));
              cutlass::complex<T> Be(T(br) / T(M), T(bi) / T(M));

              std::complex<T> Ar(T(ar) / T(M), T(ai) / T(M));
              std::complex<T> Br(T(br) / T(M), T(bi) / T(M));

              cutlass::complex<T> add_e = Ae + Be;
              cutlass::complex<T> sub_e = Ae - Be;
              cutlass::complex<T> mul_e = Ae * Be;

              std::complex<T> add_r = (Ar + Br);
              std::complex<T> sub_r = (Ar - Br);
              std::complex<T> mul_r = (Ar * Br);

              EXPECT_EQ(real(add_e), real(add_r));
              EXPECT_EQ(imag(add_e), imag(add_r));

              EXPECT_EQ(real(sub_e), real(sub_r));
              EXPECT_EQ(imag(sub_e), imag(sub_r));

              EXPECT_EQ(real(mul_e), real(mul_r));
              EXPECT_EQ(imag(mul_e), imag(mul_r));

              if (!(br == 0 && bi == 0)) {

                cutlass::complex<T> div_e = Ae / Be;
                std::complex<T> div_r = Ar / Br;

                T const kRange = T(0.001);

                EXPECT_NEAR(real(div_e), real(div_r), kRange);
                EXPECT_NEAR(imag(div_e), imag(div_r), kRange);
              }
            }
          }
        }
      }
    }
  };
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(complex, host_float) {
  test::ComplexOperators<float, 32, 8> test;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(complex, host_double) {
  test::ComplexOperators<double, 32, 8> test;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
