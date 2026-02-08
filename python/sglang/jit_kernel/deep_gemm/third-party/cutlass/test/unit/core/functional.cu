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
    \brief Unit tests for functional operators.
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/functional.h"
#include "cutlass/core_io.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/util/host_tensor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace core {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Conversion template
template <typename Element, typename Operator>
__global__ void unary_operator(Element *d, Element const *a) {

  Operator op;

  *d = op(*a);
}

/// Conversion template
template <typename Element, typename Operator>
__global__ void binary_operator(Element *d, Element const *a, Element const *b, int Iterations = 1) {

  Operator op;

  Element a_x = *a;
  Element b_x = *b;

  CUTLASS_PRAGMA_NO_UNROLL
  for (int i = 0; i < Iterations; ++i) {
    b_x = op(a_x, b_x);
  }
  
  *d = b_x;
}

/// Conversion template
template <typename Element, typename Operator>
__global__ void trinary_operator(
  Element *d, 
  Element const *a, 
  Element const *b, 
  Element const *c, 
  int Iterations = 1) {

  Operator op;

  Element a_x = a[blockIdx.x];
  Element b_x = b[blockIdx.x];
  Element c_x = c[blockIdx.x];

  CUTLASS_PRAGMA_NO_UNROLL
  for (int i = 0; i < Iterations; ++i) {
    c_x = op(a_x, b_x, c_x);
  }
  
  d[blockIdx.x] = c_x;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace core
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int kN>
void Functional_plus_f16xN() {

  using Element = cutlass::Array<cutlass::half_t, kN>;
  using Operator = cutlass::plus<Element>;

  using Tensor = cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>;

  Tensor D({1, kN});
  Tensor A({1, kN});
  Tensor B({1, kN});
  Tensor C({1, kN});

  for (int i = 0; i < kN; ++i) {
    A.host_data()[i] = cutlass::half_t((i * 2 + 1) % 5);
    B.host_data()[i] = cutlass::half_t((i * 4 + 8) % 7);
    D.host_data()[i] = cutlass::half_t(0);
  }

  D.sync_device();
  A.sync_device();
  B.sync_device();

  test::core::kernel::binary_operator<Element, Operator><<< dim3(1,1), dim3(1,1) >>>(
    reinterpret_cast<Element *>(D.device_data()),
    reinterpret_cast<Element const *>(A.device_data()),
    reinterpret_cast<Element const *>(B.device_data())
  );

  D.sync_host();

  bool some_d_nonzero = false;

  for (int i = 0; i < kN; ++i) {
    float a = float(A.host_data()[i]);
    float b = float(B.host_data()[i]);
    float d = float(D.host_data()[i]);

    EXPECT_TRUE(d == (a + b));

    if (d != 0) {
      some_d_nonzero = true;
    }
  }

  EXPECT_TRUE(some_d_nonzero);
}

TEST(Functional, plus_f16x16) {
  Functional_plus_f16xN<16>();
}

TEST(Functional, plus_f16x17) {
  Functional_plus_f16xN<17>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int kN>
void Functional_minus_f16xN() {

  using Element = cutlass::Array<cutlass::half_t, kN>;
  using Operator = cutlass::minus<Element>;

  using Tensor = cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>;

  Tensor D({1, kN});
  Tensor A({1, kN});
  Tensor B({1, kN});
  Tensor C({1, kN});

  for (int i = 0; i < kN; ++i) {
    A.host_data()[i] = cutlass::half_t((i * 2 + 1) % 5);
    B.host_data()[i] = cutlass::half_t((i * 4 + 8) % 7);
    D.host_data()[i] = cutlass::half_t(0);
  }

  D.sync_device();
  A.sync_device();
  B.sync_device();

  test::core::kernel::binary_operator<Element, Operator><<< dim3(1,1), dim3(1,1) >>>(
    reinterpret_cast<Element *>(D.device_data()),
    reinterpret_cast<Element const *>(A.device_data()),
    reinterpret_cast<Element const *>(B.device_data())
  );

  D.sync_host();

  bool some_d_nonzero = false;

  for (int i = 0; i < kN; ++i) {
    float a = float(A.host_data()[i]);
    float b = float(B.host_data()[i]);
    float d = float(D.host_data()[i]);

    EXPECT_TRUE(d == (a - b));

    if (d != 0) {
      some_d_nonzero = true;
    }
  }

  EXPECT_TRUE(some_d_nonzero);
}

TEST(Functional, minus_f16x16) {
  Functional_minus_f16xN<16>();
}

TEST(Functional, minus_f16x17) {
  Functional_minus_f16xN<17>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int kN>
void Functional_multiplies_f16xN() {

  using Element = cutlass::Array<cutlass::half_t, kN>;
  using Operator = cutlass::multiplies<Element>;

  using Tensor = cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>;

  Tensor D({1, kN});
  Tensor A({1, kN});
  Tensor B({1, kN});
  Tensor C({1, kN});

  for (int i = 0; i < kN; ++i) {
    A.host_data()[i] = cutlass::half_t((i * 2 + 1) % 5);
    B.host_data()[i] = cutlass::half_t((i * 4 + 8) % 7);
    D.host_data()[i] = cutlass::half_t(0);
  }

  D.sync_device();
  A.sync_device();
  B.sync_device();

  test::core::kernel::binary_operator<Element, Operator><<< dim3(1,1), dim3(1,1) >>>(
    reinterpret_cast<Element *>(D.device_data()),
    reinterpret_cast<Element const *>(A.device_data()),
    reinterpret_cast<Element const *>(B.device_data())
  );

  D.sync_host();

  bool some_d_nonzero = false;

  for (int i = 0; i < kN; ++i) {
    float a = float(A.host_data()[i]);
    float b = float(B.host_data()[i]);
    float d = float(D.host_data()[i]);

    EXPECT_TRUE(d == (a * b));

    if (d != 0) {
      some_d_nonzero = true;
    }
  }

  EXPECT_TRUE(some_d_nonzero);
}

TEST(Functional, multiplies_f16x16) {

  Functional_multiplies_f16xN<16>();
}

TEST(Functional, multiplies_f16x17) {

  Functional_multiplies_f16xN<17>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int kN>
void Functional_divides_f16xN() {

  using Element = cutlass::Array<cutlass::half_t, kN>;
  using Operator = cutlass::divides<Element>;

  using Tensor = cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>;

  Tensor D({1, kN});
  Tensor A({1, kN});
  Tensor B({1, kN});
  Tensor C({1, kN});

  for (int i = 0; i < kN; ++i) {
    A.host_data()[i] = cutlass::half_t((i * 2 + 1) % 5);
    B.host_data()[i] = cutlass::half_t((i * 4 + 8) % 7);
    D.host_data()[i] = cutlass::half_t(0);
  }

  D.sync_device();
  A.sync_device();
  B.sync_device();

  test::core::kernel::binary_operator<Element, Operator><<< dim3(1,1), dim3(1,1) >>>(
    reinterpret_cast<Element *>(D.device_data()),
    reinterpret_cast<Element const *>(A.device_data()),
    reinterpret_cast<Element const *>(B.device_data())
  );

  D.sync_host();

  bool some_d_nonzero = false;

  for (int i = 0; i < kN; ++i) {
    float a = float(A.host_data()[i]);
    float b = float(B.host_data()[i]);
    float d = float(D.host_data()[i]);

    float expected = a / b;

    float const kThreshold = 0.0005f;

    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(d));
    }
    else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(d));
    }
    else {
      EXPECT_TRUE(std::abs(d - expected) < kThreshold)
        << "Got: " << d << " = " << a << " / " << b << ", expected: " << (a / b); 
    }

    if (d != 0) {
      some_d_nonzero = true;
    }
  }

  EXPECT_TRUE(some_d_nonzero);
}

TEST(Functional, divides_f16x16) {

  Functional_divides_f16xN<16>();
}

TEST(Functional, divides_f16x17) {

  Functional_divides_f16xN<17>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int kN>
void Functional_multiply_add_TxN() {

  using Element = cutlass::Array<T, kN>;
  using Operator = cutlass::multiply_add<Element>;

  using Tensor = cutlass::HostTensor<T, cutlass::layout::RowMajor>;

  Tensor D({1, kN});
  Tensor A({1, kN});
  Tensor B({1, kN});
  Tensor C({1, kN});

  for (int i = 0; i < kN; ++i) {
    A.host_data()[i] = T((i * 2 + 1) % 5);
    B.host_data()[i] = T((i * 4 + 8) % 7);
    C.host_data()[i] = T((i * 3 + 11) % 11);
    D.host_data()[i] = T(0);
  }

  D.sync_device();
  A.sync_device();
  B.sync_device();
  C.sync_device();

  test::core::kernel::trinary_operator<Element, Operator><<< dim3(1,1), dim3(1,1) >>>(
    reinterpret_cast<Element *>(D.device_data()),
    reinterpret_cast<Element const *>(A.device_data()),
    reinterpret_cast<Element const *>(B.device_data()),
    reinterpret_cast<Element const *>(C.device_data())
  );

  D.sync_host();

  bool some_d_nonzero = false;

  for (int i = 0; i < kN; ++i) {
    float a = float(A.host_data()[i]);
    float b = float(B.host_data()[i]);
    float c = float(C.host_data()[i]);
    float d = float(D.host_data()[i]);

    EXPECT_TRUE(d == (a * b + c));

    if (d != 0) {
      some_d_nonzero = true;
    }
  }

  EXPECT_TRUE(some_d_nonzero);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Functional, multiply_add_f16x16) {
  Functional_multiply_add_TxN<cutlass::half_t, 16>();
}

TEST(Functional, multiply_add_f16x17) {
  Functional_multiply_add_TxN<cutlass::half_t, 17>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Functional, multiply_add_bf16x16) {
  Functional_multiply_add_TxN<cutlass::bfloat16_t, 16>();
}

TEST(Functional, multiply_add_bf16x17) {
  Functional_multiply_add_TxN<cutlass::bfloat16_t, 17>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
cutlass::Quaternion<T> random_quaternion(int range) {
  return cutlass::Quaternion<T>{
    T((rand() % range * 2) - range),
    T((rand() % range * 2) - range),
    T((rand() % range * 2) - range),
    T((rand() % range * 2) - range)
  };
}

template <typename T>
void Functional_multiply_add_QuaternionT() {

  using Element = cutlass::Quaternion<T>;
  using Operator = cutlass::multiply_add<Element, Element, Element>;
  using HostTensor = cutlass::HostTensor<Element, cutlass::layout::RowMajor>;

  int const kM = 128;
  int const kRange = 8;

  HostTensor A({kM, 1});
  HostTensor B({kM, 1});
  HostTensor C({kM, 1});
  HostTensor D({kM, 1});

  srand(2021);

  for (int m = 0; m < kM; ++m) {
    A.at({m, 0}) = random_quaternion<T>(kRange);
    B.at({m, 0}) = random_quaternion<T>(kRange);
    C.at({m, 0}) = random_quaternion<T>(kRange);
  }

  A.sync_device();
  B.sync_device();
  C.sync_device();
  D.sync_device();

  test::core::kernel::trinary_operator<Element, Operator><<< dim3(kM,1), dim3(1,1) >>>(
    D.device_data(),
    A.device_data(),
    B.device_data(),
    C.device_data()
  );

  D.sync_host();
  
  for (int m = 0; m < kM; ++m) {

    Element a = A.at({m, 0});
    Element b = B.at({m, 0});
    Element c = C.at({m, 0});
    Element got = D.at({m, 0});
    Element expected = a * b + c;

    EXPECT_TRUE(got == expected);
  }
}

TEST(Functional, multiply_add_quaternion_f32) {
  Functional_multiply_add_QuaternionT<float>();
}

namespace cutlass_test {

__global__ void
test_cutlass_maximum(cutlass::half_t const* in1, cutlass::half_t const* in2, cutlass::half_t* out)
{
  {
  constexpr bool propagate_NaN = true;
  cutlass::maximum<cutlass::half_t, propagate_NaN> op;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
    && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    *out = op(*in1, *in2);
  }
  }
  constexpr bool propagate_NaN = false;
  cutlass::maximum<cutlass::half_t, propagate_NaN> op;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
    && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    *out = op(*in1, *in2);
  }
}

} // cutlass_test

// Test compilation on both host and device.
TEST(Functional, maximum_half_host_propagate_NaN) {
  constexpr bool propagate_NaN = true;
  cutlass::maximum<cutlass::half_t, propagate_NaN> op;
  cutlass::half_t x(1.0f);
  cutlass::half_t y(2.0f);

  auto result = op(x, y);
  static_assert(std::is_same_v<decltype(result), cutlass::half_t>);
  EXPECT_EQ(result, y);
  result = op(y, x);
  EXPECT_EQ(result, y);
}

TEST(Functional, maximum_half_host_dont_propagate_NaN) {
  constexpr bool propagate_NaN = false;
  cutlass::maximum<cutlass::half_t, propagate_NaN> op;
  cutlass::half_t x(1.0f);
  cutlass::half_t y(2.0f);

  auto result = op(x, y);
  static_assert(std::is_same_v<decltype(result), cutlass::half_t>);
  EXPECT_EQ(result, y);
  result = op(y, x);
  EXPECT_EQ(result, y);
}

TEST(FUnction, maximum_half_device) {
  using Tensor = cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>;

  Tensor in1({1, 1});
  Tensor in2({1, 1});
  Tensor out({1, 1});
  in1.host_data()[0] = cutlass::half_t(1.0f);
  in2.host_data()[0] = cutlass::half_t(2.0f);
  out.host_data()[0] = cutlass::half_t(0.0f);

  in1.sync_device();
  in2.sync_device();
  out.sync_device();

  cutlass_test::test_cutlass_maximum<<< 1, 1 >>>(
    in1.device_data(),
    in2.device_data(),
    out.device_data()
  );
  out.sync_host();

  EXPECT_EQ(out.host_data()[0], 2.0f);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
