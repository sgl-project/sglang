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
    \brief Unit tests for conversion operators.
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/numeric_conversion.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/util/host_tensor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace core {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Simple conversion function
template <typename Destination, typename Source, int Count>
__global__ void convert(
  cutlass::Array<Destination, Count> *destination,
  cutlass::Array<Source, Count> const *source) {

  cutlass::NumericArrayConverter<Destination, Source, Count> convert;

  *destination = convert(*source);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Destination, typename Source, int Count>
void run_test(const char dest_name[], const char source_name[], const int range = 4, const int offset = 0) {
  const int kN = Count;

  dim3 grid(1, 1);
  dim3 block(1, 1);

  cutlass::HostTensor<Destination, cutlass::layout::RowMajor> destination({1, kN});
  cutlass::HostTensor<Source, cutlass::layout::RowMajor> source({1, kN});
  auto source_ref = source.host_ref();
  auto destination_ref = destination.host_ref();

  for (int i = 0; i < kN; ++i) {
    source_ref.at({0, i}) = Source(i % range + offset);
  }

  source.sync_device();

  convert<Destination, Source, kN><<< grid, block >>>(
    reinterpret_cast<cutlass::Array<Destination, kN> *>(destination.device_data()),
    reinterpret_cast<cutlass::Array<Source, kN> const *>(source.device_data())
  );

  destination.sync_host();

  for (int i = 0; i < kN; ++i) {
    EXPECT_TRUE(float(destination_ref.at({0, i})) == float(source_ref.at({0, i})))
      << "Destination type: " << dest_name << " "<< float(destination_ref.at({0, i}))
      << ", Source type: " << source_name << " " << float(source_ref.at({0, i}))
      << ", Count: " << Count;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Destination, typename Source, typename ScaleFactor, int Count>
__global__ void convert_with_scale_factor(
  cutlass::Array<Destination, Count> *destination,
  cutlass::Array<Source, Count> const *source,
  cutlass::Array<ScaleFactor, Count> const *scale_factor) {

  cutlass::NumericArrayConverter<Destination, Source, Count> convert;

  *destination = convert(*source, *scale_factor);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Destination, typename Source, typename ScaleFactor,  int Count>
void run_test_with_scalefactor(const char dest_name[], const char source_name[], const char scale_factor_name[], const int range = 4, const int offset = 0) {
  const int kN = Count;

  dim3 grid(1, 1);
  dim3 block(1, 1);

  cutlass::HostTensor<Destination, cutlass::layout::RowMajor> destination({1, kN});
  cutlass::HostTensor<Source, cutlass::layout::RowMajor> source({1, kN});
  cutlass::HostTensor<ScaleFactor, cutlass::layout::RowMajor> scale_factor({1, kN});
  auto source_ref = source.host_ref();
  auto destination_ref = destination.host_ref();
  auto scale_factor_ref = scale_factor.host_ref();


  for (int i = 0; i < kN; ++i) {
    source_ref.at({0, i}) = Source(i % range + offset);
  }

  for (int i = 0; i < kN; ++i) {
    scale_factor_ref.at({0, i}) = ScaleFactor(1 + i % 8);
  }

  source.sync_device();
  scale_factor.sync_device();

  convert_with_scale_factor<Destination, Source, ScaleFactor, kN><<< grid, block >>>(
    reinterpret_cast<cutlass::Array<Destination, kN> *>(destination.device_data()),
    reinterpret_cast<cutlass::Array<Source, kN> const *>(source.device_data()),
    reinterpret_cast<cutlass::Array<ScaleFactor, kN> const *>(scale_factor.device_data())
  );

  destination.sync_host();

  for (int i = 0; i < kN; ++i) {
    float ref = float(source_ref.at({0, i})) / float(scale_factor_ref.at({0, i}));
    bool pass = float(destination_ref.at({0, i})) == ref;
    EXPECT_TRUE(pass) 
      << "Destination type: " << dest_name << " "<< float(destination_ref.at({0, i})) << std::endl
      << ", Source type: " << source_name << " " << float(source_ref.at({0, i})) << std::endl
      << ", Scalefactor type: " << source_name << " " << float(scale_factor_ref.at({0, i})) << std::endl
      << ", idx: " << i << std::endl;
  }
}

} // namespace kernel
} // namespace core
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(NumericConversion, f32_to_f16_rn) {
  constexpr int kN = 1;
  using Source = float;
  const char source_name[] = "float";
  using Destination = cutlass::half_t;
  const char dest_name[] = "half_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f32x2_to_f16x2_rn) {
  constexpr int kN = 2;
  using Source = float;
  const char source_name[] = "float";
  using Destination = cutlass::half_t;
  const char dest_name[] = "half_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f32x8_to_f16x8_rn) {
  constexpr int kN = 8;
  using Source = float;
  const char source_name[] = "float";
  using Destination = cutlass::half_t;
  const char dest_name[] = "half_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(NumericConversion, f16_to_f32_rn) {  
  int const kN = 1;
  using Source = cutlass::half_t;
  const char source_name[] = "half_t";
  using Destination = float;
  const char dest_name[] = "float";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f16x8_to_f32x8_rn) {
  int const kN = 8;
  using Source = cutlass::half_t;
  const char source_name[] = "half_t";
  using Destination = float;
  const char dest_name[] = "float";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(NumericConversion, f32_to_fe4m3_rn) {
  int const kN = 1;
  using Source = float;
  const char source_name[] = "float";
  using Destination = cutlass::float_e4m3_t;
  const char dest_name[] = "float_e4m3_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f32_to_fe4m3_rn_2_elements) {
  int const kN = 2;
  using Source = float;
  const char source_name[] = "float";
  using Destination = cutlass::float_e4m3_t;
  const char dest_name[] = "float_e4m3_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f32_to_fe4m3_rn_array) {
  int const kN = 27;
  using Source = float;
  const char source_name[] = "float";
  using Destination = cutlass::float_e4m3_t;
  const char dest_name[] = "float_e4m3_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f32_to_fe5m2_rn) {
  int const kN = 1;
  using Source = float;
  const char source_name[] = "float";
  using Destination = cutlass::float_e5m2_t;
  const char dest_name[] = "float_e5m2_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f32_to_fe5m2_rn_2_elements) {
  int const kN = 2;
  using Source = float;
  const char source_name[] = "float";
  using Destination = cutlass::float_e5m2_t;
  const char dest_name[] = "float_e5m2_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f32_to_fe5m2_rn_array) {
  int const kN = 27;
  using Source = float;
  const char source_name[] = "float";
  using Destination = cutlass::float_e5m2_t;
  const char dest_name[] = "float_e5m2_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f16_to_fe4m3_rn) {
  int const kN = 1;
  using Source = cutlass::half_t;
  const char source_name[] = "half_t";
  using Destination = cutlass::float_e4m3_t;
  const char dest_name[] = "float_e4m3_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f16_to_fe4m3_rn_2_elements) {
  int const kN = 2;
  using Source = cutlass::half_t;
  const char source_name[] = "half_t";
  using Destination = cutlass::float_e4m3_t;
  const char dest_name[] = "float_e4m3_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f16_to_fe4m3_rn_array) {
  int const kN = 27;
  using Source = cutlass::half_t;
  const char source_name[] = "half_t";
  using Destination = cutlass::float_e4m3_t;
  const char dest_name[] = "float_e4m3_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f16_to_fe5m2_rn) {
  int const kN = 1;
  using Source = cutlass::half_t;
  const char source_name[] = "half_t";
  using Destination = cutlass::float_e5m2_t;
  const char dest_name[] = "float_e5m2_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f16_to_fe5m2_rn_2_elements) {
  int const kN = 27;
  using Source = cutlass::half_t;
  const char source_name[] = "half_t";
  using Destination = cutlass::float_e5m2_t;
  const char dest_name[] = "float_e5m2_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f16_to_fe5m2_rn_array) {
  int const kN = 27;
  using Source = cutlass::half_t;
  const char source_name[] = "half_t";
  using Destination = cutlass::float_e5m2_t;
  const char dest_name[] = "float_e5m2_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, bf16_to_fe4m3_rn) {
  int const kN = 1;
  using Source = cutlass::bfloat16_t;
  const char source_name[] = "bfloat16_t";
  using Destination = cutlass::float_e4m3_t;
  const char dest_name[] = "float_e4m3_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, bf16_to_fe4m3_rn_2_elements) {
  int const kN = 27;
  using Source = cutlass::bfloat16_t;
  const char source_name[] = "bfloat16_t";
  using Destination = cutlass::float_e4m3_t;
  const char dest_name[] = "float_e4m3_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, bf16_to_fe4m3_rn_array) {
  int const kN = 27;
  using Source = cutlass::bfloat16_t;
  const char source_name[] = "bfloat16_t";
  using Destination = cutlass::float_e4m3_t;
  const char dest_name[] = "float_e4m3_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, bf16_to_fe5m2_rn) {
  int const kN = 1;
  using Source = cutlass::bfloat16_t;
  const char source_name[] = "bfloat16_t";
  using Destination = cutlass::float_e5m2_t;
  const char dest_name[] = "float_e5m2_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, bf16_to_fe5m2_rn_2_elements) {
  int const kN = 27;
  using Source = cutlass::bfloat16_t;
  const char source_name[] = "bfloat16_t";
  using Destination = cutlass::float_e5m2_t;
  const char dest_name[] = "float_e5m2_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, bf16_to_fe5m2_rn_array) {
  int const kN = 27;
  using Source = cutlass::bfloat16_t;
  const char source_name[] = "bfloat16_t";
  using Destination = cutlass::float_e5m2_t;
  const char dest_name[] = "float_e5m2_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(NumericConversion, fe4m3_to_fe5m2_rn) {
  int const kN = 1;
  using Source = cutlass::float_e4m3_t;
  const char source_name[] = "float_e4m3_t";
  using Destination = cutlass::float_e5m2_t;
  const char dest_name[] = "float_e5m2_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe4m3_to_fe5m2_2_elements) {
  int const kN = 27;
  using Source = cutlass::float_e4m3_t;
  const char source_name[] = "float_e4m3_t";
  using Destination = cutlass::float_e5m2_t;
  const char dest_name[] = "float_e5m2_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe4m3_to_fe5m2_array) {
  int const kN = 27;
  using Source = cutlass::float_e4m3_t;
  const char source_name[] = "float_e4m3_t";
  using Destination = cutlass::float_e5m2_t;
  const char dest_name[] = "float_e5m2_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe5m2_to_fe4m3_rn) {
  int const kN = 1;
  using Source = cutlass::float_e5m2_t;
  const char source_name[] = "float_e5m2_t";
  using Destination = cutlass::float_e4m3_t;
  const char dest_name[] = "float_e4m3_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe5m2_to_fe4m3_2_elements) {
  int const kN = 27;
  using Source = cutlass::float_e5m2_t;
  const char source_name[] = "float_e5m2_t";
  using Destination = cutlass::float_e4m3_t;
  const char dest_name[] = "float_e4m3_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe5m2_to_fe4m3_array) {
  int const kN = 27;
  using Source = cutlass::float_e5m2_t;
  const char source_name[] = "float_e5m2_t";
  using Destination = cutlass::float_e4m3_t;
  const char dest_name[] = "float_e4m3_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe4m3_to_f32_rn) {
  int const kN = 1;
  using Source = cutlass::float_e4m3_t;
  const char source_name[] = "float_e4m3_t";
  using Destination = float;
  const char dest_name[] = "float";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(NumericConversion, f32x8_to_s8x8_rn) {

  int const kN = 8;
  using Source = float;
  const char source_name[] = "float";
  using Destination = int8_t;
  const char dest_name[] = "int8_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe4m3_to_f32_2_elements) {
  int const kN = 2;
  using Source = cutlass::float_e4m3_t;
  const char source_name[] = "float_e4m3_t";
  using Destination = float;
  const char dest_name[] = "float";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe4m3_to_f32_array) {
  int const kN = 27;
  using Source = cutlass::float_e4m3_t;
  const char source_name[] = "float_e4m3_t";
  using Destination = float;
  const char dest_name[] = "float";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe5m2_to_f32_2_elements) {
  int const kN = 2;
  using Source = cutlass::float_e5m2_t;
  const char source_name[] = "float_e5m2_t";
  using Destination = float;
  const char dest_name[] = "float";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe5m2_to_f32_array) {
  int const kN = 27;
  using Source = cutlass::float_e5m2_t;
  const char source_name[] = "float_e5m2_t";
  using Destination = float;
  const char dest_name[] = "float";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe4m3_to_f16_rn) {
  int const kN = 1;
  using Source = cutlass::float_e4m3_t;
  const char source_name[] = "float_e4m3_t";
  using Destination = cutlass::half_t;
  const char dest_name[] = "half_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe4m3_to_f16_2_elements) {
  int const kN = 2;
  using Source = cutlass::float_e4m3_t;
  const char source_name[] = "float_e4m3_t";
  using Destination = cutlass::half_t;
  const char dest_name[] = "half_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe4m3_to_f16_array) {
  int const kN = 27;
  using Source = cutlass::float_e4m3_t;
  const char source_name[] = "float_e4m3_t";
  using Destination = cutlass::half_t;
  const char dest_name[] = "half_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe5m2_to_f16_rn) {
  int const kN = 1;
  using Source = cutlass::float_e5m2_t;
  const char source_name[] = "float_e5m2_t";
  using Destination = cutlass::half_t;
  const char dest_name[] = "half_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe5m2_to_f16_2_elements) {
  int const kN = 2;
  using Source = cutlass::float_e5m2_t;
  const char source_name[] = "float_e5m2_t";
  using Destination = cutlass::half_t;
  const char dest_name[] = "half_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe5m2_to_f16_array) {
  int const kN = 27;
  using Source = cutlass::float_e5m2_t;
  const char source_name[] = "float_e5m2_t";
  using Destination = cutlass::half_t;
  const char dest_name[] = "half_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe4m3_to_bf16_rn) {
  int const kN = 1;
  using Source = cutlass::float_e4m3_t;
  const char source_name[] = "float_e4m3_t";
  using Destination = cutlass::bfloat16_t;
  const char dest_name[] = "bfloat16_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe4m3_to_bf16_2_elements) {
  int const kN = 2;
  using Source = cutlass::float_e4m3_t;
  const char source_name[] = "float_e4m3_t";
  using Destination = cutlass::bfloat16_t;
  const char dest_name[] = "bfloat16_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe4m3_to_bf16_array) {
  int const kN = 27;
  using Source = cutlass::float_e4m3_t;
  const char source_name[] = "float_e4m3_t";
  using Destination = cutlass::bfloat16_t;
  const char dest_name[] = "bfloat16_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe5m2_to_bf16_rn) {
  int const kN = 1;
  using Source = cutlass::float_e5m2_t;
  const char source_name[] = "float_e5m2_t";
  using Destination = cutlass::bfloat16_t;
  const char dest_name[] = "bfloat16_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe5m2_to_bf16_2_elements) {
  int const kN = 2;
  using Source = cutlass::float_e5m2_t;
  const char source_name[] = "float_e5m2_t";
  using Destination = cutlass::bfloat16_t;
  const char dest_name[] = "bfloat16_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe5m2_to_bf16_array) {
  int const kN = 27;
  using Source = cutlass::float_e5m2_t;
  const char source_name[] = "float_e5m2_t";
  using Destination = cutlass::bfloat16_t;
  const char dest_name[] = "bfloat16_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

// These are included as regression tests for a special case when N = 4.
TEST(NumericConversion, int4b_t_to_fe5m2_t_array_4) {
  int const kN = 4;
  using Source = cutlass::int4b_t;
  const char source_name[] = "int4b_t";
  using Destination = cutlass::float_e5m2_t;
  const char dest_name[] = "float_e5m2_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, int_to_fe4m3_t_array_4) {
  int const kN = 4;
  using Source = int;
  const char source_name[] = "int";
  using Destination = cutlass::float_e4m3_t;
  const char dest_name[] = "float_e4m3_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, int2b_t_to_fe4m3_t_array_4) {
  int const kN = 4;
  using Source = cutlass::int2b_t;
  const char source_name[] = "int2b_t";
  using Destination = cutlass::float_e4m3_t;
  const char dest_name[] = "float_e4m3_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, fe5m2_t_to_double_array_4) {
  int const kN = 4;
  using Source = cutlass::float_e5m2_t;
  const char source_name[] = "float_e5m2_t";
  using Destination = double;
  const char dest_name[] = "double";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, int_to_fe4m3_t_array_32) {
  int const kN = 32;
  using Source = int;
  const char source_name[] = "int";
  using Destination = cutlass::float_e4m3_t;
  const char dest_name[] = "float_e4m3_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct GetName {
  static constexpr char name[] = "UNSUPPORTED";
};

template <>
struct GetName<cutlass::int2b_t> {
  static constexpr char name[] = "int2b_t";
};

template <>
struct GetName<cutlass::uint2b_t> {
  static constexpr char name[] = "uint2b_t";
};

template <>
struct GetName<cutlass::int4b_t> {
  static constexpr char name[] = "int4b_t";
};

template <>
struct GetName<cutlass::uint4b_t> {
  static constexpr char name[] = "uint4b_t";
};

template <>
struct GetName<uint8_t> {
  static constexpr char name[] = "uint8_t";
};

template <>
struct GetName<int8_t> {
  static constexpr char name[] = "int8_t";
};

template <>
struct GetName<cutlass::float_e4m3_t> {
  static constexpr char name[] = "float_e4m3_t";
};

template <>
struct GetName<cutlass::float_e5m2_t> {
  static constexpr char name[] = "float_e5m2_t";
};

template <>
struct GetName<cutlass::half_t> {
  static constexpr char name[] = "half_t";
};

template <>
struct GetName<cutlass::bfloat16_t> {
  static constexpr char name[] = "bfloat16_t";
};

template <>
struct GetName<float> {
  static constexpr char name[] = "float";
};

template <typename Result_, typename Source_>
struct ResultSourcePair {
  using Result = Result_;
  using Source = Source_;
};

template <typename ResultSourcePair>
class VectorArrayConverterTest : public testing::Test {
 public:
  using Result = typename ResultSourcePair::Result;
  using Source = typename ResultSourcePair::Source;
  
  template <int N>
  static void emit_test() { 
    const int range = 1 << cutlass::sizeof_bits<Source>::value;
    const int offset = cutlass::platform::numeric_limits<Source>::lowest();
    test::core::kernel::run_test<Result, Source, N>(GetName<Result>::name, GetName<Source>::name, range, offset);
  }
};

using VectorConvertTypes = ::testing::Types<
  ResultSourcePair<float, int8_t>,
  ResultSourcePair<float, uint8_t>,

  ResultSourcePair<cutlass::half_t, int8_t>,
  ResultSourcePair<cutlass::half_t, uint8_t>,

  ResultSourcePair<cutlass::bfloat16_t, uint8_t>,
  ResultSourcePair<cutlass::bfloat16_t, int8_t>,

  ResultSourcePair<cutlass::float_e4m3_t, cutlass::int2b_t>,
  ResultSourcePair<cutlass::float_e5m2_t, cutlass::int2b_t>,
  ResultSourcePair<cutlass::half_t, cutlass::int2b_t>,
  ResultSourcePair<cutlass::bfloat16_t, cutlass::int2b_t>,
  ResultSourcePair<cutlass::float_e4m3_t, cutlass::uint2b_t>,
  ResultSourcePair<cutlass::float_e5m2_t, cutlass::uint2b_t>,
  ResultSourcePair<cutlass::half_t, cutlass::uint2b_t>,
  ResultSourcePair<cutlass::bfloat16_t, cutlass::uint2b_t>,

  ResultSourcePair<cutlass::float_e4m3_t, cutlass::int4b_t>,
  ResultSourcePair<cutlass::float_e5m2_t, cutlass::int4b_t>,
  ResultSourcePair<cutlass::half_t, cutlass::int4b_t>,
  ResultSourcePair<cutlass::bfloat16_t, cutlass::int4b_t>,
  ResultSourcePair<cutlass::float_e4m3_t, cutlass::uint4b_t>,
  ResultSourcePair<cutlass::half_t, cutlass::uint4b_t>,
  ResultSourcePair<cutlass::bfloat16_t, cutlass::uint4b_t>,
  ResultSourcePair<float, cutlass::int4b_t>
>;

TYPED_TEST_SUITE(VectorArrayConverterTest, VectorConvertTypes);

TYPED_TEST(VectorArrayConverterTest, array_1) {
  TestFixture::template emit_test<1>();
}

TYPED_TEST(VectorArrayConverterTest, array_2) {
  TestFixture::template emit_test<2>();
}

TYPED_TEST(VectorArrayConverterTest, array_3) {
  TestFixture::template emit_test<3>();
}

TYPED_TEST(VectorArrayConverterTest, array_4) {
  TestFixture::template emit_test<4>();
}

TYPED_TEST(VectorArrayConverterTest, array_5) {
  TestFixture::template emit_test<5>();
}

TYPED_TEST(VectorArrayConverterTest, array_8) {
  TestFixture::template emit_test<8>();
}

TYPED_TEST(VectorArrayConverterTest, array_10) {
  // N > 8 and N is not a multiple of 4
  TestFixture::template emit_test<10>();
}

TYPED_TEST(VectorArrayConverterTest, array_12) {
  // N > 8 and N is a multiple of 4
  TestFixture::template emit_test<12>();
}

TYPED_TEST(VectorArrayConverterTest, array_16) {
  // N > 8 and N is a multiple of 8
  TestFixture::template emit_test<16>();
}

TYPED_TEST(VectorArrayConverterTest, array_17) {
  // N > 8 and N is not a multiple of 8
  TestFixture::template emit_test<17>();
}

TYPED_TEST(VectorArrayConverterTest, array_27) {
  // Test entire conversion range with residue (for int4)
  TestFixture::template emit_test<27>();
}

TYPED_TEST(VectorArrayConverterTest, array_31) {
  // Force use of converters for 16, 8, 4, 2 and scalar 
  // if max width is 16
  TestFixture::template emit_test<31>();
}

TYPED_TEST(VectorArrayConverterTest, array_63) {
  // Force use of converters for 32, 16, 8, 4, 2 and scalar 
  // if max width is 32
  TestFixture::template emit_test<63>();
}

TYPED_TEST(VectorArrayConverterTest, array_256) {
  // Test entire conversion range (for int8)
  TestFixture::template emit_test<256>();
}

TYPED_TEST(VectorArrayConverterTest, array_259) {
  // Force use of 4, 2 and scalar converter (if max width is 4)
  TestFixture::template emit_test<259>();
}

TYPED_TEST(VectorArrayConverterTest, array_263) {
  // Force use of 8, 4, 2 and scalar converter (if max width is 8)
  TestFixture::template emit_test<263>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
