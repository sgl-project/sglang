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

#include "cutlass_unit_test.h"

#include <iostream>
#include <iomanip>
#include <utility>

#include <cute/container/array_subbyte.hpp>
#include <cute/tensor.hpp>
#include <cute/numeric/numeric_types.hpp>

TEST(CuTe_core, ArraySubbyte)
{
  using namespace cute;
  {
    array_subbyte<int4_t, 10> array0{};
    array_subbyte<int4_t,  5> array1{};
    fill(array0, int4_t(0));
    fill(array1, int4_t(1));

    for (size_t i = 0; i < array1.size(); ++i) {
      array0[i+5] = array1[i];
    }

    EXPECT_EQ(int4_t(array0.back()), int4_t(1));

    for (size_t i = 0; i < array1.size(); ++i) {
      EXPECT_EQ(int4_t(array0[i]), int4_t(int(i) / 5));
    }
  }

  {
  array_subbyte<uint8_t, 14> a{};

  //std::cout << sizeof_bits<decltype(a)>::value << std::endl;
  EXPECT_EQ(cute::sizeof_bits_v<decltype(a)>, 14*8);

  fill(a, uint8_t(13));
  for (int i = 0; i < int(a.size()); ++i) {
    //std::cout << i << ": " << int(a[i]) << " -> ";
    EXPECT_EQ(a[i], uint8_t(13));
    a[i] = uint8_t(i);
    //std::cout << int(a[i]) << std::endl;
    EXPECT_EQ(a[i], uint8_t(i));
  }

  //std::cout << std::endl;
  }

  {
  array_subbyte<int4_t, 14> a{};

  //std::cout << sizeof_bits<decltype(a)>::value << std::endl;
  EXPECT_EQ(cute::sizeof_bits_v<decltype(a)>, 14/2*8);

  fill(a, int4_t(-5));
  for (int i = 0; i < int(a.size()); ++i) {
    //std::cout << i << ": " << int4_t(a[i]) << " -> ";
    EXPECT_EQ(int4_t(a[i]), int4_t(-5));
    a[i] = int4_t(i);
    //std::cout << int4_t(a[i]) << std::endl;
    EXPECT_EQ(int4_t(a[i]), int4_t(i));
  }

  //std::cout << std::endl;
  }

  {
  array_subbyte<uint2_t, 14> a{};

  //std::cout << sizeof_bits<decltype(a)>::value << std::endl;
  EXPECT_EQ(cute::sizeof_bits_v<decltype(a)>, 4*8);

  fill(a, uint2_t(-5));
  for (int i = 0; i < int(a.size()); ++i) {
    //std::cout << i << ": " << uint2_t(a[i]) << " -> ";
    EXPECT_EQ(uint2_t(a[i]), uint2_t(-5));
    a[i] = uint2_t(i);
    //std::cout << uint2_t(a[i]) << std::endl;
    EXPECT_EQ(uint2_t(a[i]), uint2_t(i));
  }

  //std::cout << std::endl;
  }

  {
  array_subbyte<bool, 14> a{};

  //std::cout << sizeof_bits<decltype(a)>::value << std::endl;
  EXPECT_EQ(cute::sizeof_bits_v<decltype(a)>, 2*8);

  fill(a, bool(1));
  for (int i = 0; i < int(a.size()); ++i) {
    //std::cout << i << ": " << bool(a[i]) << " -> ";
    EXPECT_EQ(a[i], bool(1));
    a[i] = bool(i % 2);
    //std::cout << bool(a[i]) << std::endl;
    EXPECT_EQ(a[i], bool(i % 2));
  }
  //std::cout << std::endl;
  }
}

TEST(CuTe_core, Subbyte_iterator)
{
  using namespace cute;

  {
  array_subbyte<uint8_t, 15> a{};
  auto tensor = make_tensor(a.begin(), make_shape(15));

  fill(a, uint8_t(13));
  for (int i = 0; i < int(a.size()); ++i) {
    EXPECT_EQ(uint8_t(tensor(i)), 13);
    tensor(i) = uint8_t(i);
    EXPECT_EQ(a[i], uint8_t(tensor(i)));
  }

  }

  {
  array_subbyte<uint6b_t, 15> a{};
  auto tensor = make_tensor(a.begin(), make_shape(15));

  fill(a, uint6b_t(13));
  for (int i = 0; i < int(a.size()); ++i) {
    EXPECT_EQ(uint6b_t(tensor(i)), uint6b_t(13));
    tensor(i) = uint6b_t(i);
    EXPECT_EQ(uint6b_t(a[i]), uint6b_t(tensor(i)));
  }

  }

  {
  array_subbyte<int4_t, 15> a{};
  auto tensor = make_tensor(a.begin(), make_shape(15));

  fill(a, int4_t(-5));
  for (int i = 0; i < int(a.size()); ++i) {
    EXPECT_EQ(int4_t(tensor(i)), int4_t(-5));
    tensor(i) = int4_t(i);
    EXPECT_EQ(int4_t(a[i]), int4_t(tensor(i)));
  }

  }

  {
  array_subbyte<uint2_t, 15> a{};
  auto tensor = make_tensor(a.begin(), make_shape(15));

  fill(a, uint2_t(-5));
  for (int i = 0; i < int(a.size()); ++i) {
    EXPECT_EQ(uint2_t(tensor(i)), uint2_t(-5));
    tensor(i) = uint2_t(i);
    EXPECT_EQ(uint2_t(a[i]), uint2_t(tensor(i)));
  }

  }

  {
  array_subbyte<bool, 15> a{};
  auto tensor = make_tensor(a.begin(), make_shape(15));

  fill(a, bool(1));
  for (int i = 0; i < int(a.size()); ++i) {
    EXPECT_EQ(bool(tensor(i)), bool(1));
    tensor(i) = bool(i % 2);
    EXPECT_EQ(a[i], bool(tensor(i)));
  }
  }
}

TEST(CuTe_core, Const_subbyte_iterator)
{
  using namespace cute;

  {
  array_subbyte<uint8_t, 15> a{};
  auto tensor = make_tensor(a.begin(), make_shape(15));

  fill(a, uint8_t(13));
  for (int i = 0; i < int(a.size()); ++i) {
    EXPECT_EQ(uint8_t(tensor(i)), 13);
    a[i] = uint8_t(i);
    EXPECT_EQ(a[i], uint8_t(tensor(i)));
  }

  }

  {
  array_subbyte<int4_t, 15> a{};
  auto tensor = make_tensor(a.begin(), make_shape(15));

  fill(a, int4_t(-5));
  for (int i = 0; i < int(a.size()); ++i) {
    EXPECT_EQ(int4_t(tensor(i)), int4_t(-5));
    a[i] = int4_t(i);
    EXPECT_EQ(int4_t(a[i]), int4_t(tensor(i)));
  }

  }

  {
  array_subbyte<uint2_t, 15> a{};
  auto tensor = make_tensor(a.begin(), make_shape(15));

  fill(a, uint2_t(-5));
  for (int i = 0; i < int(a.size()); ++i) {
    EXPECT_EQ(uint2_t(tensor(i)), uint2_t(-5));
    a[i] = uint2_t(i);
    EXPECT_EQ(uint2_t(a[i]), uint2_t(tensor(i)));
  }

  }

  {
  array_subbyte<bool, 15> a{};
  auto tensor = make_tensor(a.begin(), make_shape(15));

  fill(a, bool(1));
  for (int i = 0; i < int(a.size()); ++i) {
    EXPECT_EQ(bool(tensor(i)), bool(1));
    a[i] = bool(i % 2);
    EXPECT_EQ(a[i], bool(tensor(i)));
  }
  }
}
