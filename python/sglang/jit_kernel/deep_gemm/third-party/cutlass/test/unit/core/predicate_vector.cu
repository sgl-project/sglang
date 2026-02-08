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

#include <cstring>

#include "../common/cutlass_unit_test.h"

#include "cutlass/predicate_vector.h"
#include "cutlass/util/host_tensor.h"

namespace test {

template <typename PredicateVector>
__global__ void load_predicates(unsigned *output, unsigned const *input) {

  PredicateVector predicates;

  int const word_count = (PredicateVector::kPredicates + 31) / 32;

  int i = 0;
  for (int word_idx = 0; word_idx < word_count; ++word_idx) {
    unsigned word = input[word_idx];

    CUTLASS_PRAGMA_UNROLL
    for (int bit = 0; bit < sizeof(unsigned) * 8; ++bit) {
      bool pred = ((word >> bit) & 1);
      predicates.set(i, pred);

      if (predicates.at(i) != pred) {
        printf("ERROR - cannot read back predicate\n");
      }
      ++i;
    }
  }


  __syncthreads();

  i = 0;
  for (int word_idx = 0; word_idx < word_count; ++word_idx) {

    unsigned result = 0;
    for (int bit = 0; bit < sizeof(unsigned) * 8; ++bit) {
      bool pred = predicates.at(i ++);
      result |= (unsigned(pred) << bit);
    }
    output[word_idx] = result;
  }
}
}

TEST(PredicateVector, Basic) {

  static int const Bits = 32;
  static int const Words = (Bits + 31) / 32;

  typedef cutlass::PredicateVector<Bits> PredicateVector;

  cutlass::HostTensor<unsigned, cutlass::IdentityTensorLayout<1> > output;
  cutlass::HostTensor<unsigned, cutlass::IdentityTensorLayout<1>> input;

  output.reserve(Words);
  input.reserve(Words);

  // some arbitrary test bits
  unsigned values[] = {
    0xdeadbeef,
    0xa0070032,
    0x9076d001,
    0x00000000,
    0xabdfc0ad
  };

  for (int test = 0; test < 5; ++test) {

    input.host_data(0) = values[test];
    output.host_data(0) = 0;

    input.sync_device();
    output.sync_device();

    test::load_predicates<PredicateVector><<<
      dim3(1,1,1), dim3(1,1,1)
    >>>(
      output.device_data(),
      input.device_data()
    );

    output.sync_host();

    for (int word = 0; word < Words; ++word) {
      EXPECT_EQ(input.host_data(word), output.host_data(word))
        << "Expected: 0x" << std::hex << input.host_data(word)
        << ", got: 0x" << output.host_data(word)
        << std::dec;
    }
  }
}

TEST(PredicateVector, Count) {

  {
    typedef cutlass::PredicateVector<4, 8> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 1)
        << "PredicateVector<4, 8> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<4, 4> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 1)
        << "PredicateVector<4, 4> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<4, 2> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 1)
        << "PredicateVector<4, 2> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<4, 1> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 1)
        << "PredicateVector<4, 1> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<8, 8> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 1)
        << "PredicateVector<8, 8> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<8, 4> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 1)
        << "PredicateVector<8, 4> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<8, 2> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 1)
        << "PredicateVector<8, 2> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<8, 1> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 2)
        << "PredicateVector<8, 1> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<16, 8> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 1)
        << "PredicateVector<16, 8> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<16, 4> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 1)
        << "PredicateVector<16, 4> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<16, 2> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 2)
        << "PredicateVector<16, 2> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<16, 1> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 4)
        << "PredicateVector<16, 1> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<32, 8> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 1)
        << "PredicateVector<32, 8> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<32, 4> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 2)
        << "PredicateVector<32, 4> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<32, 2> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 4)
        << "PredicateVector<32, 2> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<32, 1> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 8)
        << "PredicateVector<32, 1> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<64, 8> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 2)
        << "PredicateVector<64, 8> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<64, 4> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 4)
        << "PredicateVector<64, 4> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<64, 2> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 8)
        << "PredicateVector<64, 2> word count: " << int(PredicateVector::kWordCount);
  }

  {
    typedef cutlass::PredicateVector<64, 1> PredicateVector;
    EXPECT_EQ(int(PredicateVector::kWordCount), 16)
        << "PredicateVector<64, 1> word count: " << int(PredicateVector::kWordCount);
  }
}
