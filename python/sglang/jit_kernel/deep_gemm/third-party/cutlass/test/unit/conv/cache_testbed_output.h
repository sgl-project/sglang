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
  \brief Helper to construct cached name for
*/
#pragma once

#include <typeinfo>
#include <fstream>
#include <list>
#include <utility>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"

#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/core_io.h"
#include "cutlass/util/tensor_view_io.h"

#include "thrust/universal_vector.h"

#ifndef CUTLASS_TEST_ENABLE_CACHED_RESULTS
#define CUTLASS_TEST_ENABLE_CACHED_RESULTS false
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test::conv::device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result of a test
struct CachedTestKey {

  std::string op;         ///< Concatenated string representation of operation performed
  std::string problem;    ///< Concatenated string representation of problem description
  std::string types;      ///< Concatenated string representation of operand types
  uint32_t    A;          ///< Hashed result of tensor A
  uint32_t    B;          ///< Hashed result of tensor B
  uint32_t    C;          ///< Hashed result of tensor C

  //
  // Methods
  //
  inline CachedTestKey(): A(), B(), C() { }

  inline CachedTestKey(
    std::string op,         ///< Concatenated string representation of operation performed
    std::string problem,    ///< Concatenated string representation of problem description
    std::string types,      ///< Concatenated string representation of operand types
    uint32_t    A,          ///< Hashed result of tensor A
    uint32_t    B,          ///< Hashed result of tensor B
    uint32_t    C           ///< Hashed result of tensor C
  ):
    op(op), problem(problem), types(types), A(A), B(B), C(C)
  { }

  /// Checks for equality of the problem
  bool operator==(CachedTestKey const &rhs) const {
    return op == rhs.op && problem == rhs.problem && types == rhs.types && A == rhs.A && B == rhs.B && C == rhs.C;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

inline std::istream &operator>>(std::istream &in, CachedTestKey &result) {

  in >> result.op;
  in >> result.problem;
  in >> result.types;
  in >> result.A;
  in >> result.B;
  in >> result.C;

  return in;
}

inline std::ostream &operator<<(std::ostream &out, CachedTestKey const &result) {

  out << result.op << " ";
  out << result.problem << " ";
  out << result.types << " ";
  out << result.A << " ";
  out << result.B << " ";
  out << result.C << " ";

  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

struct CachedTestResult {
  uint32_t D;
  //
  // Methods
  //

  CachedTestResult(): D()
      { }

  CachedTestResult(uint32_t D): D(D)
      { }

  operator bool() const {
    return bool(D);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

inline std::istream &operator>>(std::istream &in, CachedTestResult &result) {
  in >> result.D;
  return in;
}

inline std::ostream &operator<<(std::ostream &out, CachedTestResult const &result) {
  out << result.D;
  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

struct CachedTestResultListing {

  std::list<std::pair<CachedTestKey, CachedTestResult>> results;

  //
  // Methods
  //

  inline CachedTestResultListing(std::string const &path) {
    std::ifstream file(path);

    while (file.good()) {
      CachedTestKey key;
      file >> key;

      CachedTestResult result;
      file >> result;

      if (result) {
        results.push_back(std::make_pair(key, result));  
      }
    }
  }

  /// Returns the cached result 
  std::pair<bool, CachedTestResult> find(CachedTestKey const &rhs) const {
    for (auto const & result : results) {
      if (result.first == rhs) {
        return std::make_pair(true, result.second);
      }
    }
    return std::make_pair(false, CachedTestResult());
  }

  /// Appends an entry
  void append(CachedTestKey const &key, CachedTestResult const &result) {
    if (result) {
      results.push_back(std::make_pair(key, result));  
    }
  }

  /// Writes the entire listing to a file
  bool write(std::string const &path) {
    std::ofstream file(path);
    if (!file.good()) {
      return false;
    }

    for (auto const &result : results) {
      file << result.first << result.second << std::endl;
    }

    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element>
struct ScalarEncoder {
  Element scalar;

  ScalarEncoder(Element s): scalar(s) { }

  std::string str() const {
    std::stringstream ss;
    Element s = scalar;
    if (s < Element()) {
      s = -s;
      ss << "n";
    }
    ss << s;
    return ss.str();
  }
};

template <typename Element>
ScalarEncoder<Element> EncodeScalar(Element a) {
  return ScalarEncoder<Element>(a);
}

template <typename Element>
struct ScalarEncoder<cutlass::complex<Element>> {
  cutlass::complex<Element> scalar;

  ScalarEncoder(cutlass::complex<Element> s): scalar(s) { }

  std::string str() const {
    std::stringstream ss;
    ss << EncodeScalar<Element>(scalar.real()) << "_" << EncodeScalar<Element>(scalar.imag()) << "i";
    return ss.str();
  }
};

template <typename Element>
std::ostream &operator<<(std::ostream &out, ScalarEncoder<Element> const &scalar) {
  out << scalar.str();
  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

inline char const *EncodeOperator(cutlass::conv::Operator conv_op) {
    switch (conv_op) {
      case cutlass::conv::Operator::kFprop: return "fprop";
      case cutlass::conv::Operator::kDgrad: return "dgrad";
      case cutlass::conv::Operator::kWgrad: return "wgrad";
      case cutlass::conv::Operator::kDeconv: return "deconv";
    }
    return "conv_unknown";
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// Encode GemmCoord (Gemm problem size)
inline std::ostream &EncodeProblemSize(
  std::ostream &out, 
  cutlass::gemm::GemmCoord const &problem) {
    
  out << problem.m() << "x" << problem.n() << "x" << problem.k() << "_";

  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Encode Conv2dProblemSize
inline std::ostream &EncodeProblemSize(
  std::ostream &out, 
  cutlass::conv::Conv2dProblemSize const &problem) {
    
  out << problem.N << "x" << problem.H << "x" << problem.W << "x" << problem.C << "_" 
    << problem.P << "x" << problem.Q << "_" << problem.K << "x" << problem.R << "x" << problem.S << "_";

  out << "pad_h" << problem.pad_h << "w" << problem.pad_w << "_";
  out << "stride_h" << problem.stride_h << "w" << problem.stride_w << "_";
  out << "dil_h" << problem.dilation_h << "w" << problem.dilation_w << "_";

  switch (problem.mode) {
    case cutlass::conv::Mode::kCrossCorrelation:
        out << "corr";
        break;
    case cutlass::conv::Mode::kConvolution:
        out << "conv";
        break;
  }

  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// Encode Conv3dProblemSize
inline std::ostream &EncodeProblemSize(
  std::ostream &out, 
  cutlass::conv::Conv3dProblemSize const &problem) {
    
  out << problem.N << "x" << problem.D << "x" << problem.H << "x" << problem.W << "x" << problem.C << "_" 
    << problem.Z << problem.P << "x" << problem.Q << "_" << problem.K << "x" << problem.R << "x" << problem.S << "_";

  out << "pad_d" << problem.pad_h << "h" << problem.pad_h << "w" << problem.pad_w << "_";
  out << "stride_d" << problem.stride_d << "h" << problem.stride_h << "w" << problem.stride_w << "_";
  out << "dil_d" << problem.dilation_d << "h" << problem.dilation_h << "w" << problem.dilation_w << "_";

  switch (problem.mode) {
    case cutlass::conv::Mode::kCrossCorrelation:
        out << "corr";
        break;
    case cutlass::conv::Mode::kConvolution:
        out << "conv";
        break;
  }

  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Encode 3.x ConvNd ProblemShape
template <class ProblemShape>
inline std::ostream &EncodeProblemSize(
  std::ostream &out, 
  ProblemShape const& problem_shape) {

  out << problem_shape.shape_A << "_";
  out << problem_shape.shape_B << "_";

  out << "padl" << problem_shape.lower_padding << "_";
  out << "padu" << problem_shape.upper_padding << "_";
  out << "str"  << problem_shape.traversal_stride << "_";
  out << "dil"  << problem_shape.dilation << "_";

  switch (problem_shape.mode) {
    case cutlass::conv::Mode::kCrossCorrelation:
        out << "corr";
        break;
    case cutlass::conv::Mode::kConvolution:
        out << "conv";
        break;
  }

  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element>
inline std::string ElementTypeName() {
  return std::string(typeid(Element).name());
}

template <>
inline std::string ElementTypeName<cutlass::half_t>() {
  return "h";
}

template <>
inline std::string ElementTypeName<cutlass::complex<cutlass::half_t>>() {
  return "ch";
}

template <>
inline std::string ElementTypeName<cutlass::bfloat16_t>() {
  return "bf16";
}

template <>
inline std::string ElementTypeName<cutlass::complex<cutlass::bfloat16_t>>() {
  return "cbf16";
}

template <>
inline std::string ElementTypeName<cutlass::tfloat32_t>() {
  return "tf32";
}

template <>
inline std::string ElementTypeName<cutlass::complex<cutlass::tfloat32_t>>() {
  return "ctf32";
}

template <>
inline std::string ElementTypeName<cutlass::complex<float>>() {
  return "c";
}

template <>
inline std::string ElementTypeName<cutlass::complex<double>>() {
  return "z";
}

template <>
inline std::string ElementTypeName<cutlass::Quaternion<float>>() {
  return "q";
}

template <>
inline std::string ElementTypeName<int8_t>() {
  return "s8";
}

template <>
inline std::string ElementTypeName<uint8_t>() {
  return "u8";
}

template <>
inline std::string ElementTypeName<cutlass::int4b_t>() {
  return "s4";
}

template <>
inline std::string ElementTypeName<cutlass::uint4b_t>() {
  return "u4";
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
inline std::string LayoutTypeName() {
  return std::string(typeid(Layout).name());
}

template <>
inline std::string LayoutTypeName<cutlass::layout::ColumnMajor>() {
  return "n";
}

template <>
inline std::string LayoutTypeName<cutlass::layout::RowMajor>() {
  return "t";
}

template <>
inline std::string LayoutTypeName<cutlass::layout::TensorNHWC>() {
  return "nhwc";
}

template <>
inline std::string LayoutTypeName<cutlass::layout::TensorNCxHWx<32>>() {
  return "nc32hw32";
}

template <>
inline std::string LayoutTypeName<cutlass::layout::TensorNCxHWx<64>>() {
  return "nc64hw64";
}

template <>
inline std::string LayoutTypeName<cutlass::layout::TensorCxRSKx<32>>() {
  return "c32rsk32";
}

template <>
inline std::string LayoutTypeName<cutlass::layout::TensorCxRSKx<64>>() {
  return "c64rsk64";
}

template <>
inline std::string LayoutTypeName<cutlass::layout::TensorNDHWC>() {
  return "ndhwc";
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element, typename Layout>
inline std::string TensorTypeName() {
  std::stringstream ss;
  ss << ElementTypeName<Element>() << LayoutTypeName<Layout>();
  return ss.str();
}

template <typename Element>
inline std::string TensorTypeName() {
  std::stringstream ss;
  ss << ElementTypeName<Element>();
  return ss.str();
}
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Hash function on a byte array
struct CRC32 {

  uint32_t table[256];

  //
  // Methods
  //

  CRC32() {

    uint32_t rem;
    int i, j;
   
    for (i = 0; i < 256; i++) {
      rem = i;
      for (j = 0; j < 8; j++) {
        if (rem & 1) {
          rem >>= 1;
          rem ^= 0xedb88320;
        } else
          rem >>= 1;
      }
      table[i] = rem;
    }
  }

  /// Computes the CRC of an array of bytes
  uint32_t operator()(void const *start, size_t length, uint32_t crc = uint32_t()) const {
    uint8_t const *p = static_cast<uint8_t const *>(start);
    uint8_t const *q = static_cast<uint8_t const *>(start) + length;

    crc = ~crc;
    
    for (; p != q; ++p) {
      uint8_t octet = *p;
      crc = (crc >> 8) ^ table[(crc & 0xff) ^ octet];
    }

    return ~crc;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Element, typename Layout
>
uint32_t TensorHash(
  cutlass::TensorView<Element, Layout> view, 
  CRC32 const &hash = CRC32(), 
  uint32_t crc = uint32_t()
) {

  return hash(view.data(), view.capacity() * cutlass::sizeof_bits<Element>::value / 8, crc);
}

template <typename Element>
uint32_t TensorHash(
  thrust::universal_vector<Element>& tensor,
  CRC32 const &hash = CRC32(), 
  uint32_t crc = uint32_t()
) {

  return hash(tensor.data().get(), tensor.size() * cutlass::sizeof_bits<Element>::value / 8, crc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA, typename LayoutA,
  typename ElementB, typename LayoutB,
  typename ElementC, typename LayoutC,
  typename ElementAccumulator,
  typename ElementCompute
>
inline std::ostream &EncodeTypes(
  std::ostream &out
) {
  
  out << TensorTypeName<ElementA, LayoutA>() << "_" 
    << TensorTypeName<ElementB, LayoutB>() << "_" 
    << TensorTypeName<ElementC, LayoutC>() << "_"
    << ElementTypeName<ElementAccumulator>() << "_"
    << ElementTypeName<ElementCompute>();

  return out;
}

template <
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename ElementD
>
inline std::ostream &EncodeTypes(
  std::ostream &out
) {
  
  out << TensorTypeName<ElementA>() << "_" 
      << TensorTypeName<ElementB>() << "_" 
      << TensorTypeName<ElementC>() << "_"
      << ElementTypeName<ElementD>();

  return out;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA, typename LayoutA,
  typename ElementB, typename LayoutB,
  typename ElementC, typename LayoutC,
  typename ElementAccumulator,
  typename ElementCompute
>
inline CachedTestKey CreateCachedGemmTestKey(
  cutlass::gemm::GemmCoord const &problem, 
  ElementCompute alpha,
  ElementCompute beta,
  cutlass::TensorView<ElementA, LayoutA> A,
  cutlass::TensorView<ElementB, LayoutB> B,
  cutlass::TensorView<ElementC, LayoutC> C
) {

  CachedTestKey key;

  // Encode gemm operator and problem sizes
  key.op = "gemm";

  std::stringstream ss_problem;
  EncodeProblemSize(ss_problem, problem);
  ss_problem << "_alpha" << EncodeScalar(alpha) << "_beta" << EncodeScalar(beta);
  key.problem = ss_problem.str();

  // Encode problem data types
  std::stringstream ss_types;
  EncodeTypes<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        ElementCompute>(ss_types);
  key.types = ss_types.str();

  // Encode hash for problem data
  CRC32 crc_hash;
  key.A = TensorHash(A, crc_hash);
  key.B = TensorHash(B, crc_hash);
  key.C = TensorHash(C, crc_hash);

  return key;
}

/////////////////////////////////////////////////////////////////////////////////////////////////


template <
  typename ElementA, typename LayoutA,
  typename ElementB, typename LayoutB,
  typename ElementC, typename LayoutC,
  typename ElementAccumulator,
  typename ElementCompute
>
inline CachedTestKey CreateCachedConv2dTestKey(

  cutlass::conv::Operator conv_operator,
  cutlass::conv::Conv2dProblemSize const &problem, 
  ElementCompute alpha,
  ElementCompute beta,
  cutlass::TensorView<ElementA, LayoutA> A,
  cutlass::TensorView<ElementB, LayoutB> B,
  cutlass::TensorView<ElementC, LayoutC> C
) {

  CachedTestKey key;

  // Encode conv2d operator and problem sizes
  key.op = "conv2d";
  
  std::stringstream ss_problem;
  ss_problem << EncodeOperator(conv_operator) << "_";
  EncodeProblemSize(ss_problem, problem);
  ss_problem << "_alpha" << EncodeScalar(alpha) << "_beta" << EncodeScalar(beta);
  
  key.problem = ss_problem.str();

  // Encode problem data types
  std::stringstream ss_types;
  EncodeTypes<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        ElementCompute>(ss_types);
  key.types = ss_types.str();

  // Encode hash for problem data
  CRC32 crc_hash;

  key.A = TensorHash(A, crc_hash);
  key.B = TensorHash(B, crc_hash);
  key.C = TensorHash(C, crc_hash);

  return key;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA, typename LayoutA,
  typename ElementB, typename LayoutB,
  typename ElementC, typename LayoutC,
  typename ElementAccumulator,
  typename ElementCompute
>
inline CachedTestKey CreateCachedConv2dWithBroadcastTestKey(

  cutlass::conv::Operator conv_operator,
  cutlass::conv::Conv2dProblemSize const &problem, 
  ElementCompute alpha,
  ElementCompute beta,
  cutlass::TensorView<ElementA, LayoutA> A,
  cutlass::TensorView<ElementB, LayoutB> B,
  cutlass::TensorView<ElementC, LayoutC> C
) {

  CachedTestKey key;

  // Encode conv2d operator and problem sizes
  key.op = "conv2d_with_broadcast";
  
  std::stringstream ss_problem;
  ss_problem << EncodeOperator(conv_operator) << "_";
  EncodeProblemSize(ss_problem, problem);
  ss_problem << "_alpha" << EncodeScalar(alpha) << "_beta" << EncodeScalar(beta);
  
  key.problem = ss_problem.str();

  // Encode problem data types
  std::stringstream ss_types;
  EncodeTypes<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        ElementCompute>(ss_types);
  key.types = ss_types.str();

  // Encode hash for problem data
  CRC32 crc_hash;

  key.A = TensorHash(A, crc_hash);
  key.B = TensorHash(B, crc_hash);
  key.C = TensorHash(C, crc_hash);

  return key;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA, typename LayoutA,
  typename ElementB, typename LayoutB,
  typename ElementC, typename LayoutC,
  typename ElementAccumulator,
  typename ElementCompute
>
inline CachedTestKey CreateCachedConv2dWithReductionTestKey(

  cutlass::conv::Operator conv_operator,
  cutlass::conv::Conv2dProblemSize const &problem, 
  ElementCompute alpha,
  ElementCompute beta,
  cutlass::TensorView<ElementA, LayoutA> A,
  cutlass::TensorView<ElementB, LayoutB> B,
  cutlass::TensorView<ElementC, LayoutC> C
) {

  CachedTestKey key;

  // Encode conv2d operator and problem sizes
  key.op = "conv2d_with_reduction";
  
  std::stringstream ss_problem;
  ss_problem << EncodeOperator(conv_operator) << "_";
  EncodeProblemSize(ss_problem, problem);
  ss_problem << "_alpha" << EncodeScalar(alpha) << "_beta" << EncodeScalar(beta);
  
  key.problem = ss_problem.str();

  // Encode problem data types
  std::stringstream ss_types;
  EncodeTypes<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        ElementCompute>(ss_types);
  key.types = ss_types.str();

  // Encode hash for problem data
  CRC32 crc_hash;

  key.A = TensorHash(A, crc_hash);
  key.B = TensorHash(B, crc_hash);
  key.C = TensorHash(C, crc_hash);

  return key;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA, typename LayoutA,
  typename ElementB, typename LayoutB,
  typename ElementC, typename LayoutC,
  typename ElementAccumulator,
  typename ElementCompute
>
inline CachedTestKey CreateCachedConv3dTestKey(
  cutlass::conv::Operator conv_operator,
  cutlass::conv::Conv3dProblemSize const &problem, 
  ElementCompute alpha,
  ElementCompute beta,
  cutlass::TensorView<ElementA, LayoutA> A,
  cutlass::TensorView<ElementB, LayoutB> B,
  cutlass::TensorView<ElementC, LayoutC> C
) {

  CachedTestKey key;

  // Encode conv3d operator and problem sizes
  key.op = "conv3d";
  
  std::stringstream ss_problem;
  
  ss_problem << EncodeOperator(conv_operator) << "_";
  EncodeProblemSize(ss_problem, problem);
  ss_problem << "_alpha" << EncodeScalar(alpha) << "_beta" << EncodeScalar(beta);
  
  key.problem = ss_problem.str();

  // Encode problem data types
  std::stringstream ss_types;
  EncodeTypes<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        ElementCompute>(ss_types);
  key.types = ss_types.str();

  // Encode problem data
  CRC32 crc_hash;
  key.A = TensorHash(A, crc_hash);
  key.B = TensorHash(B, crc_hash);
  key.C = TensorHash(C, crc_hash);

  return key;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape,
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename ElementD
>
inline CachedTestKey CreateCachedConvNd3xTestKey(
  cutlass::conv::Operator conv_operator,
  ProblemShape const& problem_shape,
  double alpha,
  double beta,
  thrust::universal_vector<ElementA> A,
  thrust::universal_vector<ElementB> B,
  thrust::universal_vector<ElementC> C
) {

  CachedTestKey key;
 
  // Encode convNd operator and problem sizes
  std::stringstream ss_op;
  ss_op << "conv" << ProblemShape::RankS <<  "d";
  key.op = ss_op.str();

  std::stringstream ss_problem;
  ss_problem << EncodeOperator(conv_operator) << "_";
  EncodeProblemSize(ss_problem, problem_shape);
  ss_problem << "_alpha" << EncodeScalar(alpha) << "_beta" << EncodeScalar(beta);
  key.problem = ss_problem.str();

  // Encode problem data types
  std::stringstream ss_types;
  EncodeTypes<
        ElementA,
        ElementB,
        ElementC,
        ElementD>(ss_types);
  key.types = ss_types.str();

  // Encode problem data
  CRC32 crc_hash;
  key.A = TensorHash(A, crc_hash);
  key.B = TensorHash(B, crc_hash);
  key.C = TensorHash(C, crc_hash);

  return key;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace test::conv::device

/////////////////////////////////////////////////////////////////////////////////////////////////
