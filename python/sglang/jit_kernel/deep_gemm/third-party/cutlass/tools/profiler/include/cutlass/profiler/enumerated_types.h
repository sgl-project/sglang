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
   \brief Provides several functions for filling tensors with data.
*/

#pragma once

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include "cutlass/library/library.h"

#define TRACE(x) { std::cout << __FILE__ << ":" << __LINE__ << "  " << x << std::endl; }

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
T from_string(std::string const &);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumerated type describing how the performance testbench evaluates kernels.
enum class ExecutionMode {
  kProfile,     ///< regular verification and profiling
  kDryRun,      ///< no kernels are launched or workspaces allocated; used to assess what operators might be launched
  kEnumerate,   ///< no kernels launched or workspaces allocated; lists all operation kind and operations
  kTrace,       ///< executes a single device-side computation with no other kernel launches
  kInvalid
};

/// Converts a ExecutionMode enumerant to a string
char const *to_string(ExecutionMode mode, bool pretty = false);

/// Parses a ExecutionMode enumerant from a string
template <>
ExecutionMode from_string<ExecutionMode>(std::string const &str);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Library algorithm mode
enum class AlgorithmMode {
  kMatching,            ///< compare against best matching algorithm
  kBest,                    ///< evaluate all library algorithms and report best
  kDefault,                 ///< use the library's default algorithm option
  kInvalid
};

/// Converts a ExecutionMode enumerant to a string
char const *to_string(AlgorithmMode mode, bool pretty = false);

/// Parses a ExecutionMode enumerant from a string
template <>
AlgorithmMode from_string<AlgorithmMode>(std::string const &str);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Outcome of a performance test
enum class Disposition {
  kPassed,
  kFailed,    // kernel itself reported an error
  kNotRun,
  kIncorrect, // kernel finished without a detected error, but result does not equal expected result
  kNotVerified,
  kInvalidProblem,
  kNotSupported,
  kInvalid
};

/// Converts a Disposition enumerant to a string
char const *to_string(Disposition disposition, bool pretty = false);

/// Parses a Disposition enumerant from a string
template <>
Disposition from_string<Disposition>(std::string const &str);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Indicates when to save 
enum class SaveWorkspace {
  kNever,
  kIncorrect,
  kAlways,
  kInvalid
};

/// Converts a SaveWorkspace enumerant to a string
char const *to_string(SaveWorkspace save_option, bool pretty = false);

/// Parses a SaveWorkspace enumerant from a string
template <>
SaveWorkspace from_string<SaveWorkspace>(std::string const &str);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Indicates the type of kernel argument
// ArgumentType can be both ScalarType or NumericType. Thus, enums kScalar and kNumeric
// 1) kScalar: e.g. of a Scalar ArgumentType is u32 is a Scalar type.
// Its c++ equivalent as "type name = initializer" is "u32 m = 32"
// 2) kNumeric: e.g. of a Numeric ArgumentType is NumericTypeID is a Numeric type.
// Its c++ equivalent as "type name = initializer" is "NumericTypeID numeric_type = u32"
enum class ArgumentTypeID {
  kScalar,
  kInteger,
  kTensor,
  kBatchedTensor,
  kStructure,
  kEnumerated,
  kInvalid
};

/// Converts a ArgumentTypeID enumerant to a string
char const *to_string(ArgumentTypeID type, bool pretty = false);

/// Parses a ArgumentTypeID enumerant from a string
template <>
ArgumentTypeID from_string<ArgumentTypeID>(std::string const &str);

/////////////////////////////////////////////////////////////////////////////////////////////////
// Profiler typedefs
using ProviderVector = std::vector<library::Provider>;
using DispositionMap = std::map<library::Provider, Disposition>;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Print vector for the report
template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    out << to_string(v[i], true) << (i + 1u != v.size() ? "," : "");
  }
  return out;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass
