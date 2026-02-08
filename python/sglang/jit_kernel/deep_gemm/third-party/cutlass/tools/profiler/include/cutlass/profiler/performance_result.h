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
   \brief Defines a math function
*/

#pragma once

#include <vector>

#include "cutlass/cutlass.h"

// CUTLASS Profiler includes
#include "enumerated_types.h"

// CUTLASS Library includes
#include "cutlass/library/library.h"

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Performance result object
struct PerformanceResult {

  /// Index of problem
  size_t problem_index;

  /// library::Provider
  library::Provider provider;

  /// Operation kind
  library::OperationKind op_kind;

  /// CUTLASS status result from kernels (success or failure)
  // Status does information on verification
  Status status;

  /// Outcome of verification (worst case verification result)
  Disposition disposition;
  
  /// Outcome of verification (all verification results)
  DispositionMap verification_map;

  /// Operation name
  std::string operation_name;

  /// Stringified vector of argument values
  std::vector<std::pair<std::string, std::string> > arguments;

  /// Number of bytes read or written
  int64_t bytes;

  /// Number of DL flops performed by the math function
  int64_t flops;

  /// Average runtime in ms
  double runtime;

  /// Average runtime in ms per device
  std::vector<double> runtime_vector;

  //
  // Members
  //

  /// Ctor
  PerformanceResult(): 
    problem_index(0),
    op_kind(library::OperationKind::kInvalid),
    provider(library::Provider::kInvalid), 
    disposition(Disposition::kNotRun),
    status(Status::kInvalid),
    bytes(0), 
    flops(0), 
    runtime(0)
  { }

  // Copy constructor for deep copy
  PerformanceResult(const PerformanceResult& other) = default;

  // Explicitly define copy assignment operator
  PerformanceResult& operator=(const PerformanceResult& other) = default;

  /// Returns true if the runtime is valid
  bool good() const {
    return runtime > 0;
  }

  /// Math throughput in units of GFLOP/s
  double gflops_per_sec() const {
    return double(flops) / runtime / 1.0e6;
  }

  /// memory bandwidth in units of GiB/s
  double gbytes_per_sec() const {
    return double(bytes) / double(1 << 30) / runtime * 1000.0;
  }

};

using PerformanceResultVector = std::vector<PerformanceResult>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

