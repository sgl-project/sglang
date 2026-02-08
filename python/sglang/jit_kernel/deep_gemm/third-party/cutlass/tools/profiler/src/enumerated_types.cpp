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

#include "cutlass/profiler/enumerated_types.h"

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  ExecutionMode enumerant;
}
ExecutionMode_enumerants[] = {
  {"profile", "Profile", ExecutionMode::kProfile},
  {"dry_run", "Dry run", ExecutionMode::kDryRun},
  {"dry", "dry run", ExecutionMode::kDryRun},
  {"trace", "Trace", ExecutionMode::kTrace},
  {"enumerate", "Enumerate", ExecutionMode::kEnumerate}
};

/// Converts a ExecutionMode enumerant to a string
char const *to_string(ExecutionMode mode, bool pretty) {

  for (auto const & possible : ExecutionMode_enumerants) {
    if (mode == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

/// Parses a ExecutionMode enumerant from a string
template <>
ExecutionMode from_string<ExecutionMode>(std::string const &str) {

  for (auto const & possible : ExecutionMode_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return ExecutionMode::kInvalid;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  AlgorithmMode enumerant;
}
AlgorithmMode_enumerants[] = {
  {"matching", "Matching", AlgorithmMode::kMatching},
  {"best", "Best", AlgorithmMode::kBest},
  {"default", "Default", AlgorithmMode::kDefault}
};

/// Converts a ExecutionMode enumerant to a string
char const *to_string(AlgorithmMode mode, bool pretty) {

  for (auto const & possible : AlgorithmMode_enumerants) {
    if (mode == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

/// Parses a ExecutionMode enumerant from a string
template <>
AlgorithmMode from_string<AlgorithmMode>(std::string const &str) {

  for (auto const & possible : AlgorithmMode_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return AlgorithmMode::kInvalid;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  Disposition enumerant;
}
Disposition_enumerants[] = {
  {"passed", "Passed", Disposition::kPassed},
  {"failed", "Failed", Disposition::kFailed},
  {"not_run", "Not run", Disposition::kNotRun},
  {"not_verified", "Not verified", Disposition::kNotVerified},
  {"invalid_problem", "Invalid problem", Disposition::kInvalidProblem},
  {"not_supported", "Not supported", Disposition::kNotSupported},
  {"incorrect", "Incorrect", Disposition::kIncorrect}
};

/// Converts a Disposition enumerant to a string
char const *to_string(Disposition disposition, bool pretty) {

  for (auto const & possible : Disposition_enumerants) {
    if (disposition == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }
  
  return pretty ? "Invalid" : "invalid";
}

/// Parses a Disposition enumerant from a string
template <>
Disposition from_string<Disposition>(std::string const &str) {

  for (auto const & possible : Disposition_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return Disposition::kInvalid;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  SaveWorkspace enumerant;
}
SaveWorkspace_enumerants[] = {
  {"never", "Never", SaveWorkspace::kNever},
  {"incorrect", "Incorrect", SaveWorkspace::kIncorrect},
  {"always", "Always", SaveWorkspace::kAlways}
};

/// Converts a SaveWorkspace enumerant to a string
char const *to_string(SaveWorkspace save_option, bool pretty) {

  for (auto const & possible : SaveWorkspace_enumerants) {
    if (save_option == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }
  
  return pretty ? "Invalid" : "invalid";
}

/// Parses a SaveWorkspace enumerant from a string
template <>
SaveWorkspace from_string<SaveWorkspace>(std::string const &str) {

  for (auto const & possible : SaveWorkspace_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return SaveWorkspace::kInvalid;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  ArgumentTypeID enumerant;
}
ArgumentTypeID_enumerants[] = {
  {"scalar", "Scalar", ArgumentTypeID::kScalar},
  {"int", "Integer", ArgumentTypeID::kInteger},
  {"tensor", "Tensor", ArgumentTypeID::kTensor},
  {"batched_tensor", "BatchedTensor", ArgumentTypeID::kBatchedTensor},
  {"struct", "Struct", ArgumentTypeID::kStructure},
  {"enum", "Enumerated type", ArgumentTypeID::kEnumerated}
};

/// Converts a ArgumentTypeID enumerant to a string
char const *to_string(ArgumentTypeID type, bool pretty) {

  for (auto const & possible : ArgumentTypeID_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

/// Parses a ArgumentTypeID enumerant from a string
template <>
ArgumentTypeID from_string<ArgumentTypeID>(std::string const &str) {

  for (auto const & possible : ArgumentTypeID_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return ArgumentTypeID::kInvalid;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

