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
    \brief Manifest of CUTLASS Library

    This is the root of the data structure containing CUTLASS objects
*/

#pragma once

#include <list>
#include <memory>
#include <map>

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "library.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Forward declaration 
class Manifest;

// init and insert all cutlass gemm operations in manifest object (procedurally generated using generator.py)
void initialize_all(Manifest &manifest);         

// init and insert all reduction op in manifest object (manually instantiated in library/reduction)
void initialize_all_reduction_op(Manifest &manifest);

/////////////////////////////////////////////////////////////////////////////////////////////////////////

/// List of operations
using OperationVector = std::vector<std::unique_ptr<Operation>>;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Manifest of CUTLASS Library
class Manifest {
private:

  /// Operation provider 
  Provider provider_;

  /// Global list of operations
  OperationVector operations_;

public:
  Manifest (Provider provider = library::Provider::kCUTLASS) : provider_(provider) { }

  /// Top-level initialization
  Status initialize();

  /// Used for initialization
  void reserve(size_t operation_count);

  /// Graceful shutdown
  Status release();

  /// Appends an operation and takes ownership
  void append(Operation *operation_ptr) {\
    // This function is inline s.t. it is present in generated libraries
    // without having to compile or link in manifest.cpp
    operations_.emplace_back(operation_ptr);
  }

  /// Returns an iterator to the first operation
  OperationVector const &operations() const;

  /// Returns a const iterator
  OperationVector::const_iterator begin() const;

  /// Returns a const iterator
  OperationVector::const_iterator end() const;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
