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

#include <memory>
#include "cutlass/library/manifest.h"

namespace cutlass {
namespace library {

//////////////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_reference_operations(Manifest &manifest);

//////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Top-level initialization
Status Manifest::initialize() {

  if (!operations_.empty()) {
    operations_.clear();
  }

  // initialize procedurally generated cutlass op in manifest object
  initialize_all(*this);

  // initialize manually instanced reference op in manifest object
  initialize_reference_operations(*this);

  // initialize manually instanced reduction reference op in manifest object
  initialize_all_reduction_op(*this);

  return Status::kSuccess;
}

/// Used for initialization
void Manifest::reserve(size_t operation_count) {
  operations_.reserve(operation_count);
}

/// Graceful shutdown
Status Manifest::release() {
  operations_.clear();
  return Status::kSuccess;
}

/// Returns an iterator to the first operation
OperationVector const & Manifest::operations() const {
  return operations_;
}

/// Returns a const iterator
OperationVector::const_iterator Manifest::begin() const {
  return operations_.begin();
}

/// Returns a const iterator
OperationVector::const_iterator Manifest::end() const {
  return operations_.end();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
