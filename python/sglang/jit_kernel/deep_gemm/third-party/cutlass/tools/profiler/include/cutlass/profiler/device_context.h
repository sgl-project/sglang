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
   \brief
*/

#pragma once

#include <map>
#include <string>


#include "cutlass/library/library.h"
#include "cutlass/library/util.h"

#include "options.h"
#include "device_allocation.h"

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Collection of allocations on the device
class DeviceContext {
public:

  //
  // Type definitions
  //
  using AllocationMap = std::map<std::string, DeviceAllocation *>;

private:
  //
  // Data members
  //

  /// Memory allocations that exist (owning)
  DeviceAllocationList device_memory_;

  /// Non-owning set of named allocations
  AllocationMap allocations_;

public:

  /// Allocates memory of a given type, capacity (elements), and name
  DeviceAllocation *allocate_block(
    Options const &options,
    std::string const &name,
    library::NumericTypeID type,
    size_t capacity,
    size_t device_index);

  /// Allocates memory of a given type, capacity (elements), and name
  DeviceAllocation *allocate_tensor(
    Options const &options,
    std::string const &name,
    library::NumericTypeID type,
    library::LayoutTypeID layout_id,
    std::vector<int> const &extent,
    std::vector<int64_t> const &stride,
    int batch_count,
    size_t device_index);

  /// Allocates memory of a given type, capacity (elements), and name
  DeviceAllocation *allocate_and_initialize_tensor(
    Options const &options,
    std::string const &name,
    library::NumericTypeID type,
    library::LayoutTypeID layout_id,
    std::vector<int> const &extent,
    std::vector<int64_t> const &stride,
    int batch_count,
    int seed_shift,
    size_t device_index);

  /// Allocates memory for sparse meta data
  DeviceAllocation *allocate_and_initialize_sparsemeta_tensor(
    Options const &options,
    std::string const &name,
    library::NumericTypeID type,
    library::LayoutTypeID layout_id,
    library::NumericTypeID type_a,
    std::vector<int> const &extent,
    std::vector<int64_t> const &stride,
    int batch_count,
    int seed_shift,
    size_t device_index);

  /// Clears named allocations (but does not necessarily free memory)
  void clear();

  /// Frees all device memory allocations
  void free();

  /// Gets the allocation by name
  DeviceAllocation &at(std::string const &name);

  size_t size() const;

  AllocationMap::iterator begin();
  AllocationMap::iterator end();
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass
