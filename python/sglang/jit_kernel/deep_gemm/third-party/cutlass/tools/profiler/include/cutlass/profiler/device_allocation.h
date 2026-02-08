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
   \brief Execution environment
*/

#pragma once

#include <stdexcept>
#include <list>
#include <vector>

#include "cutlass/library/library.h"
#include "cutlass/util/distribution.h"

#include "enumerated_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Device memory allocation
class DeviceAllocation {
private:

  /// Data type of contained elements
  library::NumericTypeID type_;

  /// Gets the stride between elements
  size_t batch_stride_;

  /// Capacity in elements of device allocation
  size_t capacity_;

  /// Pointer to device memory
  void *pointer_;

  /// Layout type ID
  library::LayoutTypeID layout_;

  /// Stride vector
  std::vector<int64_t> stride_;

  /// Extent vector
  std::vector<int> extent_;

  /// Support allocating a 'batch' of non-overlapping tensors in contiguous memory
  int batch_count_;

  /// Buffer holding TensorRef instance to recently allocated memory
  std::vector<uint8_t> tensor_ref_buffer_;

  /// The device ID where the allocation is made
  int device_;

public:
  //
  // Static member functions
  //

  /// Determines the number of bytes needed to represent this numeric type
  static size_t bytes(library::NumericTypeID type, size_t capacity);

  /// Returns the stride of a packed layout
  static std::vector<int64_t> get_packed_layout(
    library::LayoutTypeID layout_id,
    std::vector<int> const &extent);

  /// returns the capacity needed
  static size_t construct_layout(
    void *bytes,
    library::LayoutTypeID layout_id,
    std::vector<int> const &extent,
    std::vector<int64_t> &stride);

  /// Returns true if two blocks have exactly the same value
  static bool block_compare_equal(
    library::NumericTypeID numeric_type,
    void const *ptr_A,
    void const *ptr_B,
    size_t capacity);

  /// Returns true if two blocks have approximately the same value
  static bool block_compare_relatively_equal(
    library::NumericTypeID numeric_type,
    void const *ptr_A,
    void const *ptr_B,
    size_t capacity,
    double epsilon,
    double nonzero_floor);

public:
  //
  // Methods
  //

  DeviceAllocation();

  DeviceAllocation(
    library::NumericTypeID type,
    size_t capacity,
    int device = -1);

  DeviceAllocation(
    library::NumericTypeID type,
    library::LayoutTypeID layout_id,
    std::vector<int> const &extent,
    std::vector<int64_t> const &stride = std::vector<int64_t>(),
    int batch_count = 1,
    int device = -1);

  ~DeviceAllocation();

  DeviceAllocation &reset();

  /// Allocates device memory of a given type and capacity
  DeviceAllocation &reset(library::NumericTypeID type, size_t capacity);

  /// Allocates memory for a given layout and tensor
  DeviceAllocation &reset(
    library::NumericTypeID type,
    library::LayoutTypeID layout_id,
    std::vector<int> const &extent,
    std::vector<int64_t> const &stride = std::vector<int64_t>(),
    int batch_count = 1);

  /// Returns a buffer owning the tensor reference
  std::vector<uint8_t> &tensor_ref() {
    return tensor_ref_buffer_;
  }

  bool good() const;

  /// Data type of contained elements
  library::NumericTypeID type() const;

  /// Pointer to start of device memory allocation
  void *data() const;

  /// Pointer to the first element of a batch
  void *batch_data(int batch_idx) const;

  /// Gets the layout type
  library::LayoutTypeID layout() const;

  /// Gets the stride vector
  std::vector<int64_t> const & stride() const;

  /// Gets the extent vector
  std::vector<int> const & extent() const;

  /// Gets the number of adjacent tensors in memory
  int batch_count() const;

  /// Gets the stride (in units of elements) between items
  int64_t batch_stride() const;

  /// Gets the stride (in units of bytes) between items
  int64_t batch_stride_bytes() const;

  /// Capacity of allocation in number of elements
  size_t capacity() const;

  /// Capacity of allocation in bytes
  size_t bytes() const;

  /// Initializes a device allocation to a random distribution using cuRAND
  void initialize_random_device(int seed, Distribution dist);

  /// Initializes a host allocation to a random distribution using std::cout
  void initialize_random_host(int seed, Distribution dist);

  /// Initializes a device allocation to a sequential distribution
  void initialize_sequential_device(Distribution dist);

  /// Initializes a host allocation to a sequential distribution
  void initialize_sequential_host(Distribution dist);

  /// Initializes a device allocation to a random distribution using cuRAND
  void initialize_random_sparsemeta_device(int seed, int MetaSizeInBits);

  /// Initializes a host allocation to a random distribution using std::cout
  void initialize_random_sparsemeta_host(int seed, int MetaSizeInBits);

  /// Uniformly fills a tensor with a value when provided o.w. zero
  void fill_device(double value);

  /// Uniformly fills a host allocation with a value when provided o.w. zero
  void fill_host(double value);

  /// Copies from an equivalent-sized tensor in device memory
  void copy_from_device(void const *ptr);

  /// Copies from an equivalent-sized tensor in device memory
  void copy_from_host(void const *ptr);

  /// Copies from an equivalent-sized tensor in device memory
  void copy_to_host(void *ptr);

  /// Writes a tensor to csv
  void write_tensor_csv(std::ostream &out);

private:
  /// A wrapper that sets the device, performs malloc, and sets back
  cudaError_t malloc(void** ptr, size_t size);
};

using DeviceAllocationList = std::list<DeviceAllocation>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
