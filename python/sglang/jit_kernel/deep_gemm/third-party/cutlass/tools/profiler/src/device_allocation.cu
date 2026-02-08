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

#include <cstring>

#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"

#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/library/util.h"

#include "cutlass/profiler/device_allocation.h"

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

size_t DeviceAllocation::bytes(library::NumericTypeID type, size_t capacity) {
  return size_t(cutlass::library::sizeof_bits(type)) * capacity / 8;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
static std::vector<int64_t> get_packed_layout_stride(std::vector<int> const &extent) {

  typename Layout::TensorCoord extent_coord;
  typename Layout::Stride stride_coord;

  if (extent.size() != size_t(Layout::kRank)) {
    throw std::runtime_error("Layout does not have same rank as extent vector.");
  }

  for (int i = 0; i < Layout::kRank; ++i) {
    extent_coord[i] = extent.at(i);
  }

  std::vector<int64_t> stride;
  stride.resize(Layout::kStrideRank, 0);

  Layout layout = Layout::packed(extent_coord);
  stride_coord = layout.stride();

  for (int i = 0; i < Layout::kStrideRank; ++i) {
    stride.at(i) = (int64_t)stride_coord[i];
  }

  return stride;
}

/// Returns the stride of a packed layout
std::vector<int64_t> DeviceAllocation::get_packed_layout(
  library::LayoutTypeID layout_id,
  std::vector<int> const &extent) {

  std::vector<int64_t> stride;

  switch (layout_id) {
    case library::LayoutTypeID::kColumnMajor:
      stride = get_packed_layout_stride<cutlass::layout::ColumnMajor>(extent);
      break;
    case library::LayoutTypeID::kRowMajor:
      stride = get_packed_layout_stride<cutlass::layout::RowMajor>(extent);
      break;
    case library::LayoutTypeID::kColumnMajorInterleavedK2:
      stride = get_packed_layout_stride<cutlass::layout::ColumnMajorInterleaved<2>>(extent);
      break;
    case library::LayoutTypeID::kRowMajorInterleavedK2:
      stride = get_packed_layout_stride<cutlass::layout::RowMajorInterleaved<2>>(extent);
      break;
    case library::LayoutTypeID::kColumnMajorInterleavedK4:
      stride = get_packed_layout_stride<cutlass::layout::ColumnMajorInterleaved<4>>(extent);
      break;
    case library::LayoutTypeID::kRowMajorInterleavedK4:
      stride = get_packed_layout_stride<cutlass::layout::RowMajorInterleaved<4>>(extent);
      break;
    case library::LayoutTypeID::kColumnMajorInterleavedK16:
      stride = get_packed_layout_stride<cutlass::layout::ColumnMajorInterleaved<16>>(extent);
      break;
    case library::LayoutTypeID::kRowMajorInterleavedK16:
      stride = get_packed_layout_stride<cutlass::layout::RowMajorInterleaved<16>>(extent);
      break;
    case library::LayoutTypeID::kColumnMajorInterleavedK32:
      stride = get_packed_layout_stride<cutlass::layout::ColumnMajorInterleaved<32>>(extent);
      break;
    case library::LayoutTypeID::kRowMajorInterleavedK32:
      stride = get_packed_layout_stride<cutlass::layout::RowMajorInterleaved<32>>(extent);
      break;
    case library::LayoutTypeID::kColumnMajorInterleavedK64:
      stride = get_packed_layout_stride<cutlass::layout::ColumnMajorInterleaved<64>>(extent);
      break;
    case library::LayoutTypeID::kRowMajorInterleavedK64:
      stride = get_packed_layout_stride<cutlass::layout::RowMajorInterleaved<64>>(extent);
      break;
    case library::LayoutTypeID::kTensorNCHW:
      stride = get_packed_layout_stride<cutlass::layout::TensorNCHW>(extent);
      break;
    case library::LayoutTypeID::kTensorNHWC:
      stride = get_packed_layout_stride<cutlass::layout::TensorNHWC>(extent);
      break;
    case library::LayoutTypeID::kTensorNDHWC:
      stride = get_packed_layout_stride<cutlass::layout::TensorNDHWC>(extent);
      break;
    case library::LayoutTypeID::kTensorNC32HW32:
      stride = get_packed_layout_stride<cutlass::layout::TensorNCxHWx<32>>(extent);
      break;
    case library::LayoutTypeID::kTensorNC64HW64:
      stride = get_packed_layout_stride<cutlass::layout::TensorNCxHWx<64>>(extent);
      break;
    case library::LayoutTypeID::kTensorC32RSK32:
      stride = get_packed_layout_stride<cutlass::layout::TensorCxRSKx<32>>(extent);
      break;
    case library::LayoutTypeID::kTensorC64RSK64:
      stride = get_packed_layout_stride<cutlass::layout::TensorCxRSKx<64>>(extent);
      break;
    default: break;
  }

  return stride;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template to use CUTLASS Layout functions to
template <typename Layout>
static size_t construct_layout_(
  void *bytes,
  library::LayoutTypeID layout_id,
  std::vector<int> const &extent,
  std::vector<int64_t> &stride) {

  if (extent.size() != Layout::kRank) {
    throw std::runtime_error(
      "Layout must have same rank as extent vector.");
  }

  if (Layout::kStrideRank && stride.empty()) {

    stride = get_packed_layout_stride<Layout>(extent);

    return construct_layout_<Layout>(
      bytes,
      layout_id,
      extent,
      stride);
  }
  else if (Layout::kStrideRank && stride.size() != Layout::kStrideRank) {
    throw std::runtime_error(
      "Layout requires either empty stride or stride vector matching Layout::kStrideRank");
  }

  typename Layout::Stride stride_coord;
  for (int i = 0; i < Layout::kStrideRank; ++i) {
    stride_coord[i] = (int)stride.at(i);
  }

  typename Layout::TensorCoord extent_coord;
  for (int i = 0; i < Layout::kRank; ++i) {
    extent_coord[i] = extent.at(i);
  }

  // Construct the CUTLASS layout object from the stride object
  Layout layout(stride_coord);

  // Pack it into bytes
  if (bytes) {
    *reinterpret_cast<Layout *>(bytes) = layout;
  }

  // Return capacity
  size_t capacity_ = layout.capacity(extent_coord);

  return capacity_;
}

/// returns the capacity needed
size_t DeviceAllocation::construct_layout(
  void *bytes,
  library::LayoutTypeID layout_id,
  std::vector<int> const &extent,
  std::vector<int64_t> &stride) {

  switch (layout_id) {
    case library::LayoutTypeID::kColumnMajor:
      return construct_layout_<cutlass::layout::ColumnMajor>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kRowMajor:
      return construct_layout_<cutlass::layout::RowMajor>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kColumnMajorInterleavedK2:
      return construct_layout_<cutlass::layout::ColumnMajorInterleaved<2>>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kRowMajorInterleavedK2:
      return construct_layout_<cutlass::layout::RowMajorInterleaved<2>>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kColumnMajorInterleavedK4:
      return construct_layout_<cutlass::layout::ColumnMajorInterleaved<4>>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kRowMajorInterleavedK4:
      return construct_layout_<cutlass::layout::RowMajorInterleaved<4>>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kColumnMajorInterleavedK16:
      return construct_layout_<cutlass::layout::ColumnMajorInterleaved<16>>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kRowMajorInterleavedK16:
      return construct_layout_<cutlass::layout::RowMajorInterleaved<16>>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kColumnMajorInterleavedK32:
      return construct_layout_<cutlass::layout::ColumnMajorInterleaved<32>>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kRowMajorInterleavedK32:
      return construct_layout_<cutlass::layout::RowMajorInterleaved<32>>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kColumnMajorInterleavedK64:
      return construct_layout_<cutlass::layout::ColumnMajorInterleaved<64>>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kRowMajorInterleavedK64:
      return construct_layout_<cutlass::layout::RowMajorInterleaved<64>>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kTensorNCHW:
      return construct_layout_<cutlass::layout::TensorNHWC>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kTensorNHWC:
      return construct_layout_<cutlass::layout::TensorNHWC>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kTensorNDHWC:
      return construct_layout_<cutlass::layout::TensorNDHWC>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kTensorNC32HW32:
      return construct_layout_<cutlass::layout::TensorNCxHWx<32>>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kTensorNC64HW64:
      return construct_layout_<cutlass::layout::TensorNCxHWx<64>>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kTensorC32RSK32:
      return construct_layout_<cutlass::layout::TensorCxRSKx<32>>(bytes, layout_id, extent, stride);

    case library::LayoutTypeID::kTensorC64RSK64:
      return construct_layout_<cutlass::layout::TensorCxRSKx<64>>(bytes, layout_id, extent, stride);

    default: break;
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

DeviceAllocation::DeviceAllocation():
  type_(library::NumericTypeID::kInvalid),
  batch_stride_(0),
  capacity_(0),
  pointer_(nullptr),
  layout_(library::LayoutTypeID::kUnknown),
  batch_count_(1) {
  cudaGetDevice(&device_);
}

DeviceAllocation::DeviceAllocation(
  library::NumericTypeID type,
  size_t capacity,
  int device
):
  type_(type), batch_stride_(capacity), capacity_(capacity), pointer_(nullptr),
  layout_(library::LayoutTypeID::kUnknown), batch_count_(1), device_(device) {

  cudaError_t result = this->malloc((void **)&pointer_, bytes(type, capacity));

  if (result != cudaSuccess) {
    type_ = library::NumericTypeID::kInvalid;
    capacity_ = 0;
    pointer_ = nullptr;
    throw std::bad_alloc();
  }
}

DeviceAllocation::DeviceAllocation(
  library::NumericTypeID type,
  library::LayoutTypeID layout_id,
  std::vector<int> const &extent,
  std::vector<int64_t> const &stride,
  int batch_count,
  int device
):
  type_(type), batch_stride_(size_t(0)), capacity_(size_t(0)),
  pointer_(nullptr), batch_count_(1), device_(device) {

  reset(type, layout_id, extent, stride, batch_count);
}

DeviceAllocation::~DeviceAllocation() {
  if (pointer_) {
    int current_device;
    cudaGetDevice(&current_device);

    if (current_device != device_) {
      cudaSetDevice(device_);
    }
    cudaFree(pointer_);

    if (current_device != device_) {
      cudaSetDevice(current_device);
    }
  }
}

DeviceAllocation &DeviceAllocation::reset() {
  if (pointer_) {
    int current_device;
    cudaGetDevice(&current_device);

    if (current_device != device_) {
      cudaSetDevice(device_);
    }
    cudaFree(pointer_);

    if (current_device != device_) {
      cudaSetDevice(current_device);
    }
  }

  type_ = library::NumericTypeID::kInvalid;
  batch_stride_ = 0;
  capacity_ = 0;
  pointer_ = nullptr;
  layout_ = library::LayoutTypeID::kUnknown;
  stride_.clear();
  extent_.clear();
  tensor_ref_buffer_.clear();
  batch_count_ = 1;

  return *this;
}

DeviceAllocation &DeviceAllocation::reset(library::NumericTypeID type, size_t capacity) {

  reset();

  type_ = type;
  batch_stride_ = capacity;
  capacity_ = capacity;

  cudaError_t result = this->malloc((void **)&pointer_, bytes(type_, capacity_));
  if (result != cudaSuccess) {
    throw std::bad_alloc();
  }

  layout_ = library::LayoutTypeID::kUnknown;
  stride_.clear();
  extent_.clear();
  batch_count_ = 1;

  tensor_ref_buffer_.resize(sizeof(pointer_), 0);
  std::memcpy(tensor_ref_buffer_.data(), &pointer_, sizeof(pointer_));

  return *this;
}

/// Allocates memory for a given layout and tensor
DeviceAllocation &DeviceAllocation::reset(
  library::NumericTypeID type,
  library::LayoutTypeID layout_id,
  std::vector<int> const &extent,
  std::vector<int64_t> const &stride,
  int batch_count) {

  reset();

  tensor_ref_buffer_.resize(sizeof(pointer_) + (sizeof(int64_t) * library::get_layout_stride_rank(layout_id)), 0);

  type_ = type;

  layout_ = layout_id;
  stride_ = stride;
  extent_ = extent;
  batch_count_ = batch_count;

  batch_stride_ = construct_layout(
    tensor_ref_buffer_.data() + sizeof(pointer_),
    layout_id,
    extent,
    stride_);

  capacity_ = batch_stride_ * batch_count_;

  cudaError_t result = this->malloc((void **)&pointer_, bytes(type, capacity_));
  if (result != cudaSuccess) {
    throw std::bad_alloc();
  }

  std::memcpy(tensor_ref_buffer_.data(), &pointer_, sizeof(pointer_));

  return *this;
}

bool DeviceAllocation::good() const {
  return (capacity_ && pointer_);
}

library::NumericTypeID DeviceAllocation::type() const {
  return type_;
}

void *DeviceAllocation::data() const {
  return pointer_;
}

void *DeviceAllocation::batch_data(int batch_idx) const {
    return static_cast<char *>(data()) + batch_stride_bytes() * batch_idx;
}

library::LayoutTypeID DeviceAllocation::layout() const {
  return layout_;
}

std::vector<int64_t> const & DeviceAllocation::stride() const {
  return stride_;
}

/// Gets the extent vector
std::vector<int> const & DeviceAllocation::extent() const {
  return extent_;
}

/// Gets the number of adjacent tensors in memory
int DeviceAllocation::batch_count() const {
  return batch_count_;
}

/// Gets the stride (in units of elements) between items
int64_t DeviceAllocation::batch_stride() const {
  return batch_stride_;
}

/// Gets the stride (in units of bytes) between items
int64_t DeviceAllocation::batch_stride_bytes() const {
  return bytes(type_, batch_stride_);
}

size_t DeviceAllocation::capacity() const {
  return capacity_;
}

size_t DeviceAllocation::bytes() const {
  return bytes(type_, capacity_);
}

/// Copies from an equivalent-sized tensor in device memory
void DeviceAllocation::copy_from_device(void const *ptr) {
  if (!bytes()) {
#ifndef NDEBUG
    std::cout << "Skipping copy of size 0 allocation\n";
#endif
    return;
  }

  cudaError_t result = cudaMemcpy(data(), ptr, bytes(), cudaMemcpyDeviceToDevice);
  if (result != cudaSuccess) {
    throw std::runtime_error("Failed device-to-device copy");
  }
}

/// Copies from an equivalent-sized tensor in device memory
void DeviceAllocation::copy_from_host(void const *ptr) {
  if (!bytes()) {
#ifndef NDEBUG
    std::cout << "Skipping copy of size 0 allocation\n";
#endif
    return;
  }

  cudaError_t result = cudaMemcpy(data(), ptr, bytes(), cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    throw std::runtime_error("Failed host-to-device copy");
  }
}

/// Copies from an equivalent-sized tensor in device memory
void DeviceAllocation::copy_to_host(void *ptr) {
  if (!bytes()) {
#ifndef NDEBUG
    std::cout << "Skipping copy of size 0 allocation\n";
#endif
    return;
  }

  cudaError_t result = cudaMemcpy(ptr, data(), bytes(), cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    throw std::runtime_error("Failed device-to-host copy");
  }
}

void DeviceAllocation::initialize_random_device(int seed, Distribution dist) {
  if (!bytes()) {
#ifndef NDEBUG
    std::cout << "Skipping initialization of size 0 allocation\n";
#endif
    return;
  }

  if (!data()) {
    throw std::runtime_error("Attempting to initialize invalid allocation.");
  }

  // Instantiate calls to CURAND here. This file takes a long time to compile for
  // this reason.

  switch (type_) {
  case library::NumericTypeID::kF16:
    cutlass::reference::device::BlockFillRandom<cutlass::half_t>(
      reinterpret_cast<cutlass::half_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kBF16:
    cutlass::reference::device::BlockFillRandom<cutlass::bfloat16_t>(
      reinterpret_cast<cutlass::bfloat16_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kTF32:
    cutlass::reference::device::BlockFillRandom<cutlass::tfloat32_t>(
      reinterpret_cast<cutlass::tfloat32_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kF32:
    cutlass::reference::device::BlockFillRandom<float>(
      reinterpret_cast<float *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kCBF16:
    cutlass::reference::device::BlockFillRandom<complex<bfloat16_t>>(
      reinterpret_cast<complex<bfloat16_t> *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kCTF32:
    cutlass::reference::device::BlockFillRandom<cutlass::complex<cutlass::tfloat32_t>>(
      reinterpret_cast<cutlass::complex<cutlass::tfloat32_t> *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kCF32:
    cutlass::reference::device::BlockFillRandom<cutlass::complex<float>>(
      reinterpret_cast<cutlass::complex<float> *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kFE4M3:
    cutlass::reference::device::BlockFillRandom<cutlass::float_e4m3_t>(
      reinterpret_cast<cutlass::float_e4m3_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kFE5M2:
    cutlass::reference::device::BlockFillRandom<cutlass::float_e5m2_t>(
      reinterpret_cast<cutlass::float_e5m2_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kFUE4M3:
    cutlass::reference::device::BlockFillRandom<cutlass::float_ue4m3_t>(
      reinterpret_cast<cutlass::float_ue4m3_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kFUE8M0:
    cutlass::reference::device::BlockFillRandom<cutlass::float_ue8m0_t>(
      reinterpret_cast<cutlass::float_ue8m0_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kFE2M3:
    cutlass::reference::device::BlockFillRandom<cutlass::float_e2m3_t>(
      reinterpret_cast<cutlass::float_e2m3_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kFE3M2:
    cutlass::reference::device::BlockFillRandom<cutlass::float_e3m2_t>(
      reinterpret_cast<cutlass::float_e3m2_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kFE2M1:
    cutlass::reference::device::BlockFillRandom<cutlass::float_e2m1_t>(
      reinterpret_cast<cutlass::float_e2m1_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kF64:
    cutlass::reference::device::BlockFillRandom<double>(
      reinterpret_cast<double *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kCF64:
    cutlass::reference::device::BlockFillRandom<complex<double>>(
      reinterpret_cast<complex<double> *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kS2:
    cutlass::reference::device::BlockFillRandom<int2b_t>(
      reinterpret_cast<int2b_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kS4:
    cutlass::reference::device::BlockFillRandom<int4b_t>(
      reinterpret_cast<int4b_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kS8:
    cutlass::reference::device::BlockFillRandom<int8_t>(
      reinterpret_cast<int8_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kS16:
    cutlass::reference::device::BlockFillRandom<int16_t>(
      reinterpret_cast<int16_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kS32:
    cutlass::reference::device::BlockFillRandom<int32_t>(
      reinterpret_cast<int32_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kS64:
    cutlass::reference::device::BlockFillRandom<int64_t>(
      reinterpret_cast<int64_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kB1:
    cutlass::reference::device::BlockFillRandom<uint1b_t>(
      reinterpret_cast<uint1b_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kU2:
    cutlass::reference::device::BlockFillRandom<uint2b_t>(
      reinterpret_cast<uint2b_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kU4:
    cutlass::reference::device::BlockFillRandom<uint4b_t>(
      reinterpret_cast<uint4b_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kU8:
    cutlass::reference::device::BlockFillRandom<uint8_t>(
      reinterpret_cast<uint8_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kU16:
    cutlass::reference::device::BlockFillRandom<uint16_t>(
      reinterpret_cast<uint16_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kU32:
    cutlass::reference::device::BlockFillRandom<uint32_t>(
      reinterpret_cast<uint32_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kU64:
    cutlass::reference::device::BlockFillRandom<uint64_t>(
      reinterpret_cast<uint64_t *>(pointer_),
      capacity_,
      seed,
      dist
    );
    break;
  default: break;
  }
}

void DeviceAllocation::initialize_random_host(int seed, Distribution dist) {
  if (!bytes()) {
#ifndef NDEBUG
    std::cout << "Skipping initialization of size 0 allocation\n";
#endif
    return;
  }

  if (!data()) {
    throw std::runtime_error("Attempting to initialize invalid allocation.");
  }

  std::vector<uint8_t> host_data(bytes());

  switch (type_) {
  case library::NumericTypeID::kFE4M3:
    cutlass::reference::host::BlockFillRandom<cutlass::float_e4m3_t>(
      reinterpret_cast<cutlass::float_e4m3_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kFE5M2:
    cutlass::reference::host::BlockFillRandom<cutlass::float_e5m2_t>(
      reinterpret_cast<cutlass::float_e5m2_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  
  case library::NumericTypeID::kFUE4M3:
    cutlass::reference::host::BlockFillRandom<cutlass::float_ue4m3_t>(
      reinterpret_cast<cutlass::float_ue4m3_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  
  case library::NumericTypeID::kFE2M3:
    cutlass::reference::host::BlockFillRandom<cutlass::float_e2m3_t>(
      reinterpret_cast<cutlass::float_e2m3_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kFE3M2:
    cutlass::reference::host::BlockFillRandom<cutlass::float_e3m2_t>(
      reinterpret_cast<cutlass::float_e3m2_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kFE2M1:
    cutlass::reference::host::BlockFillRandom<cutlass::float_e2m1_t>(
      reinterpret_cast<cutlass::float_e2m1_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kFUE8M0:
    cutlass::reference::host::BlockFillRandom<cutlass::float_ue8m0_t>(
      reinterpret_cast<cutlass::float_ue8m0_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kF16:
    cutlass::reference::host::BlockFillRandom<cutlass::half_t>(
      reinterpret_cast<cutlass::half_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kBF16:
    cutlass::reference::host::BlockFillRandom<cutlass::bfloat16_t>(
      reinterpret_cast<cutlass::bfloat16_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kTF32:
    cutlass::reference::host::BlockFillRandom<cutlass::tfloat32_t>(
      reinterpret_cast<cutlass::tfloat32_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kF32:
    cutlass::reference::host::BlockFillRandom<float>(
      reinterpret_cast<float *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kCF16:
    cutlass::reference::host::BlockFillRandom<cutlass::complex<cutlass::half_t>>(
      reinterpret_cast<cutlass::complex<cutlass::half_t> *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kCBF16:
    cutlass::reference::host::BlockFillRandom<cutlass::complex<cutlass::bfloat16_t>>(
      reinterpret_cast<cutlass::complex<cutlass::bfloat16_t> *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kCTF32:
    cutlass::reference::host::BlockFillRandom<cutlass::complex<cutlass::tfloat32_t>>(
      reinterpret_cast<cutlass::complex<cutlass::tfloat32_t> *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kCF32:
    cutlass::reference::host::BlockFillRandom<cutlass::complex<float>>(
      reinterpret_cast<cutlass::complex<float> *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kF64:
    cutlass::reference::host::BlockFillRandom<double>(
      reinterpret_cast<double *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kCF64:
    cutlass::reference::host::BlockFillRandom<cutlass::complex<double>>(
      reinterpret_cast<cutlass::complex<double> *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kS2:
    cutlass::reference::host::BlockFillRandom<int2b_t>(
      reinterpret_cast<int2b_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kS4:
    cutlass::reference::host::BlockFillRandom<int4b_t>(
      reinterpret_cast<int4b_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kS8:
    cutlass::reference::host::BlockFillRandom<int8_t>(
      reinterpret_cast<int8_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kS16:
    cutlass::reference::host::BlockFillRandom<int16_t>(
      reinterpret_cast<int16_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kS32:
    cutlass::reference::host::BlockFillRandom<int32_t>(
      reinterpret_cast<int32_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kS64:
    cutlass::reference::host::BlockFillRandom<int64_t>(
      reinterpret_cast<int64_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kB1:
    cutlass::reference::host::BlockFillRandom<uint1b_t>(
      reinterpret_cast<uint1b_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kU2:
    cutlass::reference::host::BlockFillRandom<uint2b_t>(
      reinterpret_cast<uint2b_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kU4:
    cutlass::reference::host::BlockFillRandom<uint4b_t>(
      reinterpret_cast<uint4b_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kU8:
    cutlass::reference::host::BlockFillRandom<uint8_t>(
      reinterpret_cast<uint8_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kU16:
    cutlass::reference::host::BlockFillRandom<uint16_t>(
      reinterpret_cast<uint16_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kU32:
    cutlass::reference::host::BlockFillRandom<uint32_t>(
      reinterpret_cast<uint32_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  case library::NumericTypeID::kU64:
    cutlass::reference::host::BlockFillRandom<uint64_t>(
      reinterpret_cast<uint64_t *>(host_data.data()),
      capacity_,
      seed,
      dist
    );
    break;
  default: break;
  }

  copy_from_host(host_data.data());
}

void DeviceAllocation::initialize_sequential_device(Distribution dist) {
  if (!bytes()) {
#ifndef NDEBUG
    std::cout << "Skipping initialization of size 0 allocation\n";
#endif
    return;
  }

  if (!data()) {
    throw std::runtime_error("Attempting to initialize invalid allocation.");
  }

  switch (type_) {
  case library::NumericTypeID::kFE4M3:
    cutlass::reference::device::BlockFillSequential<cutlass::float_e4m3_t>(
      reinterpret_cast<cutlass::float_e4m3_t *>(pointer_),
      capacity_,
      static_cast<cutlass::float_e4m3_t>(dist.sequential.delta),
      static_cast<cutlass::float_e4m3_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kFE5M2:
    cutlass::reference::device::BlockFillSequential<cutlass::float_e5m2_t>(
      reinterpret_cast<cutlass::float_e5m2_t *>(pointer_),
      capacity_,
      static_cast<cutlass::float_e5m2_t>(dist.sequential.delta),
      static_cast<cutlass::float_e5m2_t>(dist.sequential.start)
    );
    break;
  
  case library::NumericTypeID::kFUE4M3:
    cutlass::reference::device::BlockFillSequential<cutlass::float_ue4m3_t>(
      reinterpret_cast<cutlass::float_ue4m3_t *>(pointer_),
      capacity_,
      static_cast<cutlass::float_ue4m3_t>(dist.sequential.delta),
      static_cast<cutlass::float_ue4m3_t>(dist.sequential.start)
    );
    break;
  
  case library::NumericTypeID::kFE2M3:
    cutlass::reference::device::BlockFillSequential<cutlass::float_e2m3_t>(
      reinterpret_cast<cutlass::float_e2m3_t *>(pointer_),
      capacity_,
      static_cast<cutlass::float_e2m3_t>(dist.sequential.delta),
      static_cast<cutlass::float_e2m3_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kFE3M2:
    cutlass::reference::device::BlockFillSequential<cutlass::float_e3m2_t>(
      reinterpret_cast<cutlass::float_e3m2_t *>(pointer_),
      capacity_,
      static_cast<cutlass::float_e3m2_t>(dist.sequential.delta),
      static_cast<cutlass::float_e3m2_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kFE2M1:
    cutlass::reference::device::BlockFillSequential<cutlass::float_e2m1_t>(
      reinterpret_cast<cutlass::float_e2m1_t *>(pointer_),
      capacity_,
      static_cast<cutlass::float_e2m1_t>(dist.sequential.delta),
      static_cast<cutlass::float_e2m1_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kFUE8M0:
    cutlass::reference::device::BlockFillSequential<cutlass::float_ue8m0_t>(
      reinterpret_cast<cutlass::float_ue8m0_t *>(pointer_),
      capacity_,
      static_cast<cutlass::float_ue8m0_t>(dist.sequential.delta),
      static_cast<cutlass::float_ue8m0_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kF16:
    cutlass::reference::device::BlockFillSequential<cutlass::half_t>(
      reinterpret_cast<cutlass::half_t *>(pointer_),
      capacity_,
      static_cast<cutlass::half_t>(dist.sequential.delta),
      static_cast<cutlass::half_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kBF16:
    cutlass::reference::device::BlockFillSequential<cutlass::bfloat16_t>(
      reinterpret_cast<cutlass::bfloat16_t *>(pointer_),
      capacity_,
      static_cast<cutlass::bfloat16_t>(dist.sequential.delta),
      static_cast<cutlass::bfloat16_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kTF32:
    cutlass::reference::device::BlockFillSequential<cutlass::tfloat32_t>(
      reinterpret_cast<cutlass::tfloat32_t *>(pointer_),
      capacity_,
      static_cast<cutlass::tfloat32_t>(dist.sequential.delta),
      static_cast<cutlass::tfloat32_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kF32:
    cutlass::reference::device::BlockFillSequential<float>(
      reinterpret_cast<float *>(pointer_),
      capacity_,
      static_cast<float>(dist.sequential.delta),
      static_cast<float>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kCF16:
    cutlass::reference::device::BlockFillSequential<cutlass::complex<cutlass::half_t>>(
      reinterpret_cast<cutlass::complex<cutlass::half_t> *>(pointer_),
      capacity_,
      cutlass::complex<cutlass::half_t>(
        static_cast<cutlass::half_t>(dist.sequential.delta)),
      cutlass::complex<cutlass::half_t>(
        static_cast<cutlass::half_t>(dist.sequential.start))
    );
    break;
  case library::NumericTypeID::kCBF16:
    cutlass::reference::device::BlockFillSequential<cutlass::complex<cutlass::bfloat16_t>>(
      reinterpret_cast<cutlass::complex<cutlass::bfloat16_t> *>(pointer_),
      capacity_,
      cutlass::complex<cutlass::bfloat16_t>(
        static_cast<cutlass::bfloat16_t>(dist.sequential.delta)),
      cutlass::complex<cutlass::bfloat16_t>(
        static_cast<cutlass::bfloat16_t>(dist.sequential.start))
    );
    break;
  case library::NumericTypeID::kCTF32:
    cutlass::reference::device::BlockFillSequential<cutlass::complex<cutlass::tfloat32_t>>(
      reinterpret_cast<cutlass::complex<cutlass::tfloat32_t> *>(pointer_),
      capacity_,
      cutlass::complex<cutlass::tfloat32_t>(
        static_cast<cutlass::tfloat32_t>(dist.sequential.delta)),
      cutlass::complex<cutlass::tfloat32_t>(
        static_cast<cutlass::tfloat32_t>(dist.sequential.start))
    );
    break;
  case library::NumericTypeID::kCF32:
    cutlass::reference::device::BlockFillSequential<cutlass::complex<float>>(
      reinterpret_cast<cutlass::complex<float> *>(pointer_),
      capacity_,
      cutlass::complex<float>(
        static_cast<float>(dist.sequential.delta)),
      cutlass::complex<float>(
        static_cast<float>(dist.sequential.start))
    );
    break;
  case library::NumericTypeID::kF64:
    cutlass::reference::device::BlockFillSequential<double>(
      reinterpret_cast<double *>(pointer_),
      capacity_,
      static_cast<double>(dist.sequential.delta),
      static_cast<double>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kCF64:
    cutlass::reference::device::BlockFillSequential<cutlass::complex<double>>(
      reinterpret_cast<cutlass::complex<double> *>(pointer_),
      capacity_,
      cutlass::complex<double>(
        static_cast<double>(dist.sequential.delta)),
      cutlass::complex<double>(
        static_cast<double>(dist.sequential.start))
    );
    break;
  case library::NumericTypeID::kS2:
    cutlass::reference::device::BlockFillSequential<int2b_t>(
      reinterpret_cast<int2b_t *>(pointer_),
      capacity_,
      static_cast<int2b_t>(dist.sequential.delta),
      static_cast<int2b_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kS4:
    cutlass::reference::device::BlockFillSequential<int4b_t>(
      reinterpret_cast<int4b_t *>(pointer_),
      capacity_,
      static_cast<int4b_t>(dist.sequential.delta),
      static_cast<int4b_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kS8:
    cutlass::reference::device::BlockFillSequential<int8_t>(
      reinterpret_cast<int8_t *>(pointer_),
      capacity_,
      static_cast<int8_t>(dist.sequential.delta),
      static_cast<int8_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kS16:
    cutlass::reference::device::BlockFillSequential<int16_t>(
      reinterpret_cast<int16_t *>(pointer_),
      capacity_,
      static_cast<int16_t>(dist.sequential.delta),
      static_cast<int16_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kS32:
    cutlass::reference::device::BlockFillSequential<int32_t>(
      reinterpret_cast<int32_t *>(pointer_),
      capacity_,
      static_cast<int32_t>(dist.sequential.delta),
      static_cast<int32_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kS64:
    cutlass::reference::device::BlockFillSequential<int64_t>(
      reinterpret_cast<int64_t *>(pointer_),
      capacity_,
      static_cast<int64_t>(dist.sequential.delta),
      static_cast<int64_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kB1:
    cutlass::reference::device::BlockFillSequential<uint1b_t>(
      reinterpret_cast<uint1b_t *>(pointer_),
      capacity_,
      static_cast<uint1b_t>(dist.sequential.delta),
      static_cast<uint1b_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kU2:
    cutlass::reference::device::BlockFillSequential<uint2b_t>(
      reinterpret_cast<uint2b_t *>(pointer_),
      capacity_,
      static_cast<uint2b_t>(dist.sequential.delta),
      static_cast<uint2b_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kU4:
    cutlass::reference::device::BlockFillSequential<uint4b_t>(
      reinterpret_cast<uint4b_t *>(pointer_),
      capacity_,
      static_cast<uint4b_t>(dist.sequential.delta),
      static_cast<uint4b_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kU8:
    cutlass::reference::device::BlockFillSequential<uint8_t>(
      reinterpret_cast<uint8_t *>(pointer_),
      capacity_,
      static_cast<uint8_t>(dist.sequential.delta),
      static_cast<uint8_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kU16:
    cutlass::reference::device::BlockFillSequential<uint16_t>(
      reinterpret_cast<uint16_t *>(pointer_),
      capacity_,
      static_cast<uint16_t>(dist.sequential.delta),
      static_cast<uint16_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kU32:
    cutlass::reference::device::BlockFillSequential<uint32_t>(
      reinterpret_cast<uint32_t *>(pointer_),
      capacity_,
      static_cast<uint32_t>(dist.sequential.delta),
      static_cast<uint32_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kU64:
    cutlass::reference::device::BlockFillSequential<uint64_t>(
      reinterpret_cast<uint64_t *>(pointer_),
      capacity_,
      static_cast<uint64_t>(dist.sequential.delta),
      static_cast<uint64_t>(dist.sequential.start)
    );
    break;
  default: break;
  }

}

void DeviceAllocation::initialize_sequential_host(Distribution dist) {
  if (!bytes()) {
#ifndef NDEBUG
    std::cout << "Skipping initialization of size 0 allocation\n";
#endif
    return;
  }

  if (!data()) {
    throw std::runtime_error("Attempting to initialize invalid allocation.");
  }

  std::vector<uint8_t> host_data(bytes());

  switch (type_) {
  case library::NumericTypeID::kFE4M3:
    cutlass::reference::host::BlockFillSequential<cutlass::float_e4m3_t>(
      reinterpret_cast<cutlass::float_e4m3_t *>(host_data.data()),
      capacity_,
      static_cast<cutlass::float_e4m3_t>(dist.sequential.delta),
      static_cast<cutlass::float_e4m3_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kFE5M2:
    cutlass::reference::host::BlockFillSequential<cutlass::float_e5m2_t>(
      reinterpret_cast<cutlass::float_e5m2_t *>(host_data.data()),
      capacity_,
      static_cast<cutlass::float_e5m2_t>(dist.sequential.delta),
      static_cast<cutlass::float_e5m2_t>(dist.sequential.start)
    );
    break;
  
  case library::NumericTypeID::kFUE4M3:
    cutlass::reference::host::BlockFillSequential<cutlass::float_ue4m3_t>(
      reinterpret_cast<cutlass::float_ue4m3_t *>(host_data.data()),
      capacity_,
      static_cast<cutlass::float_ue4m3_t>(dist.sequential.delta),
      static_cast<cutlass::float_ue4m3_t>(dist.sequential.start)
    );
    break;
  
  case library::NumericTypeID::kFE2M3:
    cutlass::reference::host::BlockFillSequential<cutlass::float_e2m3_t>(
      reinterpret_cast<cutlass::float_e2m3_t *>(host_data.data()),
      capacity_,
      static_cast<cutlass::float_e2m3_t>(dist.sequential.delta),
      static_cast<cutlass::float_e2m3_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kFE3M2:
    cutlass::reference::host::BlockFillSequential<cutlass::float_e3m2_t>(
      reinterpret_cast<cutlass::float_e3m2_t *>(host_data.data()),
      capacity_,
      static_cast<cutlass::float_e3m2_t>(dist.sequential.delta),
      static_cast<cutlass::float_e3m2_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kFE2M1:
    cutlass::reference::host::BlockFillSequential<cutlass::float_e2m1_t>(
      reinterpret_cast<cutlass::float_e2m1_t *>(host_data.data()),
      capacity_,
      static_cast<cutlass::float_e2m1_t>(dist.sequential.delta),
      static_cast<cutlass::float_e2m1_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kFUE8M0:
    cutlass::reference::host::BlockFillSequential<cutlass::float_ue8m0_t>(
      reinterpret_cast<cutlass::float_ue8m0_t *>(host_data.data()),
      capacity_,
      static_cast<cutlass::float_ue8m0_t>(dist.sequential.delta),
      static_cast<cutlass::float_ue8m0_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kF16:
    cutlass::reference::host::BlockFillSequential<cutlass::half_t>(
      reinterpret_cast<cutlass::half_t *>(host_data.data()),
      capacity_,
      static_cast<cutlass::half_t>(dist.sequential.delta),
      static_cast<cutlass::half_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kBF16:
    cutlass::reference::host::BlockFillSequential<cutlass::bfloat16_t>(
      reinterpret_cast<cutlass::bfloat16_t *>(host_data.data()),
      capacity_,
      static_cast<cutlass::bfloat16_t>(dist.sequential.delta),
      static_cast<cutlass::bfloat16_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kTF32:
    cutlass::reference::host::BlockFillSequential<cutlass::tfloat32_t>(
      reinterpret_cast<cutlass::tfloat32_t *>(host_data.data()),
      capacity_,
      static_cast<cutlass::tfloat32_t>(dist.sequential.delta),
      static_cast<cutlass::tfloat32_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kF32:
    cutlass::reference::host::BlockFillSequential<float>(
      reinterpret_cast<float *>(host_data.data()),
      capacity_,
      static_cast<float>(dist.sequential.delta),
      static_cast<float>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kCF16:
    cutlass::reference::host::BlockFillSequential<cutlass::complex<cutlass::half_t>>(
      reinterpret_cast<cutlass::complex<cutlass::half_t> *>(host_data.data()),
      capacity_,
      cutlass::complex<cutlass::half_t>(
        static_cast<cutlass::half_t>(dist.sequential.delta)),
      cutlass::complex<cutlass::half_t>(
        static_cast<cutlass::half_t>(dist.sequential.start))
    );
    break;
  case library::NumericTypeID::kCBF16:
    cutlass::reference::host::BlockFillSequential<cutlass::complex<cutlass::bfloat16_t>>(
      reinterpret_cast<cutlass::complex<cutlass::bfloat16_t> *>(host_data.data()),
      capacity_,
      cutlass::complex<cutlass::bfloat16_t>(
        static_cast<cutlass::bfloat16_t>(dist.sequential.delta)),
      cutlass::complex<cutlass::bfloat16_t>(
        static_cast<cutlass::bfloat16_t>(dist.sequential.start))
    );
    break;
  case library::NumericTypeID::kCTF32:
    cutlass::reference::host::BlockFillSequential<cutlass::complex<cutlass::tfloat32_t>>(
      reinterpret_cast<cutlass::complex<cutlass::tfloat32_t> *>(host_data.data()),
      capacity_,
      cutlass::complex<cutlass::tfloat32_t>(
        static_cast<cutlass::tfloat32_t>(dist.sequential.delta)),
      cutlass::complex<cutlass::tfloat32_t>(
        static_cast<cutlass::tfloat32_t>(dist.sequential.start))
    );
    break;
  case library::NumericTypeID::kCF32:
    cutlass::reference::host::BlockFillSequential<cutlass::complex<float>>(
      reinterpret_cast<cutlass::complex<float> *>(host_data.data()),
      capacity_,
      cutlass::complex<float>(
        static_cast<float>(dist.sequential.delta)),
      cutlass::complex<float>(
        static_cast<float>(dist.sequential.start))
    );
    break;
  case library::NumericTypeID::kF64:
    cutlass::reference::host::BlockFillSequential<double>(
      reinterpret_cast<double *>(host_data.data()),
      capacity_,
      static_cast<double>(dist.sequential.delta),
      static_cast<double>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kCF64:
    cutlass::reference::host::BlockFillSequential<cutlass::complex<double>>(
      reinterpret_cast<cutlass::complex<double> *>(host_data.data()),
      capacity_,
      cutlass::complex<double>(
        static_cast<double>(dist.sequential.delta)),
      cutlass::complex<double>(
        static_cast<double>(dist.sequential.start))
    );
    break;
  case library::NumericTypeID::kS2:
    cutlass::reference::host::BlockFillSequential<int2b_t>(
      reinterpret_cast<int2b_t *>(host_data.data()),
      capacity_,
      static_cast<int2b_t>(dist.sequential.delta),
      static_cast<int2b_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kS4:
    cutlass::reference::host::BlockFillSequential<int4b_t>(
      reinterpret_cast<int4b_t *>(host_data.data()),
      capacity_,
      static_cast<int4b_t>(dist.sequential.delta),
      static_cast<int4b_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kS8:
    cutlass::reference::host::BlockFillSequential<int8_t>(
      reinterpret_cast<int8_t *>(host_data.data()),
      capacity_,
      static_cast<int8_t>(dist.sequential.delta),
      static_cast<int8_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kS16:
    cutlass::reference::host::BlockFillSequential<int16_t>(
      reinterpret_cast<int16_t *>(host_data.data()),
      capacity_,
      static_cast<int16_t>(dist.sequential.delta),
      static_cast<int16_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kS32:
    cutlass::reference::host::BlockFillSequential<int32_t>(
      reinterpret_cast<int32_t *>(host_data.data()),
      capacity_,
      static_cast<int32_t>(dist.sequential.delta),
      static_cast<int32_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kS64:
    cutlass::reference::host::BlockFillSequential<int64_t>(
      reinterpret_cast<int64_t *>(host_data.data()),
      capacity_,
      static_cast<int64_t>(dist.sequential.delta),
      static_cast<int64_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kB1:
    cutlass::reference::host::BlockFillSequential<uint1b_t>(
      reinterpret_cast<uint1b_t *>(host_data.data()),
      capacity_,
      static_cast<uint1b_t>(dist.sequential.delta),
      static_cast<uint1b_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kU2:
    cutlass::reference::host::BlockFillSequential<uint2b_t>(
      reinterpret_cast<uint2b_t *>(host_data.data()),
      capacity_,
      static_cast<uint2b_t>(dist.sequential.delta),
      static_cast<uint2b_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kU4:
    cutlass::reference::host::BlockFillSequential<uint4b_t>(
      reinterpret_cast<uint4b_t *>(host_data.data()),
      capacity_,
      static_cast<uint4b_t>(dist.sequential.delta),
      static_cast<uint4b_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kU8:
    cutlass::reference::host::BlockFillSequential<uint8_t>(
      reinterpret_cast<uint8_t *>(host_data.data()),
      capacity_,
      static_cast<uint8_t>(dist.sequential.delta),
      static_cast<uint8_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kU16:
    cutlass::reference::host::BlockFillSequential<uint16_t>(
      reinterpret_cast<uint16_t *>(host_data.data()),
      capacity_,
      static_cast<uint16_t>(dist.sequential.delta),
      static_cast<uint16_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kU32:
    cutlass::reference::host::BlockFillSequential<uint32_t>(
      reinterpret_cast<uint32_t *>(host_data.data()),
      capacity_,
      static_cast<uint32_t>(dist.sequential.delta),
      static_cast<uint32_t>(dist.sequential.start)
    );
    break;
  case library::NumericTypeID::kU64:
    cutlass::reference::host::BlockFillSequential<uint64_t>(
      reinterpret_cast<uint64_t *>(host_data.data()),
      capacity_,
      static_cast<uint64_t>(dist.sequential.delta),
      static_cast<uint64_t>(dist.sequential.start)
    );
    break;
  default: break;
  }

  copy_from_host(host_data.data());
}

void DeviceAllocation::initialize_random_sparsemeta_device(int seed, int MetaSizeInBits) {
  if (!bytes()) {
#ifndef NDEBUG
    std::cout << "Skipping initialization of size 0 allocation\n";
#endif
    return;
  }

  if (!data()) {
    throw std::runtime_error("Attempting to initialize invalid allocation.");
  }

  // Instantiate calls to CURAND here. This file takes a long time to compile for
  // this reason.

  switch (type_) {
  case library::NumericTypeID::kU16:
    cutlass::reference::device::BlockFillRandomSparseMeta<uint16_t>(
      reinterpret_cast<uint16_t *>(pointer_),
      capacity_,
      seed,
      MetaSizeInBits
    );
    break;
  case library::NumericTypeID::kU32:
    cutlass::reference::device::BlockFillRandomSparseMeta<uint32_t>(
      reinterpret_cast<uint32_t *>(pointer_),
      capacity_,
      seed,
      MetaSizeInBits
    );
    break;
  default:
    break;
  }
}

void DeviceAllocation::initialize_random_sparsemeta_host(int seed, int MetaSizeInBits) {
  if (!bytes()) {
#ifndef NDEBUG
    std::cout << "Skipping initialization of size 0 allocation\n";
#endif
    return;
  }

  if (!data()) {
    throw std::runtime_error("Attempting to initialize invalid allocation.");
  }

  std::vector<uint8_t> host_data(bytes());

  switch (type_) {
  case library::NumericTypeID::kS16:
    cutlass::reference::host::BlockFillRandomSparseMeta<uint16_t>(
      reinterpret_cast<uint16_t *>(host_data.data()),
      capacity_,
      seed,
      MetaSizeInBits
    );
    break;
  case library::NumericTypeID::kS32:
    cutlass::reference::host::BlockFillRandomSparseMeta<uint32_t>(
      reinterpret_cast<uint32_t *>(host_data.data()),
      capacity_,
      seed,
      MetaSizeInBits
    );
    break;
  default:
    break;
  }

  copy_from_host(host_data.data());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if two blocks have exactly the same value
bool DeviceAllocation::block_compare_equal(
  library::NumericTypeID numeric_type,
  void const *ptr_A,
  void const *ptr_B,
  size_t capacity) {

  switch (numeric_type) {
  case library::NumericTypeID::kFE4M3:
    return reference::device::BlockCompareEqual<float_e4m3_t>(
      reinterpret_cast<float_e4m3_t const *>(ptr_A),
      reinterpret_cast<float_e4m3_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kFE5M2:
    return reference::device::BlockCompareEqual<float_e5m2_t>(
      reinterpret_cast<float_e5m2_t const *>(ptr_A),
      reinterpret_cast<float_e5m2_t const *>(ptr_B),
      capacity);
  case library::NumericTypeID::kFUE4M3:
    return reference::device::BlockCompareEqual<float_ue4m3_t>(
      reinterpret_cast<float_ue4m3_t const *>(ptr_A),
      reinterpret_cast<float_ue4m3_t const *>(ptr_B),
      capacity);
  case library::NumericTypeID::kFUE8M0:
    return reference::device::BlockCompareEqual<float_ue8m0_t>(
      reinterpret_cast<float_ue8m0_t const *>(ptr_A),
      reinterpret_cast<float_ue8m0_t const *>(ptr_B),
      capacity);
  case library::NumericTypeID::kFE2M3:
    return reference::device::BlockCompareEqual<float_e2m3_t>(
      reinterpret_cast<float_e2m3_t const *>(ptr_A),
      reinterpret_cast<float_e2m3_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kFE3M2:
    return reference::device::BlockCompareEqual<float_e3m2_t>(
      reinterpret_cast<float_e3m2_t const *>(ptr_A),
      reinterpret_cast<float_e3m2_t const *>(ptr_B),
      capacity);
  case library::NumericTypeID::kFE2M1:
    return reference::device::BlockCompareEqual<float_e2m1_t>(
      reinterpret_cast<float_e2m1_t const *>(ptr_A),
      reinterpret_cast<float_e2m1_t const *>(ptr_B),
      capacity);
  case library::NumericTypeID::kF16:
    return reference::device::BlockCompareEqual<half_t>(
      reinterpret_cast<half_t const *>(ptr_A),
      reinterpret_cast<half_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kBF16:
    return reference::device::BlockCompareEqual<bfloat16_t>(
      reinterpret_cast<bfloat16_t const *>(ptr_A),
      reinterpret_cast<bfloat16_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kTF32:
    return reference::device::BlockCompareEqual<tfloat32_t>(
      reinterpret_cast<tfloat32_t const *>(ptr_A),
      reinterpret_cast<tfloat32_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kF32:
    return reference::device::BlockCompareEqual<float>(
      reinterpret_cast<float const *>(ptr_A),
      reinterpret_cast<float const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kCF32:
    return reference::device::BlockCompareEqual<cutlass::complex<float> >(
      reinterpret_cast<complex<float> const *>(ptr_A),
      reinterpret_cast<complex<float> const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kCF16:
    return reference::device::BlockCompareEqual<complex<half_t>>(
      reinterpret_cast<complex<half_t> const *>(ptr_A),
      reinterpret_cast<complex<half_t> const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kCBF16:
    return reference::device::BlockCompareEqual<complex<bfloat16_t>>(
      reinterpret_cast<complex<bfloat16_t> const *>(ptr_A),
      reinterpret_cast<complex<bfloat16_t> const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kCTF32:
    return reference::device::BlockCompareEqual<complex<tfloat32_t>>(
      reinterpret_cast<complex<tfloat32_t> const *>(ptr_A),
      reinterpret_cast<complex<tfloat32_t> const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kF64:
    return reference::device::BlockCompareEqual<double>(
      reinterpret_cast<double const *>(ptr_A),
      reinterpret_cast<double const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kCF64:
    return reference::device::BlockCompareEqual<complex<double>>(
      reinterpret_cast<complex<double> const *>(ptr_A),
      reinterpret_cast<complex<double> const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kS2:
    return reference::device::BlockCompareEqual<int2b_t>(
      reinterpret_cast<int2b_t const *>(ptr_A),
      reinterpret_cast<int2b_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kS4:
    return reference::device::BlockCompareEqual<int4b_t>(
      reinterpret_cast<int4b_t const *>(ptr_A),
      reinterpret_cast<int4b_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kS8:
    return reference::device::BlockCompareEqual<int8_t>(
      reinterpret_cast<int8_t const *>(ptr_A),
      reinterpret_cast<int8_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kS16:
    return reference::device::BlockCompareEqual<int16_t>(
      reinterpret_cast<int16_t const *>(ptr_A),
      reinterpret_cast<int16_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kS32:
    return reference::device::BlockCompareEqual<int32_t>(
      reinterpret_cast<int32_t const *>(ptr_A),
      reinterpret_cast<int32_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kS64:
    return reference::device::BlockCompareEqual<int64_t>(
      reinterpret_cast<int64_t const *>(ptr_A),
      reinterpret_cast<int64_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kB1:
    return reference::device::BlockCompareEqual<uint1b_t>(
      reinterpret_cast<uint1b_t const *>(ptr_A),
      reinterpret_cast<uint1b_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kU2:
    return reference::device::BlockCompareEqual<uint2b_t>(
      reinterpret_cast<uint2b_t const *>(ptr_A),
      reinterpret_cast<uint2b_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kU4:
    return reference::device::BlockCompareEqual<uint4b_t>(
      reinterpret_cast<uint4b_t const *>(ptr_A),
      reinterpret_cast<uint4b_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kU8:
    return reference::device::BlockCompareEqual<uint8_t>(
      reinterpret_cast<uint8_t const *>(ptr_A),
      reinterpret_cast<uint8_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kU16:
    return reference::device::BlockCompareEqual<uint16_t>(
      reinterpret_cast<uint16_t const *>(ptr_A),
      reinterpret_cast<uint16_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kU32:
    return reference::device::BlockCompareEqual<uint32_t>(
      reinterpret_cast<uint32_t const *>(ptr_A),
      reinterpret_cast<uint32_t const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kU64:
    return reference::device::BlockCompareEqual<uint64_t>(
      reinterpret_cast<uint64_t const *>(ptr_A),
      reinterpret_cast<uint64_t const *>(ptr_B),
      capacity);

  default:
    throw std::runtime_error(std::string("Unsupported numeric type: ") + to_string(numeric_type));
  }
}

/// Returns true if two blocks have approximately the same value
bool DeviceAllocation::block_compare_relatively_equal(
  library::NumericTypeID numeric_type,
  void const *ptr_A,
  void const *ptr_B,
  size_t capacity,
  double epsilon,
  double nonzero_floor) {

  switch (numeric_type) {
  case library::NumericTypeID::kFE4M3:
    return reference::device::BlockCompareRelativelyEqual<float_e4m3_t>(
      reinterpret_cast<float_e4m3_t const *>(ptr_A),
      reinterpret_cast<float_e4m3_t const *>(ptr_B),
      capacity,
      static_cast<float_e4m3_t>(epsilon),
      static_cast<float_e4m3_t>(nonzero_floor));

  case library::NumericTypeID::kFE5M2:
    return reference::device::BlockCompareRelativelyEqual<float_e5m2_t>(
      reinterpret_cast<float_e5m2_t const *>(ptr_A),
      reinterpret_cast<float_e5m2_t const *>(ptr_B),
      capacity,
      static_cast<float_e5m2_t>(epsilon),
      static_cast<float_e5m2_t>(nonzero_floor));
  case library::NumericTypeID::kFUE4M3:
    return reference::device::BlockCompareRelativelyEqual<float_ue4m3_t>(
      reinterpret_cast<float_ue4m3_t const *>(ptr_A),
      reinterpret_cast<float_ue4m3_t const *>(ptr_B),
      capacity,
      static_cast<float_ue4m3_t>(epsilon),
      static_cast<float_ue4m3_t>(nonzero_floor));
  case library::NumericTypeID::kFUE8M0:
    return reference::device::BlockCompareRelativelyEqual<float_ue8m0_t>(
      reinterpret_cast<float_ue8m0_t const *>(ptr_A),
      reinterpret_cast<float_ue8m0_t const *>(ptr_B),
      capacity,
      static_cast<float_ue8m0_t>(epsilon),
      static_cast<float_ue8m0_t>(nonzero_floor));

  case library::NumericTypeID::kFE2M3:
    return reference::device::BlockCompareRelativelyEqual<float_e2m3_t>(
      reinterpret_cast<float_e2m3_t const *>(ptr_A),
      reinterpret_cast<float_e2m3_t const *>(ptr_B),
      capacity,
      static_cast<float_e2m3_t>(epsilon),
      static_cast<float_e2m3_t>(nonzero_floor));

  case library::NumericTypeID::kFE3M2:
      return reference::device::BlockCompareRelativelyEqual<float_e3m2_t>(
        reinterpret_cast<float_e3m2_t const *>(ptr_A),
        reinterpret_cast<float_e3m2_t const *>(ptr_B),
        capacity,
        static_cast<float_e3m2_t>(epsilon),
        static_cast<float_e3m2_t>(nonzero_floor));

  case library::NumericTypeID::kFE2M1:
    return reference::device::BlockCompareRelativelyEqual<float_e2m1_t>(
      reinterpret_cast<float_e2m1_t const *>(ptr_A),
      reinterpret_cast<float_e2m1_t const *>(ptr_B),
      capacity,
      static_cast<float_e2m1_t>(epsilon),
      static_cast<float_e2m1_t>(nonzero_floor));
  case library::NumericTypeID::kF16:
    return reference::device::BlockCompareRelativelyEqual<half_t>(
      reinterpret_cast<half_t const *>(ptr_A),
      reinterpret_cast<half_t const *>(ptr_B),
      capacity,
      static_cast<half_t>(epsilon),
      static_cast<half_t>(nonzero_floor));

  case library::NumericTypeID::kBF16:
    return reference::device::BlockCompareRelativelyEqual<bfloat16_t>(
      reinterpret_cast<bfloat16_t const *>(ptr_A),
      reinterpret_cast<bfloat16_t const *>(ptr_B),
      capacity,
      static_cast<bfloat16_t>(epsilon),
      static_cast<bfloat16_t>(nonzero_floor));

  case library::NumericTypeID::kTF32:
    return reference::device::BlockCompareRelativelyEqual<tfloat32_t>(
      reinterpret_cast<tfloat32_t const *>(ptr_A),
      reinterpret_cast<tfloat32_t const *>(ptr_B),
      capacity,
      static_cast<tfloat32_t>(epsilon),
      static_cast<tfloat32_t>(nonzero_floor));

  case library::NumericTypeID::kF32:
    return reference::device::BlockCompareRelativelyEqual<float>(
      reinterpret_cast<float const *>(ptr_A),
      reinterpret_cast<float const *>(ptr_B),
      capacity,
      static_cast<float>(epsilon),
      static_cast<float>(nonzero_floor));

  case library::NumericTypeID::kF64:
    return reference::device::BlockCompareRelativelyEqual<double>(
      reinterpret_cast<double const *>(ptr_A),
      reinterpret_cast<double const *>(ptr_B),
      capacity,
      static_cast<double>(epsilon),
      static_cast<double>(nonzero_floor));

  case library::NumericTypeID::kS2:
    return reference::device::BlockCompareRelativelyEqual<int2b_t>(
      reinterpret_cast<int2b_t const *>(ptr_A),
      reinterpret_cast<int2b_t const *>(ptr_B),
      capacity,
      static_cast<int2b_t>(epsilon),
      static_cast<int2b_t>(nonzero_floor));

  case library::NumericTypeID::kS4:
    return reference::device::BlockCompareRelativelyEqual<int4b_t>(
      reinterpret_cast<int4b_t const *>(ptr_A),
      reinterpret_cast<int4b_t const *>(ptr_B),
      capacity,
      static_cast<int4b_t>(epsilon),
      static_cast<int4b_t>(nonzero_floor));

  case library::NumericTypeID::kS8:
    return reference::device::BlockCompareRelativelyEqual<int8_t>(
      reinterpret_cast<int8_t const *>(ptr_A),
      reinterpret_cast<int8_t const *>(ptr_B),
      capacity,
      static_cast<int8_t>(epsilon),
      static_cast<int8_t>(nonzero_floor));

  case library::NumericTypeID::kS16:
    return reference::device::BlockCompareRelativelyEqual<int16_t>(
      reinterpret_cast<int16_t const *>(ptr_A),
      reinterpret_cast<int16_t const *>(ptr_B),
      capacity,
      static_cast<int16_t>(epsilon),
      static_cast<int16_t>(nonzero_floor));

  case library::NumericTypeID::kS32:
    return reference::device::BlockCompareRelativelyEqual<int32_t>(
      reinterpret_cast<int32_t const *>(ptr_A),
      reinterpret_cast<int32_t const *>(ptr_B),
      capacity,
      static_cast<int32_t>(epsilon),
      static_cast<int32_t>(nonzero_floor));

  case library::NumericTypeID::kS64:
    return reference::device::BlockCompareRelativelyEqual<int64_t>(
      reinterpret_cast<int64_t const *>(ptr_A),
      reinterpret_cast<int64_t const *>(ptr_B),
      capacity,
      static_cast<int64_t>(epsilon),
      static_cast<int64_t>(nonzero_floor));

  case library::NumericTypeID::kB1:
    return reference::device::BlockCompareRelativelyEqual<uint1b_t>(
      reinterpret_cast<uint1b_t const *>(ptr_A),
      reinterpret_cast<uint1b_t const *>(ptr_B),
      capacity,
      static_cast<uint1b_t>(epsilon),
      static_cast<uint1b_t>(nonzero_floor));

  case library::NumericTypeID::kU2:
    return reference::device::BlockCompareRelativelyEqual<uint2b_t>(
      reinterpret_cast<uint2b_t const *>(ptr_A),
      reinterpret_cast<uint2b_t const *>(ptr_B),
      capacity,
      static_cast<uint2b_t>(epsilon),
      static_cast<uint2b_t>(nonzero_floor));

  case library::NumericTypeID::kU4:
    return reference::device::BlockCompareRelativelyEqual<uint4b_t>(
      reinterpret_cast<uint4b_t const *>(ptr_A),
      reinterpret_cast<uint4b_t const *>(ptr_B),
      capacity,
      static_cast<uint4b_t>(epsilon),
      static_cast<uint4b_t>(nonzero_floor));

  case library::NumericTypeID::kU8:
    return reference::device::BlockCompareRelativelyEqual<uint8_t>(
      reinterpret_cast<uint8_t const *>(ptr_A),
      reinterpret_cast<uint8_t const *>(ptr_B),
      capacity,
      static_cast<uint8_t>(epsilon),
      static_cast<uint8_t>(nonzero_floor));

  case library::NumericTypeID::kU16:
    return reference::device::BlockCompareRelativelyEqual<uint16_t>(
      reinterpret_cast<uint16_t const *>(ptr_A),
      reinterpret_cast<uint16_t const *>(ptr_B),
      capacity,
      static_cast<uint16_t>(epsilon),
      static_cast<uint16_t>(nonzero_floor));

  case library::NumericTypeID::kU32:
    return reference::device::BlockCompareRelativelyEqual<uint32_t>(
      reinterpret_cast<uint32_t const *>(ptr_A),
      reinterpret_cast<uint32_t const *>(ptr_B),
      capacity,
      static_cast<uint32_t>(epsilon),
      static_cast<uint32_t>(nonzero_floor));

  case library::NumericTypeID::kU64:
    return reference::device::BlockCompareRelativelyEqual<uint64_t>(
      reinterpret_cast<uint64_t const *>(ptr_A),
      reinterpret_cast<uint64_t const *>(ptr_B),
      capacity,
      static_cast<uint64_t>(epsilon),
      static_cast<uint64_t>(nonzero_floor));

  // No relatively equal comparison for complex numbers.
  //
  // As a simplification, we can require bitwise equality. This avoids false positives.
  // (i.e. "pass" really means passing. "Fail" may not actually mean failure given appropriate epsilon.)
  //
  case library::NumericTypeID::kCF16:
    return reference::device::BlockCompareEqual<cutlass::complex<half_t> >(
      reinterpret_cast<complex<half_t> const *>(ptr_A),
      reinterpret_cast<complex<half_t> const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kCF32:
    return reference::device::BlockCompareEqual<cutlass::complex<float> >(
      reinterpret_cast<complex<float> const *>(ptr_A),
      reinterpret_cast<complex<float> const *>(ptr_B),
      capacity);

  case library::NumericTypeID::kCF64:
    return reference::device::BlockCompareEqual<cutlass::complex<double> >(
      reinterpret_cast<complex<double> const *>(ptr_A),
      reinterpret_cast<complex<double> const *>(ptr_B),
      capacity);

  default:
    {
      throw std::runtime_error(std::string("Unsupported numeric type: ") + to_string(numeric_type));
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Permits copying dynamic vectors into static-length vectors
template <typename TensorCoord, int Rank>
struct vector_to_coord {

  vector_to_coord(TensorCoord &coord, std::vector<int> const &vec) {

    coord[Rank - 1] = vec.at(Rank - 1);

    if (Rank > 1) {
      vector_to_coord<TensorCoord, Rank - 1>(coord, vec);
    }
  }

  vector_to_coord(TensorCoord &coord, std::vector<int64_t> const &vec) {

    coord[Rank - 1] = (int)vec.at(Rank - 1);

    if (Rank > 1) {
      vector_to_coord<TensorCoord, Rank - 1>(coord, vec);
    }
  }
};

/// Permits copying dynamic vectors into static-length vectors
template <typename TensorCoord>
struct vector_to_coord<TensorCoord, 1> {

  vector_to_coord(TensorCoord &coord, std::vector<int> const &vec) {

    coord[0] = vec.at(0);
  }

  vector_to_coord(TensorCoord &coord, std::vector<int64_t> const &vec) {

    coord[0] = (int)vec.at(0);
  }
};

/// Permits copying dynamic vectors into static-length vectors
template <typename TensorCoord>
struct vector_to_coord<TensorCoord, 0> {

  vector_to_coord(TensorCoord &coord, std::vector<int> const &vec) {

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element, typename Layout>
static void write_tensor_csv_static_tensor_view(
  std::ostream &out,
  DeviceAllocation &allocation) {

  Coord<Layout::kRank> extent;
  Coord<Layout::kStrideRank, typename Layout::Stride::Index> stride;

  if (allocation.extent().size() != Layout::kRank) {
    throw std::runtime_error("Allocation extent has invalid rank");
  }

  if (allocation.stride().size() != Layout::kStrideRank) {
    throw std::runtime_error("Allocation stride has invalid rank");
  }

  vector_to_coord<Coord<Layout::kRank>, Layout::kRank>(extent, allocation.extent());
  vector_to_coord<Coord<Layout::kStrideRank, typename Layout::Stride::Index>,
                        Layout::kStrideRank>(stride, allocation.stride());

  Layout layout(stride);
  HostTensor<Element, Layout> host_tensor(extent, layout, false);

  if (host_tensor.capacity() != allocation.batch_stride()) {
    throw std::runtime_error("Unexpected capacity to equal.");
  }

  host_tensor.copy_in_device_to_host(
    static_cast<Element const *>(allocation.data()),
    allocation.batch_stride());

  TensorViewWrite(out, host_tensor.host_view());

  out << "\n\n";
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
static void write_tensor_csv_static_type(
  std::ostream &out,
  DeviceAllocation &allocation) {

  switch (allocation.layout()) {
    case library::LayoutTypeID::kRowMajor:
      write_tensor_csv_static_tensor_view<T, layout::RowMajor>(out, allocation);
      break;
    case library::LayoutTypeID::kColumnMajor:
      write_tensor_csv_static_tensor_view<T, layout::ColumnMajor>(out, allocation);
      break;
    case library::LayoutTypeID::kRowMajorInterleavedK2:
      write_tensor_csv_static_tensor_view<T, layout::RowMajorInterleaved<2>>(out, allocation);
      break;
    case library::LayoutTypeID::kColumnMajorInterleavedK2:
      write_tensor_csv_static_tensor_view<T, layout::ColumnMajorInterleaved<2>>(out, allocation);
      break;
    case library::LayoutTypeID::kRowMajorInterleavedK4:
      write_tensor_csv_static_tensor_view<T, layout::RowMajorInterleaved<4>>(out, allocation);
      break;
    case library::LayoutTypeID::kColumnMajorInterleavedK4:
      write_tensor_csv_static_tensor_view<T, layout::ColumnMajorInterleaved<4>>(out, allocation);
      break;
    case library::LayoutTypeID::kRowMajorInterleavedK16:
      write_tensor_csv_static_tensor_view<T, layout::RowMajorInterleaved<16>>(out, allocation);
      break;
    case library::LayoutTypeID::kColumnMajorInterleavedK16:
      write_tensor_csv_static_tensor_view<T, layout::ColumnMajorInterleaved<16>>(out, allocation);
      break;
    case library::LayoutTypeID::kRowMajorInterleavedK32:
      write_tensor_csv_static_tensor_view<T, layout::RowMajorInterleaved<32>>(out, allocation);
      break;
    case library::LayoutTypeID::kColumnMajorInterleavedK32:
      write_tensor_csv_static_tensor_view<T, layout::ColumnMajorInterleaved<32>>(out, allocation);
      break;
    case library::LayoutTypeID::kRowMajorInterleavedK64:
      write_tensor_csv_static_tensor_view<T, layout::RowMajorInterleaved<64>>(out, allocation);
      break;
    case library::LayoutTypeID::kColumnMajorInterleavedK64:
      write_tensor_csv_static_tensor_view<T, layout::ColumnMajorInterleaved<64>>(out, allocation);
      break;
    case library::LayoutTypeID::kTensorNHWC:
      write_tensor_csv_static_tensor_view<T, layout::TensorNHWC>(out, allocation);
      break;
    case library::LayoutTypeID::kTensorNDHWC:
      write_tensor_csv_static_tensor_view<T, layout::TensorNDHWC>(out, allocation);
      break;
    case library::LayoutTypeID::kTensorNC32HW32:
      write_tensor_csv_static_tensor_view<T, layout::TensorNCxHWx<32>>(out, allocation);
      break;
    case library::LayoutTypeID::kTensorNC64HW64:
      write_tensor_csv_static_tensor_view<T, layout::TensorNCxHWx<64>>(out, allocation);
      break;
    case library::LayoutTypeID::kTensorC32RSK32:
      write_tensor_csv_static_tensor_view<T, layout::TensorCxRSKx<32>>(out, allocation);
      break;
    case library::LayoutTypeID::kTensorC64RSK64:
      write_tensor_csv_static_tensor_view<T, layout::TensorCxRSKx<64>>(out, allocation);
      break;
    default:
      throw std::runtime_error("Unhandled layout");
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Writes a tensor to csv
void DeviceAllocation::write_tensor_csv(
  std::ostream &out) {

  switch (this->type()) {
  case library::NumericTypeID::kFE4M3:
    write_tensor_csv_static_type<float_e4m3_t>(out, *this);
    break;

  case library::NumericTypeID::kFE5M2:
    write_tensor_csv_static_type<float_e5m2_t>(out, *this);
    break;
  
  case library::NumericTypeID::kFUE4M3:
    write_tensor_csv_static_type<float_ue4m3_t>(out, *this);
    break;
  
  case library::NumericTypeID::kFE2M3:
    write_tensor_csv_static_type<float_e2m3_t>(out, *this);
    break;

  case library::NumericTypeID::kFE3M2:
    write_tensor_csv_static_type<float_e3m2_t>(out, *this);
    break;

  case library::NumericTypeID::kFE2M1:
    write_tensor_csv_static_type<float_e2m1_t>(out, *this);
    break;
  case library::NumericTypeID::kFUE8M0:
    write_tensor_csv_static_type<float_ue8m0_t>(out, *this);
    break;
  case library::NumericTypeID::kF16:
    write_tensor_csv_static_type<half_t>(out, *this);
    break;

  case library::NumericTypeID::kBF16:
    write_tensor_csv_static_type<bfloat16_t>(out, *this);
    break;

  case library::NumericTypeID::kTF32:
    write_tensor_csv_static_type<tfloat32_t>(out, *this);
    break;

  case library::NumericTypeID::kF32:
    write_tensor_csv_static_type<float>(out, *this);
    break;

  case library::NumericTypeID::kF64:
    write_tensor_csv_static_type<double>(out, *this);
    break;

  case library::NumericTypeID::kS2:
    write_tensor_csv_static_type<int2b_t>(out, *this);
    break;

  case library::NumericTypeID::kS4:
    write_tensor_csv_static_type<int4b_t>(out, *this);
    break;

  case library::NumericTypeID::kS8:
    write_tensor_csv_static_type<int8_t>(out, *this);
    break;

  case library::NumericTypeID::kS16:
    write_tensor_csv_static_type<int16_t>(out, *this);
    break;

  case library::NumericTypeID::kS32:
    write_tensor_csv_static_type<int32_t>(out, *this);
    break;

  case library::NumericTypeID::kS64:
    write_tensor_csv_static_type<int64_t>(out, *this);
    break;

  case library::NumericTypeID::kB1:
    write_tensor_csv_static_type<uint1b_t>(out, *this);
    break;

  case library::NumericTypeID::kU2:
    write_tensor_csv_static_type<uint2b_t>(out, *this);
    break;

  case library::NumericTypeID::kU4:
    write_tensor_csv_static_type<uint4b_t>(out, *this);
    break;

  case library::NumericTypeID::kU8:
    write_tensor_csv_static_type<uint8_t>(out, *this);
    break;

  case library::NumericTypeID::kU16:
    write_tensor_csv_static_type<uint16_t>(out, *this);
    break;

  case library::NumericTypeID::kU32:
    write_tensor_csv_static_type<uint32_t>(out, *this);
    break;

  case library::NumericTypeID::kU64:
    write_tensor_csv_static_type<uint64_t>(out, *this);
    break;

  case library::NumericTypeID::kCF16:
    write_tensor_csv_static_type<cutlass::complex<half_t> >(out, *this);
    break;

  case library::NumericTypeID::kCF32:
    write_tensor_csv_static_type<cutlass::complex<float> >(out, *this);
    break;

  case library::NumericTypeID::kCF64:
    write_tensor_csv_static_type<cutlass::complex<double> >(out, *this);
    break;

  case library::NumericTypeID::kVoid:
    // Not dump anything as it is a empty tensor.
    break;

  default:
    throw std::runtime_error(std::string("Unsupported numeric type: ") + to_string(this->type()) ) ;
  }
}

template <typename Element, typename Layout>
static void tensor_fill_tensor_view(DeviceAllocation &allocation, Element val = Element()) {
  Coord<Layout::kRank> extent;
  Coord<Layout::kStrideRank, typename Layout::LongIndex> stride;

  if (allocation.extent().size() != Layout::kRank) {
    throw std::runtime_error("Allocation extent has invalid rank");
  }

  if (allocation.stride().size() != Layout::kStrideRank) {
    throw std::runtime_error("Allocation stride has invalid rank");
  }

  vector_to_coord<Coord<Layout::kRank>, Layout::kRank>(extent, allocation.extent());
  vector_to_coord<Coord<Layout::kStrideRank, typename Layout::LongIndex>,
                        Layout::kStrideRank>(stride, allocation.stride());

  TensorView<Element, Layout> view(
    static_cast<Element *>(allocation.data()),
    Layout(stride),
    extent
  );


  cutlass::reference::device::TensorFill<Element, Layout>(
    view,
    val
  );
}

template <typename Element>
static void tensor_fill(DeviceAllocation &allocation, Element val = Element()) {
  switch (allocation.layout()) {
    case library::LayoutTypeID::kRowMajor:
      tensor_fill_tensor_view<Element, layout::RowMajor>(allocation, val);
      break;
    case library::LayoutTypeID::kColumnMajor:
      tensor_fill_tensor_view<Element, layout::ColumnMajor>(allocation, val);
      break;
    case library::LayoutTypeID::kTensorNHWC:
      tensor_fill_tensor_view<Element, layout::TensorNHWC>(allocation, val);
      break;
    case library::LayoutTypeID::kTensorNDHWC:
      tensor_fill_tensor_view<Element, layout::TensorNDHWC>(allocation, val);
      break;
    case library::LayoutTypeID::kTensorNC32HW32:
      tensor_fill_tensor_view<Element, layout::TensorNCxHWx<32>>(allocation, val);
      break;
    case library::LayoutTypeID::kTensorNC64HW64:
      tensor_fill_tensor_view<Element, layout::TensorNCxHWx<64>>(allocation, val);
      break;
    case library::LayoutTypeID::kTensorC32RSK32:
      tensor_fill_tensor_view<Element, layout::TensorCxRSKx<32>>(allocation, val);
      break;
    case library::LayoutTypeID::kTensorC64RSK64:
      tensor_fill_tensor_view<Element, layout::TensorCxRSKx<64>>(allocation, val);
      break;
    default:
    throw std::runtime_error("Unsupported layout");
      break;
  }
}

/// Fills a tensor uniformly with a value (most frequently used to clear the tensor)
void DeviceAllocation::fill_device(double val = 0.0) {

  switch (this->type()) {
  case library::NumericTypeID::kFE4M3:
    tensor_fill<float_e4m3_t>(*this, static_cast<float_e4m3_t>(val));
    break;

  case library::NumericTypeID::kFE5M2:
    tensor_fill<float_e5m2_t>(*this, static_cast<float_e5m2_t>(val));
    break;
  
  case library::NumericTypeID::kFUE4M3:
    tensor_fill<float_ue4m3_t>(*this, static_cast<float_ue4m3_t>(val));
    break;
  
  case library::NumericTypeID::kFUE8M0:
    tensor_fill<float_ue8m0_t>(*this, static_cast<float_ue8m0_t>(val));
    break;
  case library::NumericTypeID::kFE2M3:
    tensor_fill<float_e2m3_t>(*this, static_cast<float_e2m3_t>(val));
    break;

  case library::NumericTypeID::kFE3M2:
    tensor_fill<float_e3m2_t>(*this, static_cast<float_e3m2_t>(val));
    break;

  case library::NumericTypeID::kFE2M1:
    tensor_fill<float_e2m1_t>(*this, static_cast<float_e2m1_t>(val));
    break;

  case library::NumericTypeID::kF16:
    tensor_fill<half_t>(*this, static_cast<half_t>(val));
    break;

  case library::NumericTypeID::kBF16:
    tensor_fill<bfloat16_t>(*this, static_cast<bfloat16_t>(val));
    break;

  case library::NumericTypeID::kTF32:
    tensor_fill<tfloat32_t>(*this, static_cast<tfloat32_t>(val));
    break;

  case library::NumericTypeID::kF32:
    tensor_fill<float>(*this, static_cast<float>(val));
    break;

  case library::NumericTypeID::kF64:
    tensor_fill<double>(*this, static_cast<double>(val));
    break;

  case library::NumericTypeID::kS2:
    tensor_fill<int2b_t>(*this, static_cast<int2b_t>(val));
    break;

  case library::NumericTypeID::kS4:
    tensor_fill<int4b_t>(*this, static_cast<int4b_t>(val));
    break;

  case library::NumericTypeID::kS8:
    tensor_fill<int8_t>(*this, static_cast<int8_t>(val));
    break;

  case library::NumericTypeID::kS16:
    tensor_fill<int16_t>(*this, static_cast<int16_t>(val));
    break;

  case library::NumericTypeID::kS32:
    tensor_fill<int32_t>(*this, static_cast<int32_t>(val));
    break;

  case library::NumericTypeID::kS64:
    tensor_fill<int64_t>(*this, static_cast<int64_t>(val));
    break;

  case library::NumericTypeID::kB1:
    tensor_fill<uint1b_t>(*this, static_cast<uint1b_t>(val));
    break;

  case library::NumericTypeID::kU2:
    tensor_fill<uint2b_t>(*this, static_cast<uint2b_t>(val));
    break;

  case library::NumericTypeID::kU4:
    tensor_fill<uint4b_t>(*this, static_cast<uint4b_t>(val));
    break;

  case library::NumericTypeID::kU8:
    tensor_fill<uint8_t>(*this, static_cast<uint8_t>(val));
    break;

  case library::NumericTypeID::kU16:
    tensor_fill<uint16_t>(*this, static_cast<uint16_t>(val));
    break;

  case library::NumericTypeID::kU32:
    tensor_fill<uint32_t>(*this, static_cast<uint32_t>(val));
    break;

  case library::NumericTypeID::kU64:
    tensor_fill<uint64_t>(*this, static_cast<uint64_t>(val));
    break;

  case library::NumericTypeID::kCF16:
    tensor_fill<cutlass::complex<half_t> >(*this, from_real<half_t>(val));
    break;

  case library::NumericTypeID::kCF32:
    tensor_fill<cutlass::complex<float> >(*this, from_real<float>(val));
    break;

  case library::NumericTypeID::kCF64:
    tensor_fill<cutlass::complex<double> >(*this, from_real<double>(val));
    break;

  default:
    throw std::runtime_error(std::string("Unsupported numeric type: ") + to_string(this->type()));
  }
}

/// Fills a tensor uniformly with a value (most frequently used to clear the tensor)
void DeviceAllocation::fill_host(double val = 0.0) {

  std::vector<uint8_t> host_data(bytes());

  switch (this->type()) {
  
  case library::NumericTypeID::kFUE4M3:
    cutlass::reference::host::BlockFill<float_ue4m3_t>(
      reinterpret_cast<float_ue4m3_t *>(host_data.data()),
      capacity_,
      static_cast<float_ue4m3_t>(val)
    );
    break;

  case library::NumericTypeID::kFUE8M0:
    cutlass::reference::host::BlockFill<float_ue8m0_t>(
      reinterpret_cast<float_ue8m0_t *>(host_data.data()),
      capacity_,
      static_cast<float_ue8m0_t>(val)
    );
    break;
  case library::NumericTypeID::kFE2M3:
    cutlass::reference::host::BlockFill<float_e2m3_t>(
      reinterpret_cast<float_e2m3_t *>(host_data.data()),
      capacity_,
      static_cast<float_e2m3_t>(val)
    );
    break;

  case library::NumericTypeID::kFE3M2:
    cutlass::reference::host::BlockFill<float_e3m2_t>(
      reinterpret_cast<float_e3m2_t *>(host_data.data()),
      capacity_,
      static_cast<float_e3m2_t>(val)
    );
    break;

  case library::NumericTypeID::kFE2M1:
    cutlass::reference::host::BlockFill<float_e2m1_t>(
      reinterpret_cast<float_e2m1_t *>(host_data.data()),
      capacity_,
      static_cast<float_e2m1_t>(val)
    );
    break;

  case library::NumericTypeID::kFE4M3:
    cutlass::reference::host::BlockFill<float_e4m3_t>(
      reinterpret_cast<float_e4m3_t *>(host_data.data()),
      capacity_,
      static_cast<float_e4m3_t>(val)
    );
    break;

  case library::NumericTypeID::kFE5M2:
    cutlass::reference::host::BlockFill<float_e5m2_t>(
      reinterpret_cast<float_e5m2_t *>(host_data.data()),
      capacity_,
      static_cast<float_e5m2_t>(val)
    );
    break;

  case library::NumericTypeID::kF16:
    cutlass::reference::host::BlockFill<half_t>(
      reinterpret_cast<half_t *>(host_data.data()),
      capacity_,
      static_cast<half_t>(val)
    );
    break;

  case library::NumericTypeID::kBF16:
    cutlass::reference::host::BlockFill<bfloat16_t>(
      reinterpret_cast<bfloat16_t *>(host_data.data()),
      capacity_,
      static_cast<bfloat16_t>(val)
    );
    break;

  case library::NumericTypeID::kTF32:
    cutlass::reference::host::BlockFill<tfloat32_t>(
      reinterpret_cast<tfloat32_t *>(host_data.data()),
      capacity_,
      static_cast<tfloat32_t>(val)
    );
    break;

  case library::NumericTypeID::kF32:
    cutlass::reference::host::BlockFill<float>(
      reinterpret_cast<float *>(host_data.data()),
      capacity_,
      static_cast<float>(val)
    );
    break;

  case library::NumericTypeID::kF64:
    cutlass::reference::host::BlockFill<double>(
      reinterpret_cast<double *>(host_data.data()),
      capacity_,
      static_cast<double>(val)
    );
    break;

  case library::NumericTypeID::kS2:
    cutlass::reference::host::BlockFill<int2b_t>(
      reinterpret_cast<int2b_t *>(host_data.data()),
      capacity_,
      static_cast<int2b_t>(val)
    );
    break;

  case library::NumericTypeID::kS4:
    cutlass::reference::host::BlockFill<int4b_t>(
      reinterpret_cast<int4b_t *>(host_data.data()),
      capacity_,
      static_cast<int4b_t>(val)
    );
    break;

  case library::NumericTypeID::kS8:
    cutlass::reference::host::BlockFill<int8_t>(
      reinterpret_cast<int8_t *>(host_data.data()),
      capacity_,
      static_cast<int8_t>(val)
    );
    break;

  case library::NumericTypeID::kS16:
    cutlass::reference::host::BlockFill<int16_t>(
      reinterpret_cast<int16_t *>(host_data.data()),
      capacity_,
      static_cast<int16_t>(val)
    );
    break;

  case library::NumericTypeID::kS32:
    cutlass::reference::host::BlockFill<int32_t>(
      reinterpret_cast<int32_t *>(host_data.data()),
      capacity_,
      static_cast<int32_t>(val)
    );
    break;

  case library::NumericTypeID::kS64:
    cutlass::reference::host::BlockFill<int64_t>(
      reinterpret_cast<int64_t *>(host_data.data()),
      capacity_,
      static_cast<int64_t>(val)
    );
    break;

  case library::NumericTypeID::kB1:
    cutlass::reference::host::BlockFill<uint1b_t>(
      reinterpret_cast<uint1b_t *>(host_data.data()),
      capacity_,
      static_cast<uint1b_t>(val)
    );
    break;

  case library::NumericTypeID::kU2:
    cutlass::reference::host::BlockFill<uint2b_t>(
      reinterpret_cast<uint2b_t *>(host_data.data()),
      capacity_,
      static_cast<uint2b_t>(val)
    );
    break;

  case library::NumericTypeID::kU4:
    cutlass::reference::host::BlockFill<uint4b_t>(
      reinterpret_cast<uint4b_t *>(host_data.data()),
      capacity_,
      static_cast<uint4b_t>(val)
    );
    break;

  case library::NumericTypeID::kU8:
    cutlass::reference::host::BlockFill<uint8_t>(
      reinterpret_cast<uint8_t *>(host_data.data()),
      capacity_,
      static_cast<uint8_t>(val)
    );
    break;

  case library::NumericTypeID::kU16:
    cutlass::reference::host::BlockFill<uint16_t>(
      reinterpret_cast<uint16_t *>(host_data.data()),
      capacity_,
      static_cast<uint16_t>(val)
    );
    break;

  case library::NumericTypeID::kU32:
    cutlass::reference::host::BlockFill<uint32_t>(
      reinterpret_cast<uint32_t *>(host_data.data()),
      capacity_,
      static_cast<uint32_t>(val)
    );
    break;

  case library::NumericTypeID::kU64:
    cutlass::reference::host::BlockFill<uint64_t>(
      reinterpret_cast<uint64_t *>(host_data.data()),
      capacity_,
      static_cast<uint64_t>(val)
    );
    break;

  default:
    throw std::runtime_error(std::string("Unsupported numeric type: ") + to_string(this->type()));
  }

  copy_from_host(host_data.data());
}

cudaError_t DeviceAllocation::malloc(void** ptr, size_t size) {
  cudaError_t result;
  int current_device;
  cudaGetDevice(&current_device);

  if (current_device != device_) {
    cudaSetDevice(device_);
  }

  // This performs the cudaMalloc
  result = cudaMalloc(ptr, size);
  if (result != cudaSuccess) {
    return result;
  }

  if (current_device != device_) {
    cudaSetDevice(current_device);
  }

  return cudaSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass
