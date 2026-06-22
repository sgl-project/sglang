/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace omnidreams_native {

class WorkspaceAllocator {
 public:
  WorkspaceAllocator()
      : base_(0),
        total_(0),
        used_(0),
        overflowed_(false),
        failed_offset_(0),
        failed_request_bytes_(0),
        label_("") {}

  WorkspaceAllocator(void* base, size_t total_bytes, const char* label = "workspace")
      : base_(reinterpret_cast<uintptr_t>(base)),
        total_(total_bytes),
        used_(0),
        overflowed_(false),
        failed_offset_(0),
        failed_request_bytes_(0),
        label_(label) {}

  template <typename T>
  T* alloc(size_t count, size_t alignment = 0) {
    if (alignment == 0) {
      alignment = alignof(T);
    }
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
      overflowed_ = true;
      failed_offset_ = used_;
      failed_request_bytes_ = std::numeric_limits<size_t>::max();
      return nullptr;
    }
    void* ptr = alloc_bytes(count * sizeof(T), alignment);
    return reinterpret_cast<T*>(ptr);
  }

  void* alloc_bytes(size_t bytes, size_t alignment = 1) {
    if (!align_to(alignment)) {
      return nullptr;
    }
    if (!reserve(bytes)) {
      return nullptr;
    }
    void* ptr = reinterpret_cast<void*>(base_ + used_);
    used_ += bytes;
    return ptr;
  }

  bool align_to(size_t alignment) {
    if (alignment <= 1) {
      return true;
    }
    const uintptr_t address = base_ + used_;
    const size_t remainder = static_cast<size_t>(address % alignment);
    if (remainder == 0) {
      return true;
    }
    const size_t padding = alignment - remainder;
    if (!reserve(padding)) {
      return false;
    }
    used_ += padding;
    return true;
  }

  void reset() {
    used_ = 0;
    overflowed_ = false;
    failed_offset_ = 0;
    failed_request_bytes_ = 0;
  }

  uint8_t* current() const {
    return reinterpret_cast<uint8_t*>(base_ + used_);
  }

  size_t used() const {
    return used_;
  }

  size_t remaining() const {
    return total_ > used_ ? total_ - used_ : 0;
  }

  size_t total() const {
    return total_;
  }

  bool overflowed() const {
    return overflowed_;
  }

  size_t failed_offset() const {
    return failed_offset_;
  }

  size_t failed_request_bytes() const {
    return failed_request_bytes_;
  }

  const char* label() const {
    return label_;
  }

  WorkspaceAllocator sub(size_t budget_bytes, const char* sub_label = "subworkspace") {
    if (!reserve(budget_bytes)) {
      return WorkspaceAllocator(nullptr, 0, sub_label);
    }
    WorkspaceAllocator child(reinterpret_cast<void*>(base_ + used_), budget_bytes, sub_label);
    used_ += budget_bytes;
    return child;
  }

 private:
  bool reserve(size_t bytes) {
    if (used_ > total_ || bytes > total_ - used_) {
      overflowed_ = true;
      failed_offset_ = used_;
      failed_request_bytes_ = bytes;
      return false;
    }
    return true;
  }

  uintptr_t base_;
  size_t total_;
  size_t used_;
  bool overflowed_;
  size_t failed_offset_;
  size_t failed_request_bytes_;
  const char* label_;
};

}  // namespace omnidreams_native
