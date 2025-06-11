#pragma once
#include <ATen/core/TensorBody.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cat.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common.h"

namespace radix_tree_v2 {

struct HiCacheMemoryPool {
 public:
  HiCacheMemoryPool(std::size_t page_size, std::size_t host_size) : m_free_slots(), host_size(host_size) {
    _assert(host_size % page_size == 0, "Host size must be a multiple of page size");
    this->reset();
  }

  at::Tensor alloc(std::size_t needed) {
    const auto size = this->available_size();
    _assert(needed < size, "Requested size exceeds host size");
    const auto remain = size - needed;
    auto values = m_free_slots.split_with_sizes({int64_t(needed), int64_t(remain)});
    m_free_slots = std::move(values[1]);
    return std::move(values[0]);
  }

  void free(at::Tensor indices) {
    m_free_slots = at::cat({m_free_slots, indices});
  }

  void reset() {
    m_free_slots = at::arange(host_size, at::kLong);
  }

  std::size_t available_size() const {
    return m_free_slots.size(0);
  }

 private:
  at::Tensor m_free_slots;  // free slots in the cache

 public:
  const std::size_t host_size;  // host indices
};

}  // namespace radix_tree_v2
