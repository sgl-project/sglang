#pragma once
#include <ATen/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

namespace radix_tree_v2 {

using token_t = std::int32_t;
using token_vec_t = std::vector<token_t>;
using NodeHandle = std::uintptr_t;

struct RadixTree {
 public:
  RadixTree(bool disabled, at::Device device, std::size_t page_size, std::size_t host_size);
  ~RadixTree() = default;

  // Trees should not be copied or moved, as they manage their own memory and state.
  RadixTree(const RadixTree&) = delete;
  RadixTree(RadixTree&&) = delete;
  RadixTree& operator=(const RadixTree&) = delete;
  RadixTree& operator=(RadixTree&&) = delete;

  /// @return Nodes that are locked and require write-through.
  std::vector<NodeHandle> insert(const token_vec_t& key, at::Tensor value);
  /// @return (device indices that are matched, device node, host node)
  std::tuple<std::vector<at::Tensor>, NodeHandle, NodeHandle> match_prefix(const token_vec_t& key);
  /// @return Device indices that need to be evicted (on python side).
  std::vector<at::Tensor> evict(std::size_t num_tokens);
  /// @brief (Un-)Lock a node.
  void lock_ref(NodeHandle node_id, bool increment /* increment or decrement */);

  /// @return How many size are still evictable (on device + not locked).
  std::size_t evictable_size() const;
  /// @return How many size are protected (locked).
  std::size_t protected_size() const;
  /// @return How many size are used on device.
  std::size_t total_size() const;

  struct Impl;

 private:
  std::shared_ptr<Impl> m_impl;
};

}  // namespace radix_tree_v2
