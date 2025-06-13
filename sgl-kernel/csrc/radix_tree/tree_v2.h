#pragma once
#include <ATen/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/ivalue.h>
#include <c10/core/Device.h>

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "common.h"

namespace radix_tree_v2 {

struct RadixTree : torch::CustomClassHolder {
 public:
  RadixTree(bool disabled, bool use_hicache, int64_t page_size, int64_t host_size, int64_t threshold);
  ~RadixTree();

  // Trees should not be copied or moved, as they manage their own memory and state.
  RadixTree(const RadixTree&) = delete;
  RadixTree(RadixTree&&) = delete;
  RadixTree& operator=(const RadixTree&) = delete;
  RadixTree& operator=(RadixTree&&) = delete;

  /// @return Nodes that are locked and require write-through.
  std::vector<NodeHandle> insert(const token_vec_t& key, at::Tensor value);
  /// @return (device indices that are matched, device node, host node)
  std::tuple<std::vector<at::Tensor>, int64_t, NodeHandle, NodeHandle> match_prefix(const token_vec_t& key);
  /// @return Device indices that need to be evicted (on python side).
  std::vector<at::Tensor> evict(int64_t num_tokens);
  /// @brief (Un-)Lock a node.
  void lock_ref(NodeHandle node_id, bool increment /* increment or decrement */);

  /// @brief Start a new transaction and return (device, host) indices.
  std::tuple<at::Tensor, at::Tensor> start_write_through(NodeHandle node_id);

  /// @brief Commit a transaction and return the device index.
  void commit_write_through(NodeHandle node_id, bool success);

  /// @return How many size are still evictable (on device + not locked).
  int64_t evictable_size() const;
  /// @return How many size are protected (locked).
  int64_t protected_size() const;
  /// @return How many size are used on device.
  int64_t total_size() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

}  // namespace radix_tree_v2
