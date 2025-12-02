#pragma once
#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "common.h"

namespace radix_tree_v2 {

struct RadixTree {
 public:
  RadixTree(bool disabled, std::optional<std::size_t> host_size, std::size_t page_size, std::size_t threshold);
  ~RadixTree();

  // Trees should not be copied or moved, as they manage their own memory and state.
  RadixTree(const RadixTree&) = delete;
  RadixTree(RadixTree&&) = delete;
  RadixTree& operator=(const RadixTree&) = delete;
  RadixTree& operator=(RadixTree&&) = delete;

  /// @return (device indices that are matched, host indices length, device node, host node)
  std::tuple<std::vector<at::Tensor>, std::size_t, NodeHandle, NodeHandle> match_prefix(const token_vec_t& key);
  /// @return Device indices that need to be evicted (on python side).
  std::vector<at::Tensor> evict(std::size_t num_tokens);
  /// @brief (Un-)Lock a node.
  void lock_ref(NodeHandle node_id, bool increment /* increment or decrement */);
  /// @brief Update new key-value pair and try to perform write-through.
  std::tuple<std::vector<std::tuple<IOTicket, at::Tensor, at::Tensor>>, std::size_t>
  writing_through(const token_vec_t& key, at::Tensor value);
  /// @brief Load to device from host within a range of nodes.
  std::tuple<IOTicket, std::vector<at::Tensor>> loading_onboard(NodeHandle host_id, at::Tensor indices);
  /// @brief Commit a transaction of write-through.
  void commit_writing_through(IOTicket ticket, bool success);
  /// @brief Commit a transaction of load onboard.
  void commit_loading_onboard(IOTicket ticket, bool success);
  /// @brief Clear and reset the tree.
  void reset();

  /// @return How many size are still evictable (on device + not locked).
  std::size_t evictable_size() const;
  /// @return How many size are protected (locked).
  std::size_t protected_size() const;
  /// @return How many size are used on device.
  std::size_t total_size() const;

  /// @brief Print debug information of the tree.
  void debug_print() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

}  // namespace radix_tree_v2
