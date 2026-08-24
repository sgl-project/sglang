#pragma once
#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "common.h"

namespace radix_tree_v2 {

struct RadixTree {
 public:
  RadixTree(
      bool disabled,
      std::optional<std::size_t> host_size,
      std::size_t page_size,
      std::size_t threshold,
      std::size_t sliding_window_size = 0);
  ~RadixTree();

  // Trees should not be copied or moved, as they manage their own memory and state.
  RadixTree(const RadixTree&) = delete;
  RadixTree(RadixTree&&) = delete;
  RadixTree& operator=(const RadixTree&) = delete;
  RadixTree& operator=(RadixTree&&) = delete;

  /// @return (device indices that are matched, host indices length, device node, host node)
  std::tuple<std::vector<at::Tensor>, std::size_t, NodeHandle, NodeHandle> match_prefix(token_slice key);
  std::tuple<std::vector<at::Tensor>, NodeHandle, std::size_t> match_prefix_swa(token_slice key);
  /// @return Device indices that need to be evicted (on python side).
  std::vector<at::Tensor> evict(std::size_t num_tokens);
  /// @brief (Un-)Lock a node.
  void lock_ref(NodeHandle node_id, bool increment /* increment or decrement */);
  std::tuple<std::optional<std::size_t>, std::vector<NodeHandle>>
  lock_ref_swa(NodeHandle node_id, bool lock_full, bool lock_swa);
  void unlock_ref_swa(
      NodeHandle node_id,
      bool unlock_full,
      bool unlock_swa,
      std::optional<std::size_t> swa_uuid,
      const std::vector<NodeHandle>& skip_swa_nodes);
  std::tuple<std::vector<at::Tensor>, std::size_t>
  unlock_swa_only(NodeHandle node_id, std::optional<std::size_t> swa_uuid);
  /// @brief Update new key-value pair and try to perform write-through.
  std::tuple<std::vector<std::tuple<IOTicket, at::Tensor, at::Tensor>>, std::size_t, NodeHandle>
  writing_through(token_slice key, at::Tensor value);
  std::tuple<
      std::size_t,
      NodeHandle,
      std::vector<at::Tensor>,
      std::vector<std::tuple<NodeHandle, at::Tensor>>,
      std::vector<std::tuple<NodeHandle, at::Tensor, at::Tensor>>>
  writing_through_swa(token_slice key, at::Tensor value, std::size_t prev_prefix_len, std::size_t swa_evicted_seqlen);
  /// Collect an already-inserted prefix and acquire its FULL/SWA locks in one
  /// native call.  This avoids a second key traversal and Python/C++ crossing.
  std::tuple<std::vector<at::Tensor>, NodeHandle, std::size_t, std::optional<std::size_t>, std::vector<NodeHandle>>
  match_node_and_lock(NodeHandle node_id, bool lock_full, bool lock_swa);
  /// Collect only [range_start, range_end) from an already inserted prefix,
  /// then acquire its locks.  best_prefix_length describes the complete
  /// usable prefix, allowing Python to reuse the insertion value for the
  /// untouched ranges instead of concatenating the whole root path again.
  std::tuple<
      std::vector<at::Tensor>,
      NodeHandle,
      std::size_t,
      std::size_t,
      std::optional<std::size_t>,
      std::vector<NodeHandle>>
  match_node_range_and_lock(
      NodeHandle node_id, std::size_t range_start, std::size_t range_end, bool lock_full, bool lock_swa);
  void set_swa_value(NodeHandle node_id, at::Tensor value);
  at::Tensor get_swa_value(NodeHandle node_id) const;
  std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::size_t, std::size_t>
  evict_component(std::size_t component_type, std::size_t num_tokens);
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
  std::size_t component_evictable_size(std::size_t component_type) const;
  std::size_t component_protected_size(std::size_t component_type) const;
  /// @return All device indices owned by the tree, grouped by tree node.
  std::vector<at::Tensor> all_values() const;
  std::unordered_map<std::string, std::uint64_t> debug_stats() const;

  /// @brief Print debug information of the tree.
  void debug_print() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

}  // namespace radix_tree_v2
