#include "tree_v2.h"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>

#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "common.h"
#include "tree_v2_impl.h"

namespace radix_tree_v2 {

// compare function for the TreeNode pointers based on their time
static constexpr auto cmp = [](TreeNode* lhs, TreeNode* rhs) { return lhs->time() < rhs->time(); };

std::vector<NodeHandle> RadixTree::insert(const token_vec_t& _key, const at::Tensor value) {
  if (m_impl->disabled) return {};
  const auto key = token_slice{_key.data(), m_impl->align(_key.size())};

  // the nodes that are potentially written through to the host
  std::vector<NodeHandle> potential_write_nodes;

  // walk the tree to find the right place to insert
  const auto [host_node, total_prefix_length] = m_impl->tree_walk(key);

  if (total_prefix_length != key.size()) {
    const auto new_node = m_impl->create_device_node(
        host_node,
        {key.begin(), key.begin() + total_prefix_length},
        value.slice(/*dim=*/0, total_prefix_length, key.size()));
    if (m_impl->need_write_through(new_node)) {
      potential_write_nodes.push_back(pointer_cast(new_node));
    }
  }

  std::size_t offset = total_prefix_length;
  const auto device_node = m_impl->walk_to_device(host_node, [&](TreeNode* n) {
    m_impl->load_to_device(n, value.slice(/*dim=*/0, offset - n->length(), offset));
    offset = offset - n->length();
  });
  m_impl->walk_to_root(device_node, [&](TreeNode* n) {
    n->hit_count++;
    if (m_impl->need_write_through(n)) {
      potential_write_nodes.push_back(pointer_cast(n));
    }
  });

  // reverse so that the nodes closer to the root are written back first
  std::reverse(potential_write_nodes.begin(), potential_write_nodes.end());
  const auto written_through = m_impl->try_write_through(potential_write_nodes);
  potential_write_nodes.resize(written_through);
  return potential_write_nodes;
}

std::tuple<std::vector<at::Tensor>, NodeHandle, NodeHandle> RadixTree::match_prefix(const token_vec_t& _key) {
  if (m_impl->disabled) return {{}, 0, 0};

  const auto key = token_slice{_key.data(), m_impl->align(_key.size())};
  const auto [host_node, _] = m_impl->tree_walk(key);

  // walk up to the first non-evicted node
  const auto device_node = m_impl->walk_to_device(host_node, [](auto) {});

  // collect all the device indices
  std::vector<at::Tensor> device_indices{};
  m_impl->walk_to_root(device_node, [&](TreeNode* n) { device_indices.push_back(n->device_indices()); });
  std::reverse(device_indices.begin(), device_indices.end());

  return {std::move(device_indices), pointer_cast(device_node), pointer_cast(host_node)};
}

std::vector<at::Tensor> RadixTree::evict(std::size_t num_tokens) {
  if (m_impl->disabled || num_tokens == 0) return {};
  auto heap = std::priority_queue{cmp, m_impl->collect_leaves_device()};
  std::vector<at::Tensor> evicted_values;
  // evict nodes until we reach the desired number of tokens
  std::size_t num_evict = 0;
  while (num_evict < num_tokens && !heap.empty()) {
    const auto node = heap.top();
    heap.pop();
    _assert(node->on_gpu());
    evicted_values.push_back(node->device_indices());
    num_evict += node->length();
    const auto parent = node->parent();
    if (!node->on_cpu()) {
      m_impl->remove_device_node(node);
    } else {
      m_impl->offload_to_host(node);
    }
    if (parent->is_leaf_device() && parent->ref_count == 0)
      heap.push(parent);  // push parent to the heap if it is now a free leaf
  }

  return evicted_values;
}

// similar to `evict`, but evicts nodes from the host memory (only) and batch the evictions
std::size_t RadixTree::Impl::evict_host_batch(const std::vector<std::size_t>& sizes) {
  _assert(use_hicache, "evict_host_batch called without hicache enabled");
  if (sizes.empty()) return 0;
  auto heap = std::priority_queue{cmp, collect_leaves_host()};
  std::size_t num_evict = 0;
  std::size_t num_success = 0;
  for (const auto num_tokens : sizes) {
    while (num_evict < num_tokens && !heap.empty()) {
      const auto node = heap.top();
      heap.pop();
      if (node->on_gpu()) continue;  // skip nodes that are on device
      num_evict += node->length();
      remove_host_node(node);
    }
    if (num_evict < num_tokens) break;
    num_evict -= num_tokens;
    num_success += 1;
  }
  return num_success;
}

void RadixTree::lock_ref(NodeHandle node_id, bool increment) {
  if (m_impl->disabled) return;
  m_impl->lock_ref(node_id, increment);
}

std::size_t RadixTree::evictable_size() const {
  return m_impl->evictable_size();
}

std::size_t RadixTree::protected_size() const {
  return m_impl->protected_size();
}

std::size_t RadixTree::total_size() const {
  return m_impl->total_size();
}

}  // namespace radix_tree_v2
