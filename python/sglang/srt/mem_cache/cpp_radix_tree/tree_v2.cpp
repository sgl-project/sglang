#include "tree_v2.h"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/util/irange.h>

#include <cstddef>
#include <memory>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common.h"
#include "tree_v2_impl.h"
#include "tree_v2_node.h"

namespace radix_tree_v2 {

static NodeHandle node2id(TreeNode* node) {
  return node->node_id;
}

// compare function for the TreeNode pointers based on their time
// we use LRU, so we want to evict the least recently used nodes
// since std::priority_queue is a max-heap, we need to reverse the comparison
static constexpr auto cmp = [](TreeNode* lhs, TreeNode* rhs) { return lhs->time() > rhs->time(); };

RadixTree::RadixTree(bool disabled, std::optional<std::size_t> host_size, std::size_t page_size, std::size_t threshold)
    : m_impl(std::make_unique<Impl>(disabled, host_size.has_value(), page_size, host_size.value_or(0), threshold)) {}

RadixTree::~RadixTree() = default;

std::tuple<std::vector<at::Tensor>, std::size_t, NodeHandle, NodeHandle>
RadixTree::match_prefix(const token_vec_t& _key) {
  if (m_impl->disabled) return {};

  const auto key = token_slice{_key.data(), m_impl->align(_key.size())};
  const auto [host_node, _] = m_impl->tree_walk(key);

  // walk up to the first non-evicted node
  std::size_t host_hit_length = 0;
  const auto device_node = host_node;

  // collect all the device indices
  std::vector<at::Tensor> indices{};
  walk_to_root(device_node, [&](TreeNode* n) { indices.push_back(n->device_indices()); });
  std::reverse(indices.begin(), indices.end());

  return {std::move(indices), host_hit_length, node2id(device_node), node2id(host_node)};
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
    // when ref_count == 0, can't be writing through
    _assert(node->on_gpu() && node->ref_count == 0);
    if (!node->is_io_free()) continue;  // skip nodes that are undergoing IO (i.e. indices protected)
    evicted_values.push_back(node->device_indices());
    num_evict += node->length();
    const auto parent = node->parent();
    m_impl->remove_device_node(node);
    if (parent->is_leaf_device() && parent->ref_count == 0)
      heap.push(parent);  // push parent to the heap if it is now a free leaf
  }

  return evicted_values;
}

std::tuple<std::vector<std::tuple<IOTicket, at::Tensor, at::Tensor>>, std::size_t>
RadixTree::writing_through(const token_vec_t& _key, at::Tensor value) {
  if (m_impl->disabled) return {};
  _assert(_key.size() == std::size_t(value.size(0)), "Key and value must have the same size");

  // just align the key to the page size, clip the unaligned tail
  const auto key = token_slice{_key.data(), m_impl->align(_key.size())};

  // walk the tree to find the right place to insert
  const auto [host_node, host_prefix_length] = m_impl->tree_walk(key);

  // insert and create a new node if the remaining part of the key is not empty
  if (host_prefix_length != key.size()) {
    m_impl->create_device_node(
        host_node,
        {key.begin() + host_prefix_length, key.end()},
        value.slice(/*dim=*/0, host_prefix_length, key.size()));
  }

  // add the hit count for the device node
  walk_to_root(host_node, [&](TreeNode* n) { n->hit_count++; });

  std::vector<std::tuple<IOTicket, at::Tensor, at::Tensor>> result;

  // don't write through if hicache is disabled (no host memory), fast path
  if (!m_impl->use_hicache) return {std::move(result), host_prefix_length};
  throw std::runtime_error("Not implemented yet");
}

std::tuple<IOTicket, std::vector<at::Tensor>> RadixTree::loading_onboard(NodeHandle, at::Tensor) {
  if (m_impl->disabled) return {};
  throw std::runtime_error("Not implemented yet");
}

void RadixTree::commit_writing_through(IOTicket, bool) {
  if (m_impl->disabled) return;
  throw std::runtime_error("Not implemented yet");
}

void RadixTree::commit_loading_onboard(IOTicket, bool) {
  if (m_impl->disabled) return;
  throw std::runtime_error("Not implemented yet");
}

void RadixTree::reset() {
  m_impl->reset();
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
