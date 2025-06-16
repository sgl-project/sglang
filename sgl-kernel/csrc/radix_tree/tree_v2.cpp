#include "tree_v2.h"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/util/irange.h>

#include <cstddef>
#include <memory>
#include <queue>
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
static constexpr auto cmp = [](TreeNode* lhs, TreeNode* rhs) { return lhs->time() < rhs->time(); };

RadixTree::RadixTree(bool disabled, std::optional<std::size_t> host_size, std::size_t page_size, std::size_t threshold)
    : m_impl(std::make_unique<Impl>(disabled, host_size.has_value(), page_size, host_size.value_or(0), threshold)) {}

RadixTree::~RadixTree() = default;

std::tuple<std::vector<at::Tensor>, std::size_t, NodeHandle, NodeHandle>
RadixTree::match_prefix(const token_vec_t& _key) {
  if (m_impl->disabled) return {};

  const auto key = token_slice{_key.data(), m_impl->align(_key.size())};
  const auto [host_node, _] = m_impl->tree_walk(key);

  // walk up to the first non-evicted node
  std::size_t host_length = 0;
  const auto device_node = m_impl->walk_to_device(host_node, [&](TreeNode* n) {
    host_length += n->length();  // accumulate the length of the host indices
  });

  // collect all the device indices
  std::vector<at::Tensor> indices{};
  m_impl->walk_to_root(device_node, [&](TreeNode* n) { indices.push_back(n->device_indices()); });
  std::reverse(indices.begin(), indices.end());

  return {std::move(indices), host_length, node2id(device_node), node2id(host_node)};
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
    if (!node->on_cpu()) {
      m_impl->remove_device_node(node);
    } else {
      m_impl->free_device(node);
    }
    if (parent->is_leaf_device() && parent->ref_count == 0)
      heap.push(parent);  // push parent to the heap if it is now a free leaf
  }

  return evicted_values;
}

std::size_t RadixTree::Impl::try_write_through(const std::vector<TreeNode*>& nodes) {
  if (!use_hicache) return 0;  // no write-through if hierarchical cache is not used
  std::size_t remain_size = m_host_pool.available_size();
  std::vector<std::size_t> sizes(nodes.size());

  for (const auto i : c10::irange(nodes.size())) {
    const auto node = nodes[i];
    _assert(need_write_through(node), "Node does not need write-through");
    if (const auto needed = node->length(); needed > remain_size) {
      sizes[i] = needed - remain_size;
      remain_size = 0;  // no more space left
    } else {
      sizes[i] = 0;
      remain_size -= needed;
    }
  }

  _assert(sizes.size() == nodes.size(), "sizes and nodes must have the same size");
  _assert(use_hicache, "evict_host_batch called without hicache enabled");

  if (sizes.empty()) return 0;

  // The following code, similar to `evict`, evicts nodes from the host memory in a batch manner.
  auto heap = std::priority_queue{cmp, collect_leaves_host()};
  std::size_t num_evict = 0;
  for (const auto i : c10::irange(sizes.size())) {
    const auto num_tokens = sizes[i];
    while (num_evict < num_tokens && !heap.empty()) {
      const auto node = heap.top();
      heap.pop();
      // skip nodes that are on the GPU or are undergoing IO (i.e. indices protected)
      // the first condition is our policy, which can be changed in the future
      // while the second one ensures the correctness of the eviction
      if (node->on_gpu() || !node->is_io_free()) continue;
      num_evict += node->length();
      const auto parent = node->parent();
      remove_host_node(node);
      if (parent->is_leaf_host() && parent->ref_count == 0)
        heap.push(parent);  // push parent to the heap if it is now a free leaf
    }

    if (num_evict < num_tokens) return i;  // not enough host memory for writing through
    num_evict -= num_tokens;
    // set the node as being written through
    const auto node = nodes[i];
    node->_unsafe_host_indices() = m_host_pool.alloc(node->length());
    const auto [ticket, iterator] = this->new_io_ticket();
    node->start_device_to_host(ticket, /*locked=*/true);
    this->register_io_ticket(node, iterator);
    lock(node);
  }

  return sizes.size();  // all nodes were successfully allocated
}

std::tuple<std::vector<std::tuple<IOTicket, at::Tensor, at::Tensor>>, std::size_t>
RadixTree::writing_through(const token_vec_t& _key, at::Tensor value) {
  if (m_impl->disabled) return {};
  _assert(_key.size() == std::size_t(value.size(0)), "Key and value must have the same size");
  const auto key = token_slice{_key.data(), m_impl->align(_key.size())};

  // the nodes that are potentially written through to the host
  std::vector<TreeNode*> potential_write_nodes;

  // walk the tree to find the right place to insert
  const auto [host_node, host_prefix_length] = m_impl->tree_walk(key);

  if (host_prefix_length != key.size()) {
    const auto new_node = m_impl->create_device_node(
        host_node,
        {key.begin() + host_prefix_length, key.end()},
        value.slice(/*dim=*/0, host_prefix_length, key.size()));
    if (m_impl->need_write_through(new_node)) {
      potential_write_nodes.push_back(new_node);
    }
  }

  std::size_t offset = host_prefix_length;
  const auto device_node = m_impl->walk_to_device(host_node, [&](TreeNode* n) {
    m_impl->update_device(n, value.slice(/*dim=*/0, offset - n->length(), offset));
    offset = offset - n->length();
  });

  std::size_t device_prefix_length = 0;
  m_impl->walk_to_root(device_node, [&](TreeNode* n) {
    device_prefix_length += n->length();
    n->hit_count++;
    if (m_impl->need_write_through(n)) {
      potential_write_nodes.push_back(n);
    }
  });
  _assert(device_prefix_length == offset, "Something goes wrong...");

  std::vector<std::tuple<IOTicket, at::Tensor, at::Tensor>> result;
  // don't write through if hicache is disabled (no host memory), fast path
  if (!m_impl->use_hicache) return {std::move(result), device_prefix_length};

  // reverse so that the nodes closer to the root are written back first
  std::reverse(potential_write_nodes.begin(), potential_write_nodes.end());
  const auto num_success = m_impl->try_write_through(potential_write_nodes);
  _assert(num_success <= potential_write_nodes.size());

  // fill the result with the tickets and indices
  result.resize(num_success);
  for (const auto i : c10::irange(num_success)) {
    const auto node = potential_write_nodes[i];
    const auto ticket = node->io_ticket();
    _assert(ticket.has_value(), "Ticket should be valid for the node");
    result[i] = {ticket.value(), node->device_indices(), node->host_indices()};
  }

  return {std::move(result), device_prefix_length};
}

std::tuple<IOTicket, std::vector<at::Tensor>>
RadixTree::loading_onboard(NodeHandle device_id, NodeHandle host_id, at::Tensor value) {
  if (m_impl->disabled) return {};
  _assert(device_id != host_id, "device and host must represent a range of indices");

  const auto old_host_node = m_impl->id2node(host_id);
  const auto [ticket, iterator] = m_impl->new_io_ticket();

  std::vector<at::Tensor> indices;
  std::size_t offset = value.size(0);
  const auto new_device_node = m_impl->walk_to_device(old_host_node, [&](TreeNode* n) {
    indices.push_back(n->host_indices());
    m_impl->update_device(n, value.slice(/*dim=*/0, offset - n->length(), offset));
    offset = offset - n->length();
    // we only lock the furthermost host node, since it will automatically lock all the parent nodes
    bool locked = n == old_host_node;
    n->start_host_to_device(ticket, /*locked=*/locked);
    m_impl->register_io_ticket(n, iterator);
  });
  // we can only lock node after it has device indices
  m_impl->lock(old_host_node);

  std::reverse(indices.begin(), indices.end());
  const auto old_device_node = m_impl->id2node(device_id);
  _assert(offset == 0, "Offset should be zero after updating device indices");
  _assert(old_device_node == new_device_node, "Device node is not the same as the previously matched");

  return {ticket, std::move(indices)};
}

void RadixTree::commit_writing_through(IOTicket ticket, bool success) {
  if (m_impl->disabled) return;
  m_impl->complete_io_writing_through(ticket, success);
}

void RadixTree::commit_loading_onboard(IOTicket ticket, bool success) {
  if (m_impl->disabled) return;
  m_impl->complete_io_loading_onboard(ticket, success);
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
