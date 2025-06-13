#include "tree_v2.h"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/util/irange.h>
#include <torch/library.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
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

RadixTree::RadixTree(bool disabled, bool use_hicache, int64_t page_size, int64_t host_size, int64_t threshold)
    : m_impl(std::make_unique<Impl>(disabled, use_hicache, page_size, host_size, threshold)) {}

RadixTree::~RadixTree() = default;

std::vector<NodeHandle> RadixTree::insert(const token_vec_t& _key, const at::Tensor value) {
  if (m_impl->disabled) return {};
  const auto key = token_slice{_key.data(), m_impl->align(_key.size())};

  // the nodes that are potentially written through to the host
  std::vector<TreeNode*> potential_write_nodes;

  // walk the tree to find the right place to insert
  const auto [host_node, total_prefix_length] = m_impl->tree_walk(key);

  if (total_prefix_length != key.size()) {
    const auto new_node = m_impl->create_device_node(
        host_node,
        {key.begin() + total_prefix_length, key.end()},
        value.slice(/*dim=*/0, total_prefix_length, key.size()));
    if (m_impl->need_write_through(new_node)) {
      potential_write_nodes.push_back(new_node);
    }
  }

  std::size_t offset = total_prefix_length;
  const auto device_node = m_impl->walk_to_device(host_node, [&](TreeNode* n) {
    m_impl->update_device(n, value.slice(/*dim=*/0, offset - n->length(), offset));
    offset = offset - n->length();
  });
  m_impl->walk_to_root(device_node, [&](TreeNode* n) {
    n->hit_count++;
    if (m_impl->need_write_through(n)) {
      potential_write_nodes.push_back(n);
    }
  });

  // don't write through if hicache is disabled (no host memory)
  if (!m_impl->use_hicache) return {};

  // reverse so that the nodes closer to the root are written back first
  std::reverse(potential_write_nodes.begin(), potential_write_nodes.end());
  const auto written_through = m_impl->try_write_through(potential_write_nodes);
  _assert(written_through <= potential_write_nodes.size());

  std::vector<NodeHandle> write_through_node_handles;
  write_through_node_handles.resize(written_through);
  for (const auto i : c10::irange(written_through)) {
    write_through_node_handles[i] = node2id(potential_write_nodes[i]);
  }
  return write_through_node_handles;
}

std::tuple<std::vector<at::Tensor>, int64_t, NodeHandle, NodeHandle> RadixTree::match_prefix(const token_vec_t& _key) {
  if (m_impl->disabled) return {{}, 0, 0, 0};

  const auto key = token_slice{_key.data(), m_impl->align(_key.size())};
  const auto [host_node, _] = m_impl->tree_walk(key);

  // walk up to the first non-evicted node
  std::vector<at::Tensor> indices{};
  const auto device_node = m_impl->walk_to_device(host_node, [&](TreeNode* n) {
    indices.push_back(n->host_indices());  // the last few indices are on the host
  });

  // collect all the device indices
  std::size_t num_devices = 0;
  m_impl->walk_to_root(device_node, [&](TreeNode* n) {
    ++num_devices;
    indices.push_back(n->device_indices());
  });
  std::reverse(indices.begin(), indices.end());

  return {std::move(indices), static_cast<int64_t>(num_devices), node2id(device_node), node2id(host_node)};
}

std::vector<at::Tensor> RadixTree::evict(int64_t _num_tokens) {
  const auto num_tokens = static_cast<std::size_t>(_num_tokens);
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

// similar to `evict`, but evicts nodes from the host memory (only) and batch the evictions
std::size_t RadixTree::Impl::alloc_host(const std::vector<TreeNode*>& nodes, const std::vector<std::size_t>& sizes) {
  _assert(sizes.size() == nodes.size(), "sizes and nodes must have the same size");
  _assert(use_hicache, "evict_host_batch called without hicache enabled");
  if (sizes.empty()) return 0;
  auto heap = std::priority_queue{cmp, collect_leaves_host()};
  std::size_t num_evict = 0;
  for (const auto i : c10::irange(sizes.size())) {
    const auto num_tokens = sizes[i];
    while (num_evict < num_tokens && !heap.empty()) {
      const auto node = heap.top();
      heap.pop();
      // skip nodes that are on the GPU or are being written through (i.e. indices protected)
      // the first condition is our policy, which can be changed in the future
      // while the second one ensures the correctness of the eviction
      if (node->on_gpu() || node->is_writting_through) continue;
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
    _assert(!node->is_writting_through, "Node is already being written through");
    node->is_writting_through = true;
    node->_unsafe_host_indices() = m_host_pool.alloc(node->length());
    lock_ref(node, /*increment=*/true);  // protect the node from eviction
  }

  return sizes.size();  // all nodes were successfully allocated
}

void RadixTree::lock_ref(NodeHandle node_id, bool increment) {
  if (m_impl->disabled) return;
  m_impl->lock_ref(node_id, increment);
}

int64_t RadixTree::evictable_size() const {
  return static_cast<int64_t>(m_impl->evictable_size());
}

int64_t RadixTree::protected_size() const {
  return static_cast<int64_t>(m_impl->protected_size());
}

int64_t RadixTree::total_size() const {
  return static_cast<int64_t>(m_impl->total_size());
}

std::tuple<at::Tensor, at::Tensor> RadixTree::start_write_through(NodeHandle node_id) {
  if (m_impl->disabled) return {};
  auto node = m_impl->id2node(node_id);
  // the node must be 1. writting through 2. on both device and host 3. locked (ref_count > 0)
  _assert(node->is_writting_through && node->on_both() && node->ref_count > 0, "Not a valid node for write through");
  return {node->device_indices(), node->host_indices()};
}

void RadixTree::commit_write_through(NodeHandle node_id, bool success) {
  if (m_impl->disabled) return;
  // the node must be 1. writting through 2. on both device and host 3. locked (ref_count > 0)
  auto node = m_impl->id2node(node_id);
  _assert(node->is_writting_through && node->on_both() && node->ref_count > 0, "Not a valid node for write through");
  node->is_writting_through = false;
  // if the write is not cancelled, we can safely reset the hit count
  if (success) node->hit_count = 0;
  m_impl->lock_ref(node, /*increment=*/false);  // unlock the node
}

}  // namespace radix_tree_v2

// TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
//   using namespace radix_tree_v2;
//   m.class_<RadixTree>("RadixTree")
//       .def(torch::init<bool, bool, int64_t, int64_t, int64_t>())
//       .def("insert", &RadixTree::insert)
//       .def("match_prefix", &RadixTree::match_prefix)
//       .def("evict", &RadixTree::evict)
//       .def("lock_ref", &RadixTree::lock_ref)
//       .def("evictable_size", &RadixTree::evictable_size)
//       .def("protected_size", &RadixTree::protected_size)
//       .def("total_size", &RadixTree::total_size)
//       .def("start_write_through", &RadixTree::start_write_through)
//       .def("commit_write_through", &RadixTree::commit_write_through);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace radix_tree_v2;
  namespace py = pybind11;
  py::class_<RadixTree>(m, "RadixTree")
      .def(py::init<bool, bool, int64_t, int64_t, int64_t>())
      .def("insert", &RadixTree::insert)
      .def("match_prefix", &RadixTree::match_prefix)
      .def("evict", &RadixTree::evict)
      .def("lock_ref", &RadixTree::lock_ref)
      .def("evictable_size", &RadixTree::evictable_size)
      .def("protected_size", &RadixTree::protected_size)
      .def("total_size", &RadixTree::total_size)
      .def("start_write_through", &RadixTree::start_write_through)
      .def("commit_write_through", &RadixTree::commit_write_through);
}
