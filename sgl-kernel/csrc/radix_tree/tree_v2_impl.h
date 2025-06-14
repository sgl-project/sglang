#pragma once
#include <c10/util/irange.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "memory_pool.h"
#include "tree_v2.h"
#include "tree_v2_node.h"

namespace radix_tree_v2 {

using node_iterator_t = typename TreeNode::iterator_t;

struct RadixTree::Impl {
 public:
  Impl(bool disabled, bool use_hicache, std::size_t page_size, std::size_t host_size, std::size_t threshold)
      : m_root(),
        m_evictable_size(0),
        m_protected_size(0),
        m_cached_vec(),
        m_host_pool(page_size, host_size),
        disabled(disabled),
        use_hicache(use_hicache),
        page_size(page_size),
        threshold(threshold) {
    _assert(page_size > 0, "Page size must be greater than zero");
    m_root.ref_count = 1;  // root node is always protected
    m_cached_vec.reserve(page_size);
  }

  TreeNode* split_node(node_iterator_t it, std::size_t prefix) {
    // from `parent -> old_node` to `parent-> new_node -> old_node`
    // the prefix part of the old node is moved to the new node
    auto old_node_ptr = std::move(it->second);
    auto new_node_ptr = std::make_unique<TreeNode>();
    auto* old_node = old_node_ptr.get();
    auto* new_node = new_node_ptr.get();
    auto* parent = old_node->parent();
    // set up data structures
    TreeNode::split_prefix(new_node, old_node, prefix);
    // set up parent-child relationship
    add_child(new_node, std::move(old_node_ptr));
    add_child(parent, std::move(new_node_ptr), it);
    m_node_map[new_node->node_id] = new_node;  // add to the map
    return new_node;
  }

  // node: x -> [GPU]
  TreeNode* create_device_node(TreeNode* parent, token_vec_t vec, at::Tensor indices) {
    auto new_node_ptr = std::make_unique<TreeNode>();
    auto new_node = new_node_ptr.get();
    new_node_ptr->_unsafe_tokens() = std::move(vec);
    new_node_ptr->_unsafe_device_indices() = std::move(indices);
    m_evictable_size += new_node_ptr->length();
    add_child(parent, std::move(new_node_ptr));
    m_node_map[new_node->node_id] = new_node;  // add to the map
    return new_node;
  }

  // node: [CPU] -> x
  void remove_host_node(TreeNode* node) {
    _assert(node->on_cpu_only());
    m_host_pool.free(node->host_indices());
    node->parent()->erase_child(get_key(node));
    m_node_map.erase(node->node_id);  // remove from the map
  }

  // node: [GPU] -> x
  void remove_device_node(TreeNode* node) {
    _assert(node->on_gpu_only() && node->ref_count == 0);
    m_evictable_size -= node->length();
    node->parent()->erase_child(get_key(node));
    m_node_map.erase(node->node_id);  // remove from the map
  }

  // node: [CPU] -> [CPU + GPU]
  void update_device(TreeNode* node, at::Tensor new_indices) {
    _assert(node->on_cpu_only() && node->ref_count == 0);
    node->_unsafe_device_indices() = std::move(new_indices);
    m_evictable_size += node->length();
  }

  // node: [CPU + GPU] -> [CPU]
  void free_device(TreeNode* node) {
    _assert(node->on_both() && node->ref_count == 0);
    node->_unsafe_device_indices().reset();
    m_evictable_size -= node->length();
  }

  // increase the hit count of a node
  bool need_write_through(TreeNode* node) const {
    _assert(node->on_gpu());
    return use_hicache && node->hit_count >= threshold && !node->on_cpu();
  }

  // return number of nodes that are written through
  // this will make some nodes: [GPU] -> [CPU + GPU]
  std::size_t try_write_through(const std::vector<TreeNode*>& nodes) {
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

    // best effort to reserve space for the host indices
    return alloc_host(nodes, sizes);
  }

  /// @return (last node on cpu, matched prefix length on cpu)
  std::pair<TreeNode*, std::size_t> tree_walk(token_slice key) {
    // Some helper functions
    _assert(key.size() % page_size == 0, "Key should be page-aligned");

    std::size_t total_prefix_length = 0;
    TreeNode* node = &m_root;

    const auto now = std::chrono::steady_clock::now();
    while (key.size() > 0) {
      const auto it = node->find_child(get_key(key));
      if (it == node->end()) break;
      node = it->second.get();
      node->access(now);

      // at least `page_size` tokens are matched, and there may be more tokens to match
      // the return value prefix_length is no less than `page_size`
      const auto prefix_length = align(node->diff_key(key, page_size));
      total_prefix_length += prefix_length;

      // split the node if the prefix is not the whole token vector
      if (prefix_length < node->length()) {
        return {split_node(it, prefix_length), total_prefix_length};
      }

      // we have matched the whole key, continue to the next node
      key = key.split(prefix_length)[1];
    }

    return {node, total_prefix_length};
  }

  std::vector<TreeNode*> collect_leaves_host() {
    std::vector<TreeNode*> leaves;
    std::vector<TreeNode*> stack = {&m_root};
    while (!stack.empty()) {
      auto* node = stack.back();
      stack.pop_back();
      if (node->is_leaf_host()) {
        if (node->ref_count == 0) {
          leaves.push_back(node);
        }
      } else {
        for (const auto& [_, child] : *node) {
          stack.push_back(child.get());
        }
      }
    }
    return leaves;
  }

  std::vector<TreeNode*> collect_leaves_device() {
    // for non-hicache, every leaf device node is a leaf host node (since no backup on host)
    if (!use_hicache) return collect_leaves_host();
    std::vector<TreeNode*> leaves;
    std::vector<TreeNode*> stack = {&m_root};
    while (!stack.empty()) {
      auto* node = stack.back();
      stack.pop_back();
      if (!node->on_gpu()) continue;  // skip nodes that are not on GPU
      if (node->is_leaf_device()) {
        if (node->ref_count == 0) leaves.push_back(node);
      } else {  // if node is not on GPU, don't
        for (const auto& [_, child] : *node) {
          stack.push_back(child.get());
        }
      }
    }
    return leaves;
  }

  template <typename F>
  void walk_to_root(TreeNode* t, const F& f) {
    _assert(t != nullptr, "Cannot walk to root from a null node");
    while (!t->is_root()) {
      f(t);
      t = t->parent();
    }
  }

  template <typename F>
  TreeNode* walk_to_device(TreeNode* t, const F& f) {
    _assert(t != nullptr, "Cannot walk to root from a null node");
    while (!t->is_root() && !t->on_gpu()) {
      f(t);
      t = t->parent();
    }
    return t;  // return the first non-evicted node or the root node
  }

  void lock_ref(TreeNode* node, bool increment) {
    _assert(node != nullptr, "Cannot lock reference on a null node");
    _assert(node->on_gpu(), "Cannot lock reference on an evicted node");
    if (increment)
      walk_to_root(node, [this](TreeNode* n) {
        if (n->ref_count == 0) {
          m_evictable_size -= n->length();
          m_protected_size += n->length();
        }
        n->ref_count++;
      });
    else
      walk_to_root(node, [this](TreeNode* n) {
        _assert(n->ref_count != 0, "Cannot decrement reference count = zero");
        n->ref_count--;
        if (n->ref_count == 0) {
          m_protected_size -= n->length();
          m_evictable_size += n->length();
        }
      });
  }

  void lock_ref(NodeHandle node_ptr, bool increment) {
    return lock_ref(id2node(node_ptr), increment);
  }

  void lock(TreeNode* node) {
    return lock_ref(node, /*increment=*/true);
  }

  void unlock(TreeNode* node) {
    return lock_ref(node, /*increment=*/false);
  }

  std::size_t total_size() {
    std::size_t size = 0;
    std::vector<TreeNode*> stack = {&m_root};
    while (!stack.empty()) {
      auto* node = stack.back();
      stack.pop_back();
      size += node->length();
      for (const auto& [_, child] : *node)
        stack.push_back(child.get());
    }
    return size;
  }

  std::size_t evictable_size() const {
    return m_evictable_size;
  }

  std::size_t protected_size() const {
    return m_protected_size;
  }

  std::size_t align(std::size_t size) const {
    return (size / page_size) * page_size;  // align to page size
  }

  TreeNode* id2node(NodeHandle node_id) {
    auto it = m_node_map.find(node_id);
    _assert(it != m_node_map.end(), "Node not found in the map");
    return it->second;
  }

  void reset() {
    std::destroy_at(&m_root);    // destroy the root node
    ::new (&m_root) TreeNode{};  // reinitialize the root node
    m_evictable_size = 0;
    m_protected_size = 0;
    m_node_map.clear();
    m_host_pool.reset();  // clear the host memory pool
  }

 private:
  // some auxiliary functions
  token_vec_t& get_key(token_slice tokens) {
    _assert(tokens.size() >= page_size, "Key should be at least page-sized");
    tokens = tokens.split(page_size)[0];
    m_cached_vec.assign(tokens.begin(), tokens.end());
    return m_cached_vec;
  }

  token_vec_t& get_key(TreeNode* node) {
    return get_key(node->_unsafe_tokens());
  }

  void add_child(TreeNode* parent, std::unique_ptr<TreeNode>&& child) {
    parent->add_child(get_key(child.get()), std::move(child));
  }

  void add_child(TreeNode* parent, std::unique_ptr<TreeNode>&& child, node_iterator_t it) {
    parent->add_child(it, std::move(child));
  }

  std::size_t alloc_host(const std::vector<TreeNode*>& nodes, const std::vector<std::size_t>& needed_sizes);

  TreeNode m_root;               // root node of the tree
  std::size_t m_evictable_size;  // number of evictable tokens on GPU (lock ref = 0)
  std::size_t m_protected_size;  // number of protected tokens on GPU (lock ref > 0)

  token_vec_t m_cached_vec;       // cached vector of tokens for the current operation
  HiCacheMemoryPool m_host_pool;  // memory pool for host tensor indices

  std::unordered_map<std::size_t, TreeNode*> m_node_map;  // map of node keys to nodes

 public:
  // some public constant configurations (without m_ prefix)
  const bool disabled;          // whether the cache is enabled, or just a temporary cache
  const bool use_hicache;       // whether to use the HiCache for this tree
  const std::size_t page_size;  // size of each page in the cache
  const std::size_t threshold;  // threshold for write_through
};

}  // namespace radix_tree_v2
