#pragma once
#include <c10/util/irange.h>

#include <chrono>
#include <cstddef>
#include <vector>

#include "common.h"
#include "memory_pool.h"
#include "tree_v2.h"
#include "tree_v2_node.h"

namespace radix_tree_v2 {

using node_iterator_t = typename TreeNode::iterator_t;

struct RadixTree::Impl {
 public:
  Impl(
      bool disabled,
      bool use_hicache,
      at::Device device,
      std::size_t page_size,
      std::size_t host_size,
      std::size_t threshold)
      : m_root(),
        m_evictable_size(0),
        m_protected_size(0),
        m_cached_vec(),
        m_host_pool(page_size, host_size),
        disabled(disabled),
        use_hicache(use_hicache),
        device(device),
        page_size(page_size),
        threshold(threshold) {
    _assert(page_size > 0, "Page size must be greater than zero");
    m_root.ref_count = 1;  // root node is always protected
    m_cached_vec.reserve(page_size);
  }

  // node: x -> [GPU]
  TreeNode* create_device_node(TreeNode* parent, token_vec_t vec, at::Tensor indices) {
    auto new_node = std::make_unique<TreeNode>();
    auto result = new_node.get();
    new_node->_unsafe_tokens() = std::move(vec);
    new_node->_unsafe_indices() = std::move(indices);
    m_evictable_size += new_node->length();
    add_child(parent, std::move(new_node));
    return result;
  }

  // node: [CPU] -> [CPU + GPU]
  void load_to_device(TreeNode* node, at::Tensor new_indices) {
    _assert(node->on_cpu_only());
    node->_unsafe_indices() = std::move(new_indices);
    m_evictable_size += node->length();
  }

  // node: [GPU] -> x
  void remove_device_node(TreeNode* node) {
    _assert(node->on_gpu_only());
    m_evictable_size -= node->length();
    node->parent()->erase_child(get_key(node));
  }

  // node: [CPU + GPU] -> [CPU]
  void offload_to_host(TreeNode* node) {
    _assert(node->on_both());
    node->_unsafe_indices().reset();
    m_evictable_size -= node->length();
  }

  // node: [CPU] -> x
  void remove_host_node(TreeNode* node) {
    _assert(node->on_cpu_only());
    m_host_pool.free(node->_unsafe_host_indices());
    node->parent()->erase_child(get_key(node));
  }

  // increase the hit count of a node
  bool need_write_through(TreeNode* node) const {
    _assert(node->on_gpu());
    return use_hicache && node->hit_count >= threshold && !node->on_cpu();
  }

  // return number of nodes that are written through
  // this will make some nodes: [GPU] -> [CPU + GPU]
  std::size_t try_write_through(const std::vector<NodeHandle>& nodes) {
    if (!use_hicache) return 0;  // no write-through if hierarchical cache is not used
    std::size_t remain_size = m_host_pool.available_size();
    std::vector<std::size_t> sizes;

    for (const auto& node_handle : nodes) {
      auto* node = pointer_cast(node_handle);
      _assert(need_write_through(node), "Node does not need write-through");
      if (const auto needed = node->length(); needed > remain_size) {
        sizes.push_back(needed - remain_size);
        remain_size = 0;  // no more space left
      } else {
        remain_size -= needed;
      }
    }

    // best effort to reserve space for the host indices
    const auto written_through = nodes.size() - sizes.size() + evict_host_batch(sizes);
    for (std::size_t i : c10::irange(written_through)) {
      auto* node = pointer_cast(nodes[i]);
      node->_unsafe_host_indices() = m_host_pool.alloc(node->length());
      lock_ref(node, /*increment=*/true);  // lock the node to be written through
    }

    return written_through;
  }

  // walk until the node is completely matched
  std::pair<TreeNode*, std::size_t> tree_walk(token_slice key) {
    // Some helper functions
    const auto _split_node = [this](node_iterator_t it, std::size_t prefix) {
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
      return new_node;
    };

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
        return {_split_node(it, prefix_length), total_prefix_length};
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

  std::size_t evict_host_batch(const std::vector<std::size_t>& needed_sizes);

  TreeNode m_root;               // root node of the tree
  std::size_t m_evictable_size;  // number of evictable tokens
  std::size_t m_protected_size;  // number of protected tokens

  token_vec_t m_cached_vec;       // cached vector of tokens for the current operation
  HiCacheMemoryPool m_host_pool;  // memory pool for host tensor indices

 public:
  // some public constant configurations (without m_ prefix)
  const bool disabled;          // whether the cache is enabled, or just a temporary cache
  const bool use_hicache;       // whether to use the HiCache for this tree
  const at::Device device;      // device type of the tree
  const std::size_t page_size;  // size of each page in the cache
  const std::size_t threshold;  // threshold for write_through
};

}  // namespace radix_tree_v2
