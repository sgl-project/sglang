#pragma once
#include <c10/util/irange.h>

#include <chrono>
#include <cstddef>
#include <iosfwd>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "tree_v2.h"
#include "tree_v2_node.h"

namespace radix_tree_v2 {

using node_iterator_t = typename TreeNode::iterator_t;

struct RadixTree::Impl {
 public:
  Impl(bool disabled, bool use_hicache, std::size_t page_size, std::size_t host_size, std::size_t threshold)
      : m_root(/*node_id_=*/0),
        m_evictable_size(0),
        m_protected_size(0),
        m_cached_vec(),
        m_node_map(),
        m_node_counter(1),  // start from 1 to avoid confusion with root node
        disabled(disabled),
        use_hicache(use_hicache),
        page_size(page_size),
        threshold(threshold) {
    _assert(page_size > 0, "Page size must be greater than zero");
    _assert(use_hicache == (host_size > 0), "Hierarchical cache is enabled iff host size > 0");
    m_root.ref_count = 1;                  // root node is always protected
    m_cached_vec.reserve(page_size);       // to avoid repeated allocations
    m_node_map[m_root.node_id] = &m_root;  // add root to the map
  }

  TreeNode* split_node(node_iterator_t iterator, std::size_t prefix_length) {
    // from `parent -> old_node` to `parent-> new_node -> old_node`
    // the prefix part of the old node is moved to the new node
    auto old_node_ptr = std::move(iterator->second);
    auto new_node_ptr = std::make_unique<TreeNode>(m_node_counter++);
    auto* old_node = old_node_ptr.get();
    auto* new_node = new_node_ptr.get();
    auto* parent = old_node->parent();
    // set up data structures
    split_prefix(new_node, old_node, prefix_length);
    // set up parent-child relationship
    add_child(new_node, std::move(old_node_ptr));
    add_child(parent, std::move(new_node_ptr), iterator);
    m_node_map[new_node->node_id] = new_node;  // add to the map
    return new_node;
  }

  // node: x -> [GPU]
  TreeNode* create_device_node(TreeNode* parent, token_vec_t vec, at::Tensor indices) {
    auto new_node_ptr = std::make_unique<TreeNode>(m_node_counter++);
    auto new_node = new_node_ptr.get();
    new_node_ptr->_unsafe_tokens() = std::move(vec);
    new_node_ptr->_unsafe_device_indices() = std::move(indices);
    m_evictable_size += new_node_ptr->length();
    add_child(parent, std::move(new_node_ptr));
    m_node_map[new_node->node_id] = new_node;  // add to the map
    return new_node;
  }

  // node: [GPU] -> x
  void remove_device_node(TreeNode* node) {
    _assert(node->on_gpu_only() && node->ref_count == 0);
    m_evictable_size -= node->length();
    node->parent()->erase_child(get_key(node));
    m_node_map.erase(node->node_id);  // remove from the map
  }

  /**
   * @brief Walk the tree to find the node that matches the key.
   * If the key partially matches a node, it will split that node.
   * @return A pair containing the last node that matches the key and
   * the total prefix length matched (on gpu and cpu) so far.
   */
  std::pair<TreeNode*, std::size_t> tree_walk(token_slice key) {
    _assert(key.size() % page_size == 0, "Key should be page-aligned");

    std::size_t total_prefix_length = 0;
    TreeNode* node = &m_root;

    const auto now = std::chrono::steady_clock::now();
    while (key.size() > 0) {
      const auto iterator = node->find_child(get_key(key));
      if (iterator == node->end()) break;

      // walk to the child node
      node = iterator->second.get();

      // at least `page_size` tokens are matched, and there may be more tokens to match
      // the return value prefix_length is no less than `page_size`
      const auto prefix_length = align(node->diff_key(key, page_size) + page_size);
      total_prefix_length += prefix_length;

      // split the node if the prefix is not the whole token vector
      if (prefix_length < node->length()) {
        return {split_node(iterator, prefix_length), total_prefix_length};
      }

      // we have matched the whole key, continue to the next node
      node->access(now);
      key = key.subspan(prefix_length);
    }

    return {node, total_prefix_length};
  }

  std::vector<TreeNode*> collect_leaves() const {
    std::vector<TreeNode*> leaves;
    std::vector<TreeNode*> stack = {};
    for (const auto& [_, child] : m_root) {
      stack.push_back(child.get());
    }
    while (!stack.empty()) {
      const auto node = stack.back();
      stack.pop_back();
      if (node->is_leaf()) {
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

  std::vector<TreeNode*> collect_leaves_device() const {
    // for non-hicache, every leaf device node is a leaf node (since no backup on host)
    if (!use_hicache) return collect_leaves();
    std::vector<TreeNode*> leaves;
    std::vector<TreeNode*> stack = {};
    for (const auto& [_, child] : m_root) {
      stack.push_back(child.get());
    }
    while (!stack.empty()) {
      const auto node = stack.back();
      stack.pop_back();
      if (!node->on_gpu()) continue;  // skip nodes that are not on GPU
      if (node->is_leaf_device()) {
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

  void lock_ref(TreeNode* node, bool increment) {
    if (node->is_root()) return;  // skip root node
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

  std::size_t total_size() const {
    std::size_t size = 0;
    std::vector<const TreeNode*> stack = {&m_root};
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

  TreeNode* id2node(NodeHandle node_id) const {
    const auto iterator = m_node_map.find(node_id);
    _assert(iterator != m_node_map.end(), "Node not found in the map");
    return iterator->second;
  }

  void reset() {
    _assert(m_root.ref_count == 1, "Root node must be protected during reset");
    m_node_counter = 1;  // reset node counter
    m_root.root_reset();
    m_evictable_size = 0;
    m_protected_size = 0;
    m_node_map.clear();
    m_node_map[m_root.node_id] = &m_root;  // re-add root to the map
  }

  void debug_print(std::ostream& os) const;

 private:
  // some auxiliary functions
  token_vec_t& get_key(token_slice tokens) {
    _assert(tokens.size() >= page_size, "Key should be at least page-sized");
    tokens = tokens.subspan(0, page_size);
    m_cached_vec.assign(tokens.begin(), tokens.end());
    return m_cached_vec;
  }

  // justify for _unsafe call: we need to read the key part of the tokens
  token_vec_t& get_key(TreeNode* node) {
    return get_key(node->_unsafe_tokens());
  }

  void add_child(TreeNode* parent, std::unique_ptr<TreeNode>&& child) {
    parent->add_child(get_key(child.get()), std::move(child));
  }

  void add_child(TreeNode* parent, std::unique_ptr<TreeNode>&& child, node_iterator_t it) {
    parent->add_child(it, std::move(child));
  }

  TreeNode m_root;                                        // root node of the tree
  std::size_t m_evictable_size;                           // number of evictable tokens on GPU (lock ref = 0)
  std::size_t m_protected_size;                           // number of protected tokens on GPU (lock ref > 0)
  token_vec_t m_cached_vec;                               // cached vector of tokens for the current operation
  std::unordered_map<std::size_t, TreeNode*> m_node_map;  // map of node keys to nodes
  std::size_t m_node_counter;                             // counter for node IDs

 public:
  // some public constant configurations (without m_ prefix)
  const bool disabled;          // whether the cache is enabled, or just a temporary cache
  const bool use_hicache;       // whether to use the HiCache for this tree
  const std::size_t page_size;  // size of each page in the cache
  const std::size_t threshold;  // threshold for write_through
};

}  // namespace radix_tree_v2
