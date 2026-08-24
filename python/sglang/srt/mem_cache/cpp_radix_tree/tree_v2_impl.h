#pragma once
#include <c10/util/irange.h>

#include <chrono>
#include <cstddef>
#include <iosfwd>
#include <list>
#include <memory>
#include <stdexcept>
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
  using lru_list_t = std::list<TreeNode*>;
  using lru_position_map_t = std::unordered_map<NodeHandle, lru_list_t::iterator>;

  Impl(
      bool disabled,
      bool use_hicache,
      std::size_t page_size,
      std::size_t host_size,
      std::size_t threshold,
      std::size_t sliding_window_size)
      : m_root(/*node_id_=*/0),
        m_evictable_size(0),
        m_protected_size(0),
        m_swa_evictable_size(0),
        m_swa_protected_size(0),
        m_node_map(),
        m_node_counter(1),  // start from 1 to avoid confusion with root node
        m_swa_uuid_counter(1),
        disabled(disabled),
        use_hicache(use_hicache),
        page_size(page_size),
        threshold(threshold),
        sliding_window_size(sliding_window_size) {
    _assert(page_size > 0, "Page size must be greater than zero");
    _assert(use_hicache == (host_size > 0), "Hierarchical cache is enabled iff host size > 0");
    m_root.ref_count = 1;                  // root node is always protected
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
    // Invalidate the old node's lazy heap entries and register both halves.
    touch_full(new_node);
    touch_full(old_node);
    touch_swa(new_node);
    touch_swa(old_node);
    return new_node;
  }

  TreeNode* split_node(TreeNode* old_node, std::size_t prefix_length) {
    auto* parent = old_node->parent();
    const auto tokens = token_slice{old_node->_unsafe_tokens()};
    _assert(tokens.size() >= page_size, "Node key should be at least page-sized");
    auto it = parent->find_child(tokens.first(page_size));
    _assert(it != parent->end() && it->second.get() == old_node, "Child node not found while splitting");
    return split_node(it, prefix_length);
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
    touch_full(new_node);
    return new_node;
  }

  // node: [GPU] -> x
  void remove_device_node(TreeNode* node) {
    _assert(node->on_gpu_only() && node->ref_count == 0);
    _assert(node->swa_ref_count == 0, "Cannot remove a node with a protected SWA value");
    m_evictable_size -= node->length();
    if (node->has_swa()) m_swa_evictable_size -= node->length();
    auto* parent = node->parent();
    const auto node_id = node->node_id;
    erase_lru_entry(node, /*swa=*/false);
    erase_lru_entry(node, /*swa=*/true);
    const auto tokens = token_slice{node->_unsafe_tokens()};
    _assert(tokens.size() >= page_size, "Node key should be at least page-sized");
    auto iterator = parent->find_child(tokens.first(page_size));
    _assert(iterator != parent->end() && iterator->second.get() == node, "Child node not found while removing");
    m_node_map.erase(node_id);  // invalidate lazy LRU entries before destruction
    parent->erase_child(iterator);
  }

  /**
   * @brief Walk the tree to find the node that matches the key.
   * If the key partially matches a node, it will split that node.
   * @return A pair containing the last node that matches the key and
   * the total prefix length matched (on gpu and cpu) so far.
   */
  std::pair<TreeNode*, std::size_t> tree_walk(token_slice key) {
    _assert(key.size() % page_size == 0, "Key should be page-aligned");
    ++m_tree_walk_calls;
    m_tree_walk_tokens += key.size();

    std::size_t total_prefix_length = 0;
    TreeNode* node = &m_root;

    const auto now = std::chrono::steady_clock::now();
    while (key.size() > 0) {
      const auto iterator = node->find_child(key.first(page_size));
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
      touch_full(node, now);
      key = key.subspan(prefix_length);
    }

    return {node, total_prefix_length};
  }

  std::vector<TreeNode*> collect_leaves() const {
    std::vector<TreeNode*> leaves;
    std::vector<TreeNode*> stack = {};

    auto process_node = [&](TreeNode* node) {
      if (node->is_leaf()) {
        if (node->ref_count == 0 && node->swa_ref_count == 0) {
          leaves.push_back(node);
        }
      } else {
        stack.push_back(node);
      }
    };
    for (const auto& [_, child] : m_root) {
      process_node(child.get());
    }
    while (!stack.empty()) {
      const auto node = stack.back();
      stack.pop_back();
      for (const auto& [_, child] : *node) {
        process_node(child.get());
      }
    }
    return leaves;
  }

  std::vector<TreeNode*> collect_leaves_device() const {
    // for non-hicache, every leaf device node is a leaf node (since no backup on host)
    if (!use_hicache) return collect_leaves();
    std::vector<TreeNode*> leaves;
    std::vector<TreeNode*> stack = {};

    auto process_node = [&](TreeNode* node) {
      if (!node->on_gpu()) return;  // skip nodes that are not on GPU

      if (node->is_leaf_device()) {
        if (node->ref_count == 0 && node->swa_ref_count == 0) {
          leaves.push_back(node);
        }
      } else {
        stack.push_back(node);
      }
    };
    for (const auto& [_, child] : m_root) {
      process_node(child.get());
    }
    while (!stack.empty()) {
      const auto node = stack.back();
      stack.pop_back();
      for (const auto& [_, child] : *node) {
        process_node(child.get());
      }
    }

    return leaves;
  }

  std::vector<TreeNode*> collect_all_nodes() const {
    std::vector<TreeNode*> nodes;
    std::vector<TreeNode*> stack;
    for (const auto& [_, child] : m_root)
      stack.push_back(child.get());
    while (!stack.empty()) {
      auto* node = stack.back();
      stack.pop_back();
      nodes.push_back(node);
      for (const auto& [_, child] : *node)
        stack.push_back(child.get());
    }
    return nodes;
  }

  void touch_full(TreeNode* node, TreeNode::timestamp_t now = std::chrono::steady_clock::now()) {
    if (node->is_root()) return;
    node->access(now);
    if (node->on_gpu()) {
      move_lru_to_mru(node, /*swa=*/false);
    }
  }

  void touch_full_path(TreeNode* node) {
    const auto now = std::chrono::steady_clock::now();
    walk_to_root(node, [&](TreeNode* current) { touch_full(current, now); });
  }

  void touch_swa(TreeNode* node, TreeNode::timestamp_t now = std::chrono::steady_clock::now()) {
    if (node->is_root() || !node->has_swa()) return;
    node->access_swa(now);
    move_lru_to_mru(node, /*swa=*/true);
  }

  void touch_swa_window(TreeNode* node) {
    std::size_t covered = 0;
    const auto now = std::chrono::steady_clock::now();
    while (!node->is_root() && covered < sliding_window_size) {
      if (node->has_swa()) {
        touch_swa(node, now);
        covered += node->swa_indices().size(0);
      }
      node = node->parent();
    }
  }

  TreeNode* pop_full_lru_candidate() {
    while (!m_full_lru.empty()) {
      auto* node = m_full_lru.back();
      m_full_lru.pop_back();
      m_full_lru_positions.erase(node->node_id);
      ++m_full_lru_pops;
      if (!node->on_gpu() || node->ref_count != 0 || node->swa_ref_count != 0 || !node->is_leaf_device() ||
          !node->is_io_free())
        continue;
      return node;
    }
    return nullptr;
  }

  TreeNode* pop_swa_lru_candidate() {
    while (!m_swa_lru.empty()) {
      auto* node = m_swa_lru.back();
      m_swa_lru.pop_back();
      m_swa_lru_positions.erase(node->node_id);
      ++m_swa_lru_pops;
      if (!node->has_swa() || node->swa_ref_count != 0) continue;
      return node;
    }
    return nullptr;
  }

  std::size_t next_swa_uuid() {
    return m_swa_uuid_counter++;
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
          touch_full(n);
        }
      });
  }

  void lock_swa(TreeNode* node, bool increment) {
    _assert(node->has_swa(), "Cannot lock a SWA tombstone");
    if (increment) {
      if (node->swa_ref_count == 0) {
        m_swa_evictable_size -= node->length();
        m_swa_protected_size += node->length();
      }
      node->swa_ref_count++;
    } else {
      _assert(node->swa_ref_count != 0, "Cannot decrement SWA reference count = zero");
      node->swa_ref_count--;
      if (node->swa_ref_count == 0) {
        m_swa_protected_size -= node->length();
        m_swa_evictable_size += node->length();
        touch_swa(node);
        // FULL eviction also requires swa_ref_count == 0.  Its prior heap
        // entry may have been consumed while this SWA lock was held.
        touch_full(node);
      }
    }
  }

  void set_swa_value(TreeNode* node, at::Tensor value) {
    _assert(!node->is_root(), "Root cannot own a SWA value");
    _assert(!node->has_swa(), "SWA value already exists on node");
    _assert(static_cast<std::size_t>(value.size(0)) == node->length(), "SWA value length mismatch");
    node->_unsafe_swa_indices() = std::move(value);
    if (node->swa_ref_count == 0)
      m_swa_evictable_size += node->length();
    else
      m_swa_protected_size += node->length();
    touch_swa(node);
  }

  void clear_swa_value(TreeNode* node) {
    if (!node->has_swa()) return;
    erase_lru_entry(node, /*swa=*/true);
    if (node->swa_ref_count == 0)
      m_swa_evictable_size -= node->length();
    else
      m_swa_protected_size -= node->length();
    node->_unsafe_swa_indices() = at::Tensor();
    node->swa_ref_count = 0;
    node->swa_uuid.reset();
    touch_full(node);
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

  std::vector<at::Tensor> all_values() const {
    std::vector<at::Tensor> values;
    std::vector<const TreeNode*> stack = {&m_root};
    while (!stack.empty()) {
      const auto* node = stack.back();
      stack.pop_back();
      if (node->on_gpu()) values.push_back(node->device_indices());
      for (const auto& [_, child] : *node)
        stack.push_back(child.get());
    }
    return values;
  }

  std::size_t evictable_size() const {
    return m_evictable_size;
  }

  std::size_t protected_size() const {
    return m_protected_size;
  }

  std::size_t component_evictable_size(std::size_t component_type) const {
    if (component_type == 0) return m_evictable_size;
    if (component_type == 1) return m_swa_evictable_size;
    return 0;
  }

  std::size_t component_protected_size(std::size_t component_type) const {
    if (component_type == 0) return m_protected_size;
    if (component_type == 1) return m_swa_protected_size;
    return 0;
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
    m_swa_evictable_size = 0;
    m_swa_protected_size = 0;
    m_swa_uuid_counter = 1;
    m_tree_walk_calls = 0;
    m_tree_walk_tokens = 0;
    m_node_match_calls = 0;
    m_full_lru_pops = 0;
    m_swa_lru_pops = 0;
    m_lru_stale_pops = 0;
    m_lru_rebuilds = 0;
    m_full_lru = {};
    m_swa_lru = {};
    m_full_lru_positions.clear();
    m_swa_lru_positions.clear();
    m_node_map.clear();
    m_node_map[m_root.node_id] = &m_root;  // re-add root to the map
  }

  void debug_print(std::ostream& os) const;

 private:
  void erase_lru_entry(TreeNode* node, bool swa) {
    auto& positions = swa ? m_swa_lru_positions : m_full_lru_positions;
    auto& lru = swa ? m_swa_lru : m_full_lru;
    const auto iterator = positions.find(node->node_id);
    if (iterator == positions.end()) return;
    lru.erase(iterator->second);
    positions.erase(iterator);
  }

  void move_lru_to_mru(TreeNode* node, bool swa) {
    auto& positions = swa ? m_swa_lru_positions : m_full_lru_positions;
    auto& lru = swa ? m_swa_lru : m_full_lru;
    const auto iterator = positions.find(node->node_id);
    if (iterator != positions.end()) lru.erase(iterator->second);
    lru.push_front(node);
    positions[node->node_id] = lru.begin();
  }

  void add_child(TreeNode* parent, std::unique_ptr<TreeNode>&& child) {
    const auto tokens = token_slice{child->_unsafe_tokens()};
    _assert(tokens.size() >= page_size, "Node key should be at least page-sized");
    parent->add_child(token_vec_t(tokens.begin(), tokens.begin() + page_size), std::move(child));
  }

  void add_child(TreeNode* parent, std::unique_ptr<TreeNode>&& child, node_iterator_t it) {
    parent->add_child(it, std::move(child));
  }

  TreeNode m_root;               // root node of the tree
  std::size_t m_evictable_size;  // number of evictable tokens on GPU (lock ref = 0)
  std::size_t m_protected_size;  // number of protected tokens on GPU (lock ref > 0)
  std::size_t m_swa_evictable_size;
  std::size_t m_swa_protected_size;
  std::unordered_map<std::size_t, TreeNode*> m_node_map;  // map of node keys to nodes
  std::size_t m_node_counter;                             // counter for node IDs
  std::size_t m_swa_uuid_counter;
  lru_list_t m_full_lru;
  lru_list_t m_swa_lru;
  lru_position_map_t m_full_lru_positions;
  lru_position_map_t m_swa_lru_positions;

 public:
  std::uint64_t m_tree_walk_calls = 0;
  std::uint64_t m_tree_walk_tokens = 0;
  std::uint64_t m_node_match_calls = 0;
  std::uint64_t m_full_lru_pops = 0;
  std::uint64_t m_swa_lru_pops = 0;
  std::uint64_t m_lru_stale_pops = 0;
  std::uint64_t m_lru_rebuilds = 0;

  // some public constant configurations (without m_ prefix)
  const bool disabled;          // whether the cache is enabled, or just a temporary cache
  const bool use_hicache;       // whether to use the HiCache for this tree
  const std::size_t page_size;  // size of each page in the cache
  const std::size_t threshold;  // threshold for write_through
  const std::size_t sliding_window_size;
};

}  // namespace radix_tree_v2
