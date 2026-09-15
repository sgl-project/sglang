#pragma once
#include <ATen/core/TensorBody.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ranges>
#include <unordered_map>

#include "common.h"

namespace radix_tree_v2 {

struct std_vector_hash {
  // see https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
  std::size_t operator()(const token_vec_t& vec) const {
    std::size_t hash = 0;
    for (const auto& token : vec) {
      hash ^= token + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

struct TreeNode {
 public:
  using childern_map_t = std::unordered_map<token_vec_t, std::unique_ptr<TreeNode>, std_vector_hash>;
  using iterator_t = typename childern_map_t::iterator;
  using const_iterator_t = typename childern_map_t::const_iterator;
  using timestamp_t = std::chrono::steady_clock::time_point;

  TreeNode(std::size_t node_id_)
      : ref_count(0),
        hit_count(0),
        m_io_locked(std::nullopt),
        m_io_status(IOStatus::None),
        m_io_ticket(),
        m_tokens(),
        m_device_indices(),
        m_host_indices(),
        m_parent(),
        m_children(),
        m_last_access_time(std::chrono::steady_clock::now()),
        node_id(node_id_) {}

  void access(timestamp_t time = std::chrono::steady_clock::now()) {
    m_last_access_time = time;
  }

  bool is_root() const {
    return m_parent == nullptr;
  }

  timestamp_t time() const {
    return m_last_access_time;
  }

  bool on_gpu() const {
    return m_device_indices.defined();
  }

  bool on_cpu() const {
    return m_host_indices.defined();
  }

  bool on_gpu_only() const {
    return on_gpu() && !on_cpu();
  }

  bool on_cpu_only() const {
    return !on_gpu() && on_cpu();
  }

  bool on_both() const {
    return on_gpu() && on_cpu();
  }

  std::size_t length() const {
    return m_tokens.size();
  }

  bool is_leaf() const {
    return m_children.empty();
  }

  bool is_leaf_device() const {
    for (const auto& [_, child] : m_children)
      if (child->on_gpu()) return false;  // at least one child is on the device
    return true;
  }

  void add_child(const token_vec_t& v, std::unique_ptr<TreeNode>&& child) {
    child->m_parent = this;
    m_children[v] = std::move(child);
  }

  void add_child(iterator_t it, std::unique_ptr<TreeNode>&& child) {
    child->m_parent = this;
    it->second = std::move(child);
  }

  void erase_child(const token_vec_t& v) {
    _assert(m_children.erase(v) > 0, "Child node not found");
  }

  iterator_t find_child(const token_vec_t& v) {
    return m_children.find(v);
  }

  iterator_t begin() {
    return m_children.begin();
  }

  iterator_t end() {
    return m_children.end();
  }

  const_iterator_t begin() const {
    return m_children.begin();
  }

  const_iterator_t end() const {
    return m_children.end();
  }

  TreeNode* parent() {
    return m_parent;
  }

  // set up all data structures except for parent-child relationship
  friend void split_prefix(TreeNode* new_node, TreeNode* old_node, std::size_t prefix_length) {
    auto tokens = std::move(old_node->m_tokens);
    _assert(0 < prefix_length && prefix_length < tokens.size(), "Invalid prefix size for split");

    // set up tokens
    old_node->m_tokens = token_vec_t(tokens.begin() + prefix_length, tokens.end());
    new_node->m_tokens = std::move(tokens);
    new_node->m_tokens.resize(prefix_length);

    // set up values
    const int64_t new_size = new_node->length();
    const int64_t old_size = old_node->length();
    if (old_node->m_device_indices.defined()) {
      auto new_indices = old_node->m_device_indices.split_with_sizes({new_size, old_size});
      new_node->m_device_indices = std::move(new_indices[0]);
      old_node->m_device_indices = std::move(new_indices[1]);
    }
    if (old_node->m_host_indices.defined()) {
      auto new_indices = old_node->m_host_indices.split_with_sizes({new_size, old_size});
      new_node->m_host_indices = std::move(new_indices[0]);
      old_node->m_host_indices = std::move(new_indices[1]);
    }

    // set up ref counts and hit counts
    new_node->ref_count = old_node->ref_count;
    new_node->hit_count = old_node->hit_count;

    // If the old node (child) was locked for IO, the new node (parent) does not need
    // to be locked, since it is naturally protected by the child node's lock.
    if (old_node->m_io_locked.has_value()) {
      new_node->m_io_locked = false;
      new_node->m_io_status = old_node->m_io_status;
      new_node->m_io_ticket = old_node->m_io_ticket;
    }
  }

  /// @return The first index in `m_tokens` that differs from `key`.
  std::size_t diff_key(token_slice key, std::size_t offset) const {
    const auto a = token_slice{key}.subspan(offset);
    const auto b = token_slice{m_tokens}.subspan(offset);
    const auto [it_a, it_b] = std::ranges::mismatch(a, b);
    return it_a - a.begin();  // return the index of the first differing token
  }

  at::Tensor device_indices() const {
    return m_device_indices;
  }
  at::Tensor host_indices() const {
    return m_host_indices;
  }

  // visiting tokens are always unsafe (use `diff_key` instead)
  token_vec_t& _unsafe_tokens() {
    return m_tokens;
  }
  at::Tensor& _unsafe_device_indices() {
    return m_device_indices;
  }
  at::Tensor& _unsafe_host_indices() {
    return m_host_indices;
  }

  bool is_io_free() const {
    return m_io_status == IOStatus::None;
  }

  bool is_io_device_to_host() const {
    return m_io_status == IOStatus::DeviceToHost;
  }

  bool is_io_host_to_device() const {
    return m_io_status == IOStatus::HostToDevice;
  }

  void root_reset() {
    _assert(is_root(), "Only root node can call root_reset");
    _assert(
        m_io_status == IOStatus::None && m_io_locked == std::nullopt,
        "IO operation in progress, cannot reset root node");
    _assert(this->m_tokens.empty(), "Root node tokens should be empty on reset");
    _assert(
        !this->m_device_indices.defined() && !this->m_host_indices.defined(),
        "Root node indices should be always be empty and never assigned");
    m_children.clear();
    this->access();
  }

 public:
  std::size_t ref_count;
  std::size_t hit_count;

 private:
  enum class IOStatus : std::uint8_t {
    None,
    HostToDevice,
    DeviceToHost,
  };

  std::optional<bool> m_io_locked;  // whether the node is locked in IO operation
  IOStatus m_io_status;
  IOTicket m_io_ticket;

  token_vec_t m_tokens;
  at::Tensor m_device_indices;  // indices of device value
  at::Tensor m_host_indices;    // indices of host value
  TreeNode* m_parent;
  childern_map_t m_children;
  timestamp_t m_last_access_time;

 public:
  const std::size_t node_id;  // unique ID for the node
};

template <typename F>
inline TreeNode* walk_to_root(TreeNode* t, const F& f) {
  while (!t->is_root()) {
    f(t);
    t = t->parent();
  }
  return t;  // return the root node
}

}  // namespace radix_tree_v2
