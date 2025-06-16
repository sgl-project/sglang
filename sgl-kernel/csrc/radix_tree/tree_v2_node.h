#pragma once
#include <ATen/core/TensorBody.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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

struct token_slice {
 public:
  token_slice(const token_vec_t& tokens) : m_data(tokens.data()), m_size(tokens.size()) {}
  token_slice(const token_t* data, std::size_t size) : m_data(data), m_size(size) {}

  std::array<token_slice, 2> split(std::size_t offset) const {
    return {
        token_slice{m_data, offset},
        token_slice{m_data + offset, m_size - offset},
    };
  }

  std::size_t size() const {
    return m_size;
  }

  const token_t* begin() const {
    return m_data;
  }
  const token_t* end() const {
    return m_data + m_size;
  }

 private:
  const token_t* m_data;
  std::size_t m_size;
};

/**
 * Every node is a host node, which means either it is on the host or on the device (or both).
 * A device node stands for a node that is on the device, and it may have a backup on the host.
 * A host node stands for a node that is on the host, and it may be on device as well.
 */
struct TreeNode {
 public:
  using map_t = std::unordered_map<token_vec_t, std::unique_ptr<TreeNode>, std_vector_hash>;
  using iterator_t = typename map_t::iterator;
  using timestamp_t = std::chrono::steady_clock::time_point;

  // static member counter
  inline static std::size_t _counter = 0;

  TreeNode()
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
        node_id(_counter++) {}

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

  bool is_leaf_host() const {
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

  TreeNode* parent() {
    return m_parent;
  }

  // set up all data structures except for parent-child relationship
  static void split_prefix(TreeNode* new_node, TreeNode* old_node, std::size_t prefix) {
    auto tokens = std::move(old_node->m_tokens);
    _assert(0 < prefix && prefix < tokens.size(), "Invalid prefix size for split");

    old_node->m_tokens = token_vec_t(tokens.begin() + prefix, tokens.end());
    new_node->m_tokens = std::move(tokens);
    new_node->m_tokens.resize(prefix);

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

    if (old_node->m_io_locked.has_value()) new_node->m_io_locked = false;  // keep the IO status
    new_node->m_io_status = old_node->m_io_status;
    new_node->m_io_ticket = old_node->m_io_ticket;
  }

  std::size_t diff_key(token_slice key, std::size_t offset) const {
    auto a = m_tokens;
    auto b = key;
    auto it = std::mismatch(a.begin() + offset, a.end(), b.begin() + offset, b.end());
    return it.first - a.begin();  // >= offset
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

  void start_host_to_device(IOTicket ticket, bool locked) {
    _assert(this->on_both(), "Node must be on both host and device before copying");
    _assert(m_io_status == IOStatus::None, "IO operation already in progress");
    m_io_locked = locked;  // whether ref_count will be locked during IO operation
    m_io_status = IOStatus::HostToDevice;
    m_io_ticket = ticket;
  }

  void start_device_to_host(IOTicket ticket, bool locked) {
    _assert(this->on_both(), "Node must be on both host and device before copying");
    _assert(m_io_status == IOStatus::None, "IO operation already in progress");
    m_io_locked = locked;  // whether ref_count will be locked during IO operation
    m_io_status = IOStatus::DeviceToHost;
    m_io_ticket = ticket;
  }

  void complete_host_to_device(IOTicket ticket) {
    _assert(this->on_both(), "Node must be on both device and host when host -> device");
    _assert(m_io_status == IOStatus::HostToDevice, "Wrong IO operation status (should be host -> device)");
    _assert(ticket == m_io_ticket, "IO ticket mismatch during host -> device completion");
    m_io_status = IOStatus::None;
    m_io_locked.reset();  // reset IO locked status after completion
  }

  void complete_device_to_host(IOTicket ticket) {
    _assert(this->on_both(), "Node must be on both device and host when device -> host");
    _assert(m_io_status == IOStatus::DeviceToHost, "Wrong IO operation status (should be device -> host)");
    _assert(ticket == m_io_ticket, "IO ticket mismatch during device -> host completion");
    m_io_status = IOStatus::None;
    m_io_locked.reset();  // reset IO locked status after completion
  }

  bool is_io_locked() const {
    _assert(m_io_locked.has_value(), "IO locked status is not set");
    return m_io_locked.value();
  }

  std::optional<IOTicket> io_ticket() const {
    if (m_io_status == IOStatus::None) {
      _assert(!m_io_locked.has_value(), "IO lock in wrong status");
      return std::nullopt;  // no IO operation in progress
    }
    _assert(m_io_locked.has_value(), "IO lock in wrong status");
    return m_io_ticket;
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
    _assert(is_root(), "Only root node can be reset");
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
  at::Tensor m_device_indices;
  at::Tensor m_host_indices;  // indices of host value, if applicable
  TreeNode* m_parent;
  std::unordered_map<token_vec_t, std::unique_ptr<TreeNode>, std_vector_hash> m_children;
  timestamp_t m_last_access_time;

 public:
  const std::size_t node_id;  // unique ID for the node
};

}  // namespace radix_tree_v2
