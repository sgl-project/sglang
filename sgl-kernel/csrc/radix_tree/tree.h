#pragma once
#include <ATen/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace radix_tree {

using token_t = std::int32_t;
using token_vec_t = std::vector<token_t>;
// standard timestamp type using steady_clock to ensure monotonicity
using timestamp_t = std::chrono::steady_clock::time_point;

struct VecHash {
  // see https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
  std::size_t operator()(const token_vec_t& vec) const {
    std::size_t hash = 0;
    for (const auto& token : vec) {
      hash ^= token + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

inline std::size_t init_node_id() {
  static std::size_t next_id = 0;
  return next_id++;
}

struct TreeNode {
 public:
  TreeNode() : ref_count(0), m_node_id(init_node_id()) {}

  void access() {
    m_last_access_time = std::chrono::steady_clock::now();
  }

  bool is_root() const {
    return parent == this;
  }

  timestamp_t time() const {
    return m_last_access_time;
  }

 public:
  token_vec_t key;
  at::Tensor value;
  std::size_t ref_count;
  TreeNode* parent;
  std::unordered_map<token_vec_t, std::unique_ptr<TreeNode>, VecHash> children;

 private:
  timestamp_t m_last_access_time;
  const std::size_t m_node_id;  // unique ID for the node
};

struct RadixTree {
 public:
  RadixTree(bool disabled, at::Device device, std::size_t page_size);
  ~RadixTree() = default;

  // Trees should not be copied or moved, as they manage their own memory and state.
  RadixTree(const RadixTree&) = delete;
  RadixTree(RadixTree&&) = delete;
  RadixTree& operator=(const RadixTree&) = delete;
  RadixTree& operator=(RadixTree&&) = delete;

  std::size_t insert(const token_vec_t& key, at::Tensor value);
  std::pair<at::Tensor, std::uintptr_t> match_prefix(const token_vec_t& key);
  std::vector<at::Tensor> evict(std::size_t num_tokens);
  void lock_ref(std::uintptr_t node_id, bool increment /* increment or decrement */);

  std::size_t evictable_size() const {
    return m_evictable_size;
  }
  std::size_t protected_size() const {
    return m_protected_size;
  }
  std::size_t total_size() const;

 private:
  TreeNode m_root;                // root node of the tree
  std::size_t m_evictable_size;   // number of evictable tokens
  std::size_t m_protected_size;   // number of protected tokens
  const bool m_disabled;          // whether the cache is enabled, or just a temporary cache
  const at::Device m_device;      // device type of the tree
  const std::size_t m_page_size;  // size of each page in the cache
};

}  // namespace radix_tree
