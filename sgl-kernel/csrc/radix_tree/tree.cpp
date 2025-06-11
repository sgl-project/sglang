#include "tree.h"

#include <ATen/TensorIndexing.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/zeros.h>
#include <c10/core/ScalarType.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

namespace radix_tree {

namespace {

// minimal implementation of std::span<const int>/std::string_view
struct token_view {
 public:
  token_view(const token_vec_t& tokens) : m_data(tokens.data()), m_size(tokens.size()) {}
  token_view(const token_t* data, std::size_t size) : m_data(data), m_size(size) {}

  std::array<token_view, 2> split(std::size_t offset) const {
    return {
        token_view{m_data, offset},
        token_view{m_data + offset, m_size - offset},
    };
  }

  std::size_t size() const {
    return m_size;
  }

  void copy_to(std::vector<token_t>& out) const {
    out.resize(m_size);
    out.assign(m_data, m_data + m_size);
  }

  token_vec_t to_vector() const {
    token_vec_t out;
    copy_to(out);
    return out;
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

std::size_t _match_aux(std::size_t page_size, const token_view& src_0, const token_view& src_1, std::size_t offset) {
  // match from the offset, the first page_size tokens must be the same
  const auto [it_0, it_1] = std::mismatch(src_0.begin() + offset, src_0.end(), src_1.begin() + offset, src_1.end());
  return static_cast<std::size_t>(std::distance(src_0.begin(), it_0));
}

token_view _key_aux(std::size_t page_size, const token_view& src_0) {
  return src_0.split(std::min(page_size, src_0.size()))[0];
}

void _insert_aux(std::size_t page_size, TreeNode* node, std::unique_ptr<TreeNode>&& child) {
  node->children.try_emplace(_key_aux(page_size, child->key).to_vector(), std::move(child));
}

void _remove_aux(std::size_t page_size, TreeNode* node, token_view key) {
  node->children.erase(_key_aux(page_size, key).to_vector());
}

using node_iterator_t = typename decltype(TreeNode::children)::iterator;

struct MatchInfo {
  node_iterator_t it;               // iterator to the child node
  TreeNode* current_node;           // the current node we are at
  std::size_t total_prefix_length;  // before this node
  std::size_t match_prefix_length;  // of this node
};

MatchInfo _tree_walk(std::size_t page_size, TreeNode* current_node, token_view current_tokens) {
  std::size_t total_prefix_length = 0;
  token_vec_t temporary_tokens;
  temporary_tokens.reserve(page_size);
  current_node->access();
  while (current_tokens.size() > 0) {
    const auto prefix = _key_aux(page_size, current_tokens);
    prefix.copy_to(temporary_tokens);
    const auto it = current_node->children.find(temporary_tokens);
    if (it == current_node->children.end()) break;  // No child node found, we can insert here.
    current_node = it->second.get();
    current_node->access();
    // at least prefix.size() are matched, and the key might be compressed
    // due to the nature of the radix tree
    const auto common_length = _match_aux(page_size, current_tokens, current_node->key, prefix.size());
    // maybe we should tell compiler common_length > 0?
    if (common_length < current_node->key.size())  // partial match case
      return {it, current_node, total_prefix_length, common_length};
    total_prefix_length += common_length;
    // we have matched the whole key, continue to the next level
    current_tokens = current_tokens.split(common_length)[1];
  }
  return {{}, current_node, total_prefix_length, 0};
}

TreeNode* _split_aux(std::size_t page_size, node_iterator_t it, std::size_t common_length) {
  auto* const parent = it->second->parent;
  auto old_node = std::move(it->second);
  auto new_node = std::make_unique<TreeNode>();
  // set up keys
  new_node->key = std::move(old_node->key);
  old_node->key.assign(new_node->key.begin() + common_length, new_node->key.end());
  new_node->key.resize(common_length);
  // set up values
  const auto remain_length = old_node->key.size();
  auto values = old_node->value.split_with_sizes({int64_t(common_length), int64_t(remain_length)});
  new_node->value = std::move(values[0]);
  old_node->value = std::move(values[1]);
  // set up ref counts
  new_node->ref_count = old_node->ref_count;
  // set up parents
  new_node->parent = parent;
  old_node->parent = new_node.get();
  // set up childrens
  _insert_aux(page_size, new_node.get(), std::move(old_node));
  it->second = std::move(new_node);
  return it->second.get();
}

std::vector<TreeNode*> _collect_leaves(TreeNode* root) {
  std::vector<TreeNode*> leaves;
  std::vector<TreeNode*> stack = {root};
  while (!stack.empty()) {
    auto* current_node = stack.back();
    stack.pop_back();
    if (current_node->children.empty()) {
      if (current_node->ref_count == 0)  // skip those that are still in use
        leaves.push_back(current_node);
    } else {
      for (const auto& [_, child] : current_node->children) {
        stack.push_back(child.get());
      }
    }
  }
  return leaves;
}

std::uintptr_t pointer_cast(const TreeNode* node) {
  return reinterpret_cast<std::uintptr_t>(node);
}

TreeNode* pointer_cast(std::uintptr_t ptr) {
  return reinterpret_cast<TreeNode*>(ptr);
}

}  // namespace

RadixTree::RadixTree(bool disabled, at::Device device, std::size_t page_size)
    : m_root(), m_evictable_size(), m_protected_size(), m_disabled(disabled), m_device(device), m_page_size(page_size) {
  m_root.access();
  m_root.parent = &m_root;
  m_root.ref_count = 1;  // root's ref_count is always greater than 0
}

std::size_t RadixTree::insert(const token_vec_t& key, at::Tensor value) {
  if (m_disabled) return 0;

  const auto matched = _tree_walk(m_page_size, &m_root, token_view{key});
  auto [it, current_node, total_prefix_length, match_prefix_length] = matched;
  total_prefix_length += match_prefix_length;

  if (total_prefix_length == key.size())  // fully matched, do nothing
    return total_prefix_length;

  if (match_prefix_length > 0)  // split the last node
    current_node = _split_aux(m_page_size, it, match_prefix_length);

  auto new_node = std::make_unique<TreeNode>();
  new_node->access();
  new_node->key.assign(key.begin() + total_prefix_length, key.end());
  new_node->value = value.slice(/*dim=*/0, total_prefix_length, key.size());
  new_node->ref_count = 0;
  new_node->parent = current_node;
  _insert_aux(m_page_size, current_node, std::move(new_node));
  m_evictable_size += key.size() - total_prefix_length;
  return total_prefix_length;
}

std::pair<at::Tensor, std::uintptr_t> RadixTree::match_prefix(const token_vec_t& key) {
  if (m_disabled) return {at::zeros({0}, at::TensorOptions().dtype(at::kLong).device(m_device)), pointer_cast(&m_root)};

  const auto page_aligned_length = key.size() / m_page_size * m_page_size;
  const auto matched = _tree_walk(m_page_size, &m_root, token_view{key.data(), page_aligned_length});
  auto [it, current_node, _, match_prefix_length] = matched;

  if (match_prefix_length > 0) current_node = _split_aux(m_page_size, it, match_prefix_length);

  std::vector<at::Tensor> values;

  // count the number of elements in the path
  std::size_t num_elems = 1;
  for (auto* node = current_node; !node->is_root(); node = node->parent)
    ++num_elems;
  values.resize(num_elems);

  // collect all values in the path
  for (auto* node = current_node; !node->is_root(); node = node->parent)
    values[--num_elems] = node->value;

  if (values.size() == 0)  // no values in the path, return empty tensor
    return {at::zeros({0}, at::TensorOptions().dtype(at::kLong).device(m_device)), pointer_cast(current_node)};
  if (values.size() == 1)  // skip costly cat if we have only one value
    return {std::move(values[0]), pointer_cast(current_node)};
  return {at::cat(std::move(values)).to(m_device), pointer_cast(current_node)};
}

std::vector<at::Tensor> RadixTree::evict(std::size_t num_tokens) {
  if (m_disabled || num_tokens == 0) return {};

  // sort by access time, the least recently used will be at the front
  static constexpr auto cmp = [](TreeNode* lhs, TreeNode* rhs) { return lhs->time() < rhs->time(); };
  std::priority_queue<TreeNode*, std::vector<TreeNode*>, decltype(cmp)> heap{cmp, _collect_leaves(&m_root)};
  std::vector<at::Tensor> evicted_values;

  // evict nodes until we reach the desired number of tokens
  std::size_t num_evict = 0;
  while (num_evict < num_tokens && !heap.empty()) {
    const auto node = heap.top();
    heap.pop();
    if (node->ref_count > 0) throw std::runtime_error("This should never happen");
    evicted_values.push_back(std::move(node->value));
    // erase the leaf from the tree
    num_evict += node->key.size();
    m_evictable_size -= node->key.size();
    const auto parent = node->parent;
    _remove_aux(m_page_size, parent, node->key);
    if (parent->children.empty() && parent->ref_count == 0)
      heap.push(parent);  // push parent to the heap if it is now a free leaf
  }

  return evicted_values;
}

void RadixTree::lock_ref(std::uintptr_t addr, bool increment /* increment or decrement */) {
  if (m_disabled) return;

  auto* node = pointer_cast(addr);
  if (increment) {
    while (!node->is_root()) {
      node->ref_count++;
      if (node->ref_count == 1) {
        m_evictable_size -= node->key.size();
        m_protected_size += node->key.size();
      }
      node = node->parent;
    }
  } else {
    while (!node->is_root()) {
      node->ref_count--;
      if (node->ref_count == 0) {
        m_protected_size -= node->key.size();
        m_evictable_size += node->key.size();
      }
      node = node->parent;
    }
  }
}

std::size_t RadixTree::total_size() const {
  std::size_t size = 0;
  std::vector<const TreeNode*> stack = {&m_root};
  while (!stack.empty()) {
    auto* current_node = stack.back();
    stack.pop_back();
    size += current_node->key.size();
    for (const auto& [_, child] : current_node->children)
      stack.push_back(child.get());
  }
  return size;
}

}  // namespace radix_tree
