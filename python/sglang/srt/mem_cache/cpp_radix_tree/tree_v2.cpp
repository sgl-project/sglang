#include "tree_v2.h"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/util/irange.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common.h"
#include "tree_v2_impl.h"
#include "tree_v2_node.h"

namespace radix_tree_v2 {

static NodeHandle node2id(TreeNode* node) {
  return node->node_id;
}

static bool swa_prefix_valid(TreeNode* node, std::size_t sliding_window_size) {
  std::size_t covered = 0;
  while (!node->is_root() && covered < sliding_window_size) {
    if (!node->has_swa()) return false;
    covered += node->swa_indices().size(0);
    node = node->parent();
  }
  return true;
}

static std::vector<TreeNode*> root_path(TreeNode* node) {
  std::vector<TreeNode*> path;
  while (!node->is_root()) {
    path.push_back(node);
    node = node->parent();
  }
  std::reverse(path.begin(), path.end());
  return path;
}

RadixTree::RadixTree(
    bool disabled,
    std::optional<std::size_t> host_size,
    std::size_t page_size,
    std::size_t threshold,
    std::size_t sliding_window_size)
    : m_impl(
          std::make_unique<Impl>(
              disabled, host_size.has_value(), page_size, host_size.value_or(0), threshold, sliding_window_size)) {}

RadixTree::~RadixTree() = default;

std::tuple<std::vector<at::Tensor>, std::size_t, NodeHandle, NodeHandle> RadixTree::match_prefix(token_slice _key) {
  if (m_impl->disabled) return {};

  const auto key = _key.first(m_impl->align(_key.size()));
  const auto [host_node, _] = m_impl->tree_walk(key);

  // walk up to the first non-evicted node
  std::size_t host_hit_length = 0;
  const auto device_node = host_node;

  // collect all the device indices
  std::vector<at::Tensor> indices{};
  walk_to_root(device_node, [&](TreeNode* n) { indices.push_back(n->device_indices()); });
  std::reverse(indices.begin(), indices.end());

  return {std::move(indices), host_hit_length, node2id(device_node), node2id(host_node)};
}

std::tuple<std::vector<at::Tensor>, NodeHandle, std::size_t> RadixTree::match_prefix_swa(token_slice _key) {
  if (m_impl->disabled) return {};
  _assert(m_impl->sliding_window_size > 0, "SWA matching is not enabled");

  const auto key = _key.first(m_impl->align(_key.size()));
  const auto [full_node, full_hit_length] = m_impl->tree_walk(key);
  auto* best_node = full_node;
  while (!best_node->is_root() && !swa_prefix_valid(best_node, m_impl->sliding_window_size))
    best_node = best_node->parent();

  std::vector<at::Tensor> indices;
  walk_to_root(best_node, [&](TreeNode* n) { indices.push_back(n->device_indices()); });
  std::reverse(indices.begin(), indices.end());
  m_impl->touch_swa_window(best_node);
  return {std::move(indices), node2id(best_node), full_hit_length};
}

std::vector<at::Tensor> RadixTree::evict(std::size_t num_tokens) {
  if (m_impl->disabled || num_tokens == 0) return {};
  std::vector<at::Tensor> evicted_values;
  // evict nodes until we reach the desired number of tokens
  std::size_t num_evict = 0;
  while (num_evict < num_tokens) {
    const auto node = m_impl->pop_full_lru_candidate();
    if (node == nullptr) break;
    // when ref_count == 0, can't be writing through
    _assert(node->on_gpu() && node->ref_count == 0);
    if (!node->is_io_free()) continue;  // skip nodes that are undergoing IO (i.e. indices protected)
    evicted_values.push_back(node->device_indices());
    num_evict += node->length();
    const auto parent = node->parent();
    m_impl->remove_device_node(node);
    if (!parent->is_root() && parent->is_leaf_device() && parent->ref_count == 0 && parent->swa_ref_count == 0)
      m_impl->touch_full(parent);
  }

  return evicted_values;
}

std::tuple<std::vector<std::tuple<IOTicket, at::Tensor, at::Tensor>>, std::size_t, NodeHandle>
RadixTree::writing_through(token_slice _key, at::Tensor value) {
  if (m_impl->disabled) return {};
  _assert(_key.size() == std::size_t(value.size(0)), "Key and value must have the same size");

  // just align the key to the page size, clip the unaligned tail
  const auto key = _key.first(m_impl->align(_key.size()));

  // walk the tree to find the right place to insert
  const auto [host_node, host_prefix_length] = m_impl->tree_walk(key);

  // insert and create a new node if the remaining part of the key is not empty
  auto* target = host_node;
  if (host_prefix_length != key.size()) {
    target = m_impl->create_device_node(
        host_node,
        {key.begin() + host_prefix_length, key.end()},
        value.slice(/*dim=*/0, host_prefix_length, key.size()));
  }

  // add the hit count for the device node
  walk_to_root(host_node, [&](TreeNode* n) { n->hit_count++; });

  std::vector<std::tuple<IOTicket, at::Tensor, at::Tensor>> result;

  // don't write through if hicache is disabled (no host memory), fast path
  if (!m_impl->use_hicache) return {std::move(result), host_prefix_length, node2id(target)};
  throw std::runtime_error("Not implemented yet");
}

std::tuple<
    std::size_t,
    NodeHandle,
    std::vector<at::Tensor>,
    std::vector<std::tuple<NodeHandle, at::Tensor>>,
    std::vector<std::tuple<NodeHandle, at::Tensor, at::Tensor>>>
RadixTree::writing_through_swa(
    token_slice _key, at::Tensor value, std::size_t prev_prefix_len, std::size_t swa_evicted_seqlen) {
  if (m_impl->disabled) return {};
  _assert(m_impl->sliding_window_size > 0, "SWA insertion is not enabled");
  _assert(_key.size() == std::size_t(value.size(0)), "Key and value must have the same size");

  const auto key = _key.first(m_impl->align(_key.size()));
  const auto [matched_node, prefix_length] = m_impl->tree_walk(key);
  std::vector<at::Tensor> duplicate_frees;
  std::vector<std::tuple<NodeHandle, at::Tensor>> rebuilds;
  std::vector<std::tuple<NodeHandle, at::Tensor, at::Tensor>> recoveries;

  std::size_t total = 0;
  for (auto* walked : root_path(matched_node)) {
    const std::size_t node_len = walked->length();
    auto incoming = value.slice(0, total, total + node_len);
    std::size_t consumed_from = node_len;

    // The request already owns [0, prev_prefix_len).  Mirroring
    // SWAComponent.update_component_on_insert_overlap, do not consume that
    // protected prefix to recover a tombstone: it may alias the tree's current
    // FULL value, so replacing and freeing it would release live request KV.
    const bool can_consume_for_swa = prev_prefix_len < total + node_len;
    if (!walked->has_swa() && can_consume_for_swa) {
      if (swa_evicted_seqlen <= total) {
        if (walked->ref_count > 0) {
          recoveries.emplace_back(node2id(walked), walked->device_indices(), incoming);
        } else {
          auto old_full = walked->device_indices();
          walked->_unsafe_device_indices() = incoming.clone();
          duplicate_frees.push_back(old_full);
          rebuilds.emplace_back(node2id(walked), walked->device_indices());
        }
        consumed_from = 0;
      } else if (swa_evicted_seqlen < total + node_len) {
        const std::size_t start = swa_evicted_seqlen - total;
        m_impl->split_node(walked, start);
        auto incoming_tail = incoming.slice(0, start, node_len);
        if (walked->ref_count > 0) {
          recoveries.emplace_back(node2id(walked), walked->device_indices(), incoming_tail);
        } else {
          auto old_full = walked->device_indices();
          walked->_unsafe_device_indices() = incoming_tail.clone();
          duplicate_frees.push_back(old_full);
          rebuilds.emplace_back(node2id(walked), walked->device_indices());
        }
        consumed_from = start;
      }
    }

    const std::size_t duplicate_start = prev_prefix_len > total ? std::min(node_len, prev_prefix_len - total) : 0;
    if (duplicate_start < consumed_from) duplicate_frees.push_back(incoming.slice(0, duplicate_start, consumed_from));
    m_impl->touch_full(walked);
    total += node_len;
  }

  TreeNode* target = matched_node;
  bool is_new_leaf = false;
  if (prefix_length != key.size()) {
    target = m_impl->create_device_node(
        matched_node, {key.begin() + prefix_length, key.end()}, value.slice(0, prefix_length, key.size()));
    is_new_leaf = true;
  }

  if (is_new_leaf) {
    const std::size_t node_start = prefix_length;
    const std::size_t leaf_len = target->length();
    if (swa_evicted_seqlen < node_start + leaf_len) {
      const std::size_t split_pos = swa_evicted_seqlen > node_start ? swa_evicted_seqlen - node_start : 0;
      if (split_pos > 0) m_impl->split_node(target, split_pos);

      const std::size_t tail_size =
          ((m_impl->sliding_window_size + m_impl->page_size - 1) / m_impl->page_size) * m_impl->page_size;
      if (target->length() > tail_size) {
        auto* capped_parent = m_impl->split_node(target, target->length() - tail_size);
        rebuilds.emplace_back(node2id(capped_parent), capped_parent->device_indices());
      }
      rebuilds.emplace_back(node2id(target), target->device_indices());
    }
  }

  m_impl->touch_swa_window(target);
  return {prefix_length, node2id(target), std::move(duplicate_frees), std::move(rebuilds), std::move(recoveries)};
}

std::tuple<std::vector<at::Tensor>, NodeHandle, std::size_t, std::optional<std::size_t>, std::vector<NodeHandle>>
RadixTree::match_node_and_lock(NodeHandle node_id, bool lock_full, bool lock_swa) {
  auto [indices, best_node, best_prefix_length, full_hit_length, swa_uuid, skipped] =
      match_node_range_and_lock(node_id, 0, std::numeric_limits<std::size_t>::max(), lock_full, lock_swa);
  return {std::move(indices), best_node, full_hit_length, swa_uuid, std::move(skipped)};
}

std::tuple<
    std::vector<at::Tensor>,
    NodeHandle,
    std::size_t,
    std::size_t,
    std::optional<std::size_t>,
    std::vector<NodeHandle>>
RadixTree::match_node_range_and_lock(
    NodeHandle node_id, std::size_t range_start, std::size_t range_end, bool lock_full, bool lock_swa) {
  if (m_impl->disabled) return {};
  _assert(range_start <= range_end, "Invalid inserted-prefix result range");
  ++m_impl->m_node_match_calls;
  auto* target = m_impl->id2node(node_id);
  m_impl->touch_full_path(target);

  std::size_t full_hit_length = 0;
  walk_to_root(target, [&](TreeNode* node) { full_hit_length += node->length(); });

  auto* best_node = target;
  if (m_impl->sliding_window_size > 0) {
    while (!best_node->is_root() && !swa_prefix_valid(best_node, m_impl->sliding_window_size))
      best_node = best_node->parent();
    m_impl->touch_swa_window(best_node);
  } else {
    _assert(!lock_swa, "Cannot lock SWA when SWA is disabled");
  }

  const auto path = root_path(best_node);
  std::size_t best_prefix_length = 0;
  for (const auto* node : path)
    best_prefix_length += node->length();

  const auto clipped_start = std::min(range_start, best_prefix_length);
  const auto clipped_end = std::min(range_end, best_prefix_length);
  std::vector<at::Tensor> indices;
  if (clipped_start < clipped_end) {
    std::size_t node_start = 0;
    for (auto* node : path) {
      const auto node_end = node_start + node->length();
      if (node_end > clipped_start && node_start < clipped_end) {
        const auto slice_start = clipped_start > node_start ? clipped_start - node_start : 0;
        const auto slice_end = std::min(clipped_end, node_end) - node_start;
        indices.push_back(node->device_indices().slice(0, slice_start, slice_end));
      }
      node_start = node_end;
      if (node_start >= clipped_end) break;
    }
  }

  auto [swa_uuid, skipped] = lock_ref_swa(node2id(best_node), lock_full, lock_swa);
  return {std::move(indices), node2id(best_node), best_prefix_length, full_hit_length, swa_uuid, std::move(skipped)};
}

std::tuple<IOTicket, std::vector<at::Tensor>> RadixTree::loading_onboard(NodeHandle, at::Tensor) {
  if (m_impl->disabled) return {};
  throw std::runtime_error("Not implemented yet");
}

void RadixTree::commit_writing_through(IOTicket, bool) {
  if (m_impl->disabled) return;
  throw std::runtime_error("Not implemented yet");
}

void RadixTree::commit_loading_onboard(IOTicket, bool) {
  if (m_impl->disabled) return;
  throw std::runtime_error("Not implemented yet");
}

void RadixTree::reset() {
  m_impl->reset();
}

void RadixTree::lock_ref(NodeHandle node_id, bool increment) {
  if (m_impl->disabled) return;
  m_impl->lock_ref(node_id, increment);
}

std::tuple<std::optional<std::size_t>, std::vector<NodeHandle>>
RadixTree::lock_ref_swa(NodeHandle node_id, bool lock_full, bool lock_swa) {
  if (m_impl->disabled) return {};
  auto* node = m_impl->id2node(node_id);
  if (lock_full) m_impl->lock_ref(node, true);

  std::optional<std::size_t> boundary_uuid;
  std::vector<NodeHandle> skipped;
  if (lock_swa) {
    std::size_t covered = 0;
    auto* cur = node;
    while (!cur->is_root() && covered < m_impl->sliding_window_size) {
      if (!cur->has_swa()) {
        skipped.push_back(node2id(cur));
        cur = cur->parent();
        continue;
      }
      m_impl->lock_swa(cur, true);
      covered += cur->swa_indices().size(0);
      if (covered >= m_impl->sliding_window_size) {
        if (!cur->swa_uuid.has_value()) cur->swa_uuid = m_impl->next_swa_uuid();
        boundary_uuid = cur->swa_uuid;
      }
      cur = cur->parent();
    }
  }
  return {boundary_uuid, std::move(skipped)};
}

void RadixTree::unlock_ref_swa(
    NodeHandle node_id,
    bool unlock_full,
    bool unlock_swa,
    std::optional<std::size_t> swa_uuid,
    const std::vector<NodeHandle>& skip_swa_nodes) {
  if (m_impl->disabled) return;
  auto* node = m_impl->id2node(node_id);
  if (unlock_full) m_impl->lock_ref(node, false);
  if (!unlock_swa) return;

  std::unordered_set<NodeHandle> skipped(skip_swa_nodes.begin(), skip_swa_nodes.end());
  auto* cur = node;
  bool keep_going = true;
  while (!cur->is_root() && keep_going) {
    if (!skipped.contains(node2id(cur)) && cur->swa_ref_count > 0) m_impl->lock_swa(cur, false);
    if (swa_uuid.has_value() && cur->swa_uuid == swa_uuid) keep_going = false;
    cur = cur->parent();
  }
}

std::tuple<std::vector<at::Tensor>, std::size_t>
RadixTree::unlock_swa_only(NodeHandle node_id, std::optional<std::size_t> swa_uuid) {
  if (m_impl->disabled) return {};
  std::vector<at::Tensor> frees;
  std::size_t freed = 0;
  auto* cur = m_impl->id2node(node_id);
  bool keep_going = true;
  while (!cur->is_root() && keep_going) {
    const bool is_boundary = swa_uuid.has_value() && cur->swa_uuid == swa_uuid;
    if (cur->has_swa() && cur->swa_ref_count > 0) {
      m_impl->lock_swa(cur, false);
      if (cur->swa_ref_count == 0 && cur->ref_count == 0 && cur->is_leaf_device()) {
        frees.push_back(cur->device_indices());
        freed += cur->length();
        m_impl->clear_swa_value(cur);
      }
    }
    if (is_boundary) keep_going = false;
    cur = cur->parent();
  }
  return {std::move(frees), freed};
}

void RadixTree::set_swa_value(NodeHandle node_id, at::Tensor value) {
  if (m_impl->disabled) return;
  m_impl->set_swa_value(m_impl->id2node(node_id), std::move(value));
}

at::Tensor RadixTree::get_swa_value(NodeHandle node_id) const {
  if (m_impl->disabled) return {};
  return m_impl->id2node(node_id)->swa_indices();
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::size_t, std::size_t>
RadixTree::evict_component(std::size_t component_type, std::size_t num_tokens) {
  if (m_impl->disabled || num_tokens == 0) return {};
  _assert(component_type <= 1, "Only FULL and SWA eviction are supported");

  std::vector<at::Tensor> full_frees;
  std::vector<at::Tensor> swa_frees;
  std::size_t full_evicted = 0;
  std::size_t swa_evicted = 0;

  if (component_type == 0) {
    while (full_evicted < num_tokens) {
      auto* node = m_impl->pop_full_lru_candidate();
      if (node == nullptr) break;
      const auto* parent = node->parent();
      full_frees.push_back(node->device_indices());
      full_evicted += node->length();
      if (node->has_swa()) {
        swa_frees.push_back(node->device_indices());
        swa_evicted += node->length();
      }
      m_impl->remove_device_node(node);
      auto* mutable_parent = const_cast<TreeNode*>(parent);
      if (!mutable_parent->is_root() && mutable_parent->is_leaf_device() && mutable_parent->ref_count == 0 &&
          mutable_parent->swa_ref_count == 0)
        m_impl->touch_full(mutable_parent);
    }
  } else {
    while (swa_evicted < num_tokens) {
      TreeNode* victim = m_impl->pop_swa_lru_candidate();
      if (victim == nullptr) break;

      if (victim->is_leaf_device() && victim->ref_count == 0) {
        auto* parent = victim->parent();
        full_frees.push_back(victim->device_indices());
        full_evicted += victim->length();
        swa_frees.push_back(victim->device_indices());
        swa_evicted += victim->length();
        m_impl->remove_device_node(victim);
        if (!parent->is_root() && parent->is_leaf_device() && parent->ref_count == 0 && parent->swa_ref_count == 0)
          m_impl->touch_full(parent);
      } else {
        swa_frees.push_back(victim->device_indices());
        swa_evicted += victim->length();
        m_impl->clear_swa_value(victim);
      }
    }
  }
  return {std::move(full_frees), std::move(swa_frees), full_evicted, swa_evicted};
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

std::size_t RadixTree::component_evictable_size(std::size_t component_type) const {
  return m_impl->component_evictable_size(component_type);
}

std::size_t RadixTree::component_protected_size(std::size_t component_type) const {
  return m_impl->component_protected_size(component_type);
}

std::vector<at::Tensor> RadixTree::all_values() const {
  return m_impl->all_values();
}

std::unordered_map<std::string, std::uint64_t> RadixTree::debug_stats() const {
  return {
      {"tree_walk_calls", m_impl->m_tree_walk_calls},
      {"tree_walk_tokens", m_impl->m_tree_walk_tokens},
      {"node_match_calls", m_impl->m_node_match_calls},
      {"full_lru_pops", m_impl->m_full_lru_pops},
      {"swa_lru_pops", m_impl->m_swa_lru_pops},
      {"lru_stale_pops", m_impl->m_lru_stale_pops},
      {"lru_rebuilds", m_impl->m_lru_rebuilds},
  };
}

}  // namespace radix_tree_v2
