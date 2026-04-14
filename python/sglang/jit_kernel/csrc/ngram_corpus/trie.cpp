#include "trie.h"

#include <algorithm>
#include <cstring>
#include <list>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace ngram {

TrieArena::TrieArena(size_t capacity) {
  nodes_.resize(capacity);
  node_pool_.reserve(capacity);
  for (auto& node : nodes_) {
    node_pool_.emplace_back(&node);
  }
  free_node_count_ = node_pool_.size();
}

Trie::Trie(TrieArena* arena, const Param& param) : arena_(arena), param_(param) {
  root_ = getNode();
}

Trie::~Trie() {
  clear_(true);
}

void Trie::releaseNode_(TrieNode* node) {
  retireNode(node);
  arena_->node_pool_[arena_->free_node_count_++] = node;
}

void Trie::clear_(bool release_root) {
  if (root_ == nullptr) {
    return;
  }

  std::vector<TrieNode*> stack;
  stack.reserve(root_->child.size());
  for (const auto& [_, child] : root_->child) {
    stack.push_back(child);
  }

  std::vector<TrieNode*> postorder;
  postorder.reserve(stack.size());
  while (!stack.empty()) {
    TrieNode* current = stack.back();
    stack.pop_back();
    postorder.push_back(current);
    for (const auto& [_, child] : current->child) {
      stack.push_back(child);
    }
  }

  for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
    TrieNode* current = *it;
    arena_->global_lru_.erase(current->global_lru_pos);
    current->parent->lru.erase(current->parent_lru_pos);
    current->parent->sorted_children.erase(current);
    current->parent->child.erase(current->token);
    releaseNode_(current);
  }

  path_.clear();
  if (release_root) {
    releaseNode_(root_);
    root_ = nullptr;
    return;
  }

  auto version = root_->version;
  root_->~TrieNode();
  new (root_) TrieNode();
  root_->version = version;
}

void Trie::insert(const int32_t* tokens, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    auto start = tokens + i;
    auto end = start + std::min(len - i, param_.max_trie_depth);

    if (static_cast<size_t>(end - start) > arena_->free_node_count_) {
      squeeze(end - start - arena_->free_node_count_);
    }

    TrieNode* cursor = root_;
    path_.clear();
    while (start != end) {
      auto token = *start;
      auto iter = cursor->child.find(token);
      if (iter == cursor->child.end()) {
        iter = cursor->child.insert({token, getNode()}).first;
        auto node = iter->second;

        cursor->lru.emplace_front(node);
        arena_->global_lru_.emplace_back(node);

        node->token = token;
        node->parent = cursor;
        node->parent_lru_pos = cursor->lru.begin();
        node->global_lru_pos = --arena_->global_lru_.end();
        node->freq = 1;
        cursor->sorted_children.insert(node);
      } else {
        auto node = iter->second;
        cursor->sorted_children.erase(node);
        node->freq++;
        cursor->sorted_children.insert(node);
        cursor->lru.splice(cursor->lru.begin(), cursor->lru, node->parent_lru_pos);
      }
      cursor = iter->second;
      path_.emplace_back(cursor);
      ++start;
    }

    for (auto it = path_.rbegin(); it != path_.rend(); ++it) {
      TrieNode* node = *it;
      arena_->global_lru_.splice(arena_->global_lru_.begin(), arena_->global_lru_, node->global_lru_pos);
    }
  }
}

void Trie::squeeze(size_t count) {
  if (arena_->global_lru_.size() < count) {
    throw std::runtime_error(
        "Insufficient node size to release required nodes. "
        "available to release: " +
        std::to_string(arena_->global_lru_.size()) + ", required to release: " + std::to_string(count));
  }
  while (count--) {
    auto last = arena_->global_lru_.back();
    arena_->global_lru_.pop_back();

    if (!last->child.empty()) {
      throw std::runtime_error(
          "The node to be released still has child nodes and cannot be "
          "released. ");
    }

    last->parent->lru.erase(last->parent_lru_pos);
    last->parent->sorted_children.erase(last);
    last->parent->child.erase(last->token);
    releaseNode_(last);
  }
}

void Trie::reset() {
  ++trie_epoch_;
  clear_(false);
}

const TrieNode* Trie::resolve(const MatchState& state, const NodeRef& ref) const {
  if (ref.ptr == nullptr || state.trie_epoch != trie_epoch_ || ref.ptr->version != ref.version) {
    return nullptr;
  }
  return ref.ptr;
}

bool Trie::validateMatchState_(const MatchState& state) const {
  if (state.trie_epoch != trie_epoch_) {
    return false;
  }
  for (const auto& ref : state.anchors) {
    if (ref.ptr && !resolve(state, ref)) {
      return false;
    }
  }
  return true;
}

void Trie::rebuildMatchState_(const int32_t* context, size_t len, MatchState& state, size_t total_len) const {
  const auto max_match_depth = std::min(len, param_.max_trie_depth);
  state.trie_epoch = trie_epoch_;
  state.processed_total_len = total_len;
  state.anchors.assign(max_match_depth, {});
  for (size_t match_depth = 1; match_depth <= max_match_depth; ++match_depth) {
    auto start = context + len - match_depth;
    auto end = start + match_depth;
    auto cursor = root_;
    while (start != end) {
      auto iter = cursor->child.find(*start);
      if (iter == cursor->child.end()) {
        cursor = nullptr;
        break;
      }
      ++start;
      cursor = iter->second;
    }
    if (cursor != nullptr) {
      state.anchors[match_depth - 1] = capture(cursor);
    }
  }
}

bool Trie::advanceMatchState_(MatchState& state, const int32_t* tokens, size_t len, size_t total_len) const {
  if (!validateMatchState_(state)) {
    return false;
  }

  // Reuse a single buffer across iterations to avoid per-token heap allocation.
  std::vector<NodeRef> next;
  next.reserve(param_.max_trie_depth);

  for (size_t i = 0; i < len; ++i) {
    const auto next_depth = std::min(state.anchors.size() + 1, param_.max_trie_depth);
    next.assign(next_depth, {});

    // Root is never evicted, so we access it directly; the epoch was already
    // validated above.
    if (auto iter = root_->child.find(tokens[i]); iter != root_->child.end()) {
      next[0] = capture(iter->second);
    }

    for (size_t depth = 1; depth < next_depth; ++depth) {
      const auto& prev_ref = state.anchors[depth - 1];
      if (prev_ref.ptr == nullptr) {
        continue;
      }
      const auto prev_node = resolve(state, prev_ref);
      if (prev_node == nullptr) {
        return false;
      }
      if (auto iter = prev_node->child.find(tokens[i]); iter != prev_node->child.end()) {
        next[depth] = capture(iter->second);
      }
    }

    state.anchors.swap(next);
  }

  state.processed_total_len = total_len;
  return true;
}

std::vector<std::pair<const TrieNode*, int32_t>> Trie::getExpandableAnchors_(const MatchState& state) const {
  std::vector<std::pair<const TrieNode*, int32_t>> result;
  result.reserve(state.anchors.size());
  for (size_t depth = state.anchors.size(); depth > 0; --depth) {
    const auto node = resolve(state, state.anchors[depth - 1]);
    if (node != nullptr && !node->child.empty()) {
      result.emplace_back(node, static_cast<int32_t>(depth));
    }
  }
  return result;
}

std::vector<std::pair<const TrieNode*, int32_t>>
Trie::match(const int32_t* context, size_t len, MatchState& state, size_t total_len) const {
  const bool has_forward_progress = total_len >= state.processed_total_len;
  const auto appended_len = has_forward_progress ? total_len - state.processed_total_len : 0;
  const auto expected_prev_depth = std::min(state.processed_total_len, param_.max_trie_depth);
  const bool can_advance = state.trie_epoch == trie_epoch_ && has_forward_progress && appended_len <= len &&
                           state.anchors.size() == expected_prev_depth;

  if (can_advance && advanceMatchState_(state, context + len - appended_len, appended_len, total_len)) {
    return getExpandableAnchors_(state);
  }

  rebuildMatchState_(context, len, state, total_len);
  return getExpandableAnchors_(state);
}

MatchQuality
Trie::summarizeMatchQuality(const std::vector<std::pair<const TrieNode*, int32_t>>& anchors, const Param& param) const {
  MatchQuality quality;
  if (anchors.empty()) {
    return quality;
  }

  const auto& [best_node, best_depth] = anchors.front();
  quality.has_match = true;
  quality.specificity = std::min(1.0, static_cast<double>(best_depth) / std::max<size_t>(1, param.max_trie_depth));

  const int top_k = std::max(1, static_cast<int>(param.max_bfs_breadth));
  double top_mass = 0.0;
  double total_mass = 0.0;
  int count = 0;
  for (auto* child : best_node->sorted_children) {
    const auto mass = static_cast<double>(child->freq);
    if (count == 0) {
      top_mass = mass;
    }
    total_mass += mass;
    if (++count >= top_k) {
      break;
    }
  }
  if (total_mass > 0.0) {
    quality.confidence = top_mass / total_mass;
  }
  return quality;
}

Result Trie::buildRecencyFromAnchors(
    const std::vector<std::pair<const TrieNode*, int32_t>>& anchors,
    int32_t last_token,
    size_t draft_token_num,
    const Param& param) const {
  const auto max_match_depth = std::max<int32_t>(1, static_cast<int32_t>(param.max_trie_depth - 1));
  double bfs_breadth_scale = double(param.max_bfs_breadth - param.min_bfs_breadth) / max_match_depth;

  std::vector<Node> tree(draft_token_num + 1);
  int root = 0;
  int cursor = 1;

  for (auto [node, depth] : anchors) {
    std::queue<std::tuple<int32_t, double, const TrieNode*>> queue;
    queue.push({root, (max_match_depth - depth) * bfs_breadth_scale + param.min_bfs_breadth, node});
    while (queue.size() && cursor <= static_cast<int>(draft_token_num)) {
      auto front = queue.front();
      queue.pop();

      auto parent = std::get<0>(front);
      auto cur_breadth = std::get<1>(front);
      auto iter = std::get<2>(front)->lru.begin();

      auto breadth = std::max(1, int32_t(cur_breadth));
      for (int i = 0;
           i < breadth && iter != std::get<2>(front)->lru.end() && cursor <= static_cast<int>(draft_token_num);
           ++i, ++iter) {
        auto token = (*iter)->token;
        auto pos = -1;
        if (auto tit = tree[parent].next.find(token); tit != tree[parent].next.end()) {
          pos = tit->second;
        } else {
          pos = tree[parent].next.insert(std::make_pair(token, cursor++)).first->second;
        }
        queue.emplace(pos, cur_breadth - bfs_breadth_scale, *iter);
      }
    }
  }

  return fillResult(last_token, draft_token_num + 1, tree, root);
}

Result Trie::buildFrequencyFromAnchors(
    const std::vector<std::pair<const TrieNode*, int32_t>>& anchors,
    int32_t last_token,
    size_t draft_token_num,
    const Param& param) const {
  struct CompareByLastDouble {
    bool operator()(
        const std::tuple<double, const TrieNode*, double>& a,
        const std::tuple<double, const TrieNode*, double>& b) const {
      return std::get<2>(a) < std::get<2>(b);
    }
  };

  std::priority_queue<
      std::tuple<double, const TrieNode*, double>,
      std::vector<std::tuple<double, const TrieNode*, double>>,
      CompareByLastDouble>
      heap;

  std::vector<Node> tree(draft_token_num + 1);

  int root = 0;
  int cursor = 1;
  int top_k = param.max_bfs_breadth;

  auto addToHeap = [&heap, &top_k](int parent, const TrieNode* trie_node, double prob) -> void {
    double sum_freq = 0.0;
    int count = 0;
    std::list<std::pair<TrieNode*, int32_t>> topk_children;
    for (auto* child : trie_node->sorted_children) {
      sum_freq += static_cast<double>(child->freq);
      topk_children.emplace_back(child, child->freq);
      if (++count >= top_k) break;
    }
    if (sum_freq <= 0) sum_freq = 1.0;
    for (const auto& [child, freq] : topk_children) {
      double norm_freq = static_cast<double>(freq) / sum_freq * prob;
      heap.emplace(parent, child, norm_freq);
    }
  };

  for (auto [node, _] : anchors) {
    addToHeap(root, node, 1.0);

    while (!heap.empty() && cursor <= static_cast<int>(draft_token_num)) {
      auto [parent, trie_node, prob] = heap.top();
      heap.pop();
      auto token = trie_node->token;
      int pos = -1;
      auto tit = tree[parent].next.find(token);
      if (tit != tree[parent].next.end()) {
        pos = tit->second;
      } else {
        pos = cursor++;
        tree[parent].next[token] = pos;
      }
      addToHeap(pos, trie_node, prob);
    }
  }

  return fillResult(last_token, draft_token_num + 1, tree, root);
}

}  // namespace ngram
