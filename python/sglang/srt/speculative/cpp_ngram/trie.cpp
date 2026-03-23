#include "trie.h"

#include <algorithm>
#include <cstring>
#include <list>
#include <queue>
#include <tuple>
#include <vector>

namespace ngram {

Trie::Trie(size_t capacity, const Param& param) : param_(param) {
  nodes_.resize(capacity);
  for (auto& node : nodes_) {
    node_pool_.emplace_back(&node);
  }
  free_node_count_ = node_pool_.size();
  root_ = getNode();
}

void Trie::insert(const int32_t* tokens, size_t len) {
  for (size_t i = 0; i + param_.min_match_window_size < len; ++i) {
    auto start = tokens + i;
    auto end = start + std::min(len - i, param_.branch_length);

    if (static_cast<size_t>(end - start) > free_node_count_) {
      squeeze(end - start - free_node_count_);
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
        global_lru_.emplace_back(node);

        node->token = token;
        node->parent = cursor;
        node->parent_lru_pos = cursor->lru.begin();
        node->global_lru_pos = --global_lru_.end();
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
      global_lru_.splice(global_lru_.begin(), global_lru_, node->global_lru_pos);
    }
  }
}

void Trie::squeeze(size_t count) {
  if (!(node_pool_.size() >= free_node_count_ + count)) {
    throw std::runtime_error(
        "Insufficient node size to release required nodes. "
        "available to release: " +
        std::to_string(node_pool_.size() - free_node_count_) + ", required to release: " + std::to_string(count));
  }
  while (count--) {
    auto last = global_lru_.back();
    global_lru_.pop_back();

    if (!last->child.empty()) {
      throw std::runtime_error(
          "The node to be released still has child nodes and cannot be "
          "released. ");
    }

    last->parent->lru.erase(last->parent_lru_pos);
    last->parent->sorted_children.erase(last);
    last->parent->child.erase(last->token);

    node_pool_[free_node_count_++] = last;
  }
}

void Trie::reset() {
  global_lru_.clear();
  path_.clear();
  node_pool_.clear();
  for (auto& node : nodes_) {
    node_pool_.emplace_back(&node);
  }
  free_node_count_ = node_pool_.size();
  root_ = getNode();
}

std::vector<std::pair<TrieNode*, int32_t>>
Trie::match(const int32_t* context, size_t len, size_t min_window, size_t max_window) const {
  std::vector<std::pair<TrieNode*, int32_t>> result;
  result.reserve(max_window - min_window);
  for (int32_t match_window_size = std::min(len, max_window); match_window_size >= static_cast<int32_t>(min_window);
       --match_window_size) {
    auto start = context + len - match_window_size;
    auto end = start + match_window_size;
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
    if (cursor) {
      result.emplace_back(std::make_pair(cursor, match_window_size));
    }
  }
  return result;
}

Result Trie::buildRecency(
    const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const {
  auto anchors = match(context, len, param.min_match_window_size, param.max_match_window_size);

  double bfs_breadth_scale = double(param.max_bfs_breadth - param.min_bfs_breadth) /
                             (param.max_match_window_size - param.min_match_window_size + 1);

  std::vector<Node> tree(draft_token_num + 1);
  int root = 0;
  int cursor = 1;

  for (auto [node, depth] : anchors) {
    std::queue<std::tuple<int32_t, double, const TrieNode*>> queue;
    queue.push({root, (param.max_match_window_size - depth) * bfs_breadth_scale + param.min_bfs_breadth, node});
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

Result Trie::buildFrequency(
    const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const {
  auto anchors = match(context, len, param.min_match_window_size, param.max_match_window_size);

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
