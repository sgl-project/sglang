#include "ngram.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <queue>
#include <vector>

namespace ngram {

struct Node {
  std::unordered_map<int32_t, int32_t> next;
};

Ngram::Result fillResult(int last_token, int draft_token_num, std::vector<Node>& tree, int root) {
  Ngram::Result info;
  std::vector<int32_t> prevs;
  info.token.reserve(draft_token_num);
  prevs.reserve(draft_token_num);
  std::queue<std::tuple<int32_t, int32_t, int32_t>> queue;
  info.token.emplace_back(last_token);
  prevs.emplace_back(-1);

  for (auto [token, next] : tree[root].next) {
    queue.emplace(token, next, 0);
  }
  while (queue.size()) {
    auto [token, next, prev] = queue.front();
    queue.pop();
    info.token.emplace_back(token);
    prevs.emplace_back(prev);
    for (auto [t, n] : tree[next].next) {
      queue.emplace(t, n, info.token.size() - 1);
    }
  }

  // zero padding to length
  while (info.token.size() < draft_token_num) {
    info.token.emplace_back(0);
    prevs.emplace_back(0);
  }

  int n = info.token.size();
  info.mask.resize(n * n, 0);
  info.mask[0] = 1;
  for (int i = 0; i < n; ++i) {
    if (prevs[i] != -1) {
      memcpy(&info.mask[i * n], &info.mask[prevs[i] * n], prevs[i] + 1);
    }
    info.mask[i * n + i] = 1;
  }

  return info;
}

Ngram::Ngram(size_t capacity, const Param& param) {
  param_ = param;
  nodes_.resize(capacity);
  for (auto& node : nodes_) {
    node_pool_.emplace_back(&node);
  }
  free_node_count_ = node_pool_.size();
  root_ = getNode();

  if (!(param_.branch_length > 1)) {
    throw std::runtime_error(
        "param_.branch_length must be greater than 1, current value: " + std::to_string(param_.branch_length));
  }
  if (!(param_.min_match_window_size > 0)) {
    throw std::runtime_error(
        "min_match_window_size must be greater than 0, current value: " + std::to_string(param_.min_match_window_size));
  }
  if (!(param_.min_match_window_size <= param_.max_match_window_size)) {
    throw std::runtime_error(
        "min_match_window_size must be less than or equal to max_match_window_size, current min_match_window_size: " +
        std::to_string(param_.min_match_window_size) +
        ", max_match_window_size: " + std::to_string(param_.max_match_window_size));
  }
  if (!(param_.max_match_window_size < param_.branch_length)) {
    throw std::runtime_error(
        "max_match_window_size must be less than branch_length, current max_match_window_size: " +
        std::to_string(param_.max_match_window_size) + ", branch_length: " + std::to_string(param_.branch_length));
  }
  if (!(param_.min_bfs_breadth > 0)) {
    throw std::runtime_error(
        "min_bfs_breadth must be greater than 0, current value: " + std::to_string(param_.min_bfs_breadth));
  }
  if (!(param_.min_bfs_breadth <= param_.max_bfs_breadth)) {
    throw std::runtime_error(
        "min_bfs_breadth must be less than or equal to max_bfs_breadth, current min_bfs_breadth: " +
        std::to_string(param_.min_bfs_breadth) + ", max_bfs_breadth: " + std::to_string(param_.max_bfs_breadth));
  }
  if (!(param_.draft_token_num > 0)) {
    throw std::runtime_error(
        "draft_token_num must be greater than 0, current value: " + std::to_string(param_.draft_token_num));
  }
  for (auto config : param_.batch_draft_token_num) {
    if (config != std::numeric_limits<decltype(config)>::max()) {
      if (!(config <= param_.draft_token_num)) {
        throw std::runtime_error(
            "batch_draft_token_num config value " + std::to_string(config) +
            " must be less than or equal to draft_token_num: " + std::to_string(param_.draft_token_num));
      }
    }
  }
  for (auto config : param_.batch_min_match_window_size) {
    if (config != std::numeric_limits<decltype(config)>::max()) {
      if (!(config >= param_.min_match_window_size)) {
        throw std::runtime_error(
            "batch_min_match_window_size config value " + std::to_string(config) +
            " must be greater than or equal to min_match_window_size: " + std::to_string(param_.min_match_window_size));
      }
      if (!(config <= param_.max_match_window_size)) {
        throw std::runtime_error(
            "batch_min_match_window_size config value " + std::to_string(config) +
            " must be less than or equal to max_match_window_size: " + std::to_string(param_.max_match_window_size));
      }
    }
  }

  quit_flag_ = false;
  insert_worker_ = std::thread(&Ngram::insert, this);
}

Ngram::~Ngram() {
  quit_flag_ = true;
  insert_queue_.close();
  insert_worker_.join();
}

std::vector<std::pair<TrieNode*, int32_t>> Ngram::match(const std::vector<int32_t>& tokens, size_t batch_size) const {
  auto draft_token_num = param_.get_draft_token_num(batch_size);
  auto min_match_window_size = param_.get_min_match_window_size(batch_size);
  auto max_match_window_size = param_.max_match_window_size;
  std::vector<std::pair<TrieNode*, int32_t>> result;
  result.reserve(param_.max_match_window_size - param_.min_match_window_size);
  for (int32_t match_window_size = std::min(tokens.size(), param_.max_match_window_size);
       match_window_size >= param_.min_match_window_size;
       --match_window_size) {
    auto start = tokens.data() + tokens.size() - match_window_size;
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

void Ngram::squeeze(size_t count) {
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
      throw std::runtime_error("The node to be released still has child nodes and cannot be released. ");
    }

    last->parent->lru.erase(last->parent_lru_pos);
    last->parent->sorted_children.erase(last);
    last->parent->child.erase(last->token);

    node_pool_[free_node_count_++] = last;
  }
}

void Ngram::synchronize() const {
  while (!insert_queue_.empty()) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}

void Ngram::insert() {
  while (!quit_flag_) {
    std::vector<int32_t> data;
    if (!insert_queue_.dequeue(data)) {
      continue;
    }
    const auto* token = data.data();
    size_t size = data.size();
    std::unique_lock<std::mutex> lock(mutex_);

    for (size_t i = 0; i + param_.min_match_window_size < size; ++i) {
      auto start = token + i;
      auto end = start + std::min(size - i, param_.branch_length);

      if (end - start > free_node_count_) {
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
}

void Ngram::asyncInsert(std::vector<std::vector<int32_t>>&& tokens) {
  for (auto&& token : tokens) {
    insert_queue_.enqueue(std::move(token));
  }
}

Ngram::Result Ngram::matchBFS(const std::vector<int32_t>& tokens, size_t batch_size) const {
  std::vector<std::pair<TrieNode*, int32_t>> nodes = match(tokens, batch_size);

  double bfs_breadth_scale = double(param_.max_bfs_breadth - param_.min_bfs_breadth) /
                             (param_.max_match_window_size - param_.min_match_window_size + 1);

  auto draft_token_num = param_.get_draft_token_num(batch_size);
  std::vector<Node> tree(draft_token_num + 1);
  int root = 0;
  int cursor = 1;

  for (auto [node, depth] : nodes) {
    std::queue<std::tuple<int32_t, double, const TrieNode*>> queue;  // parent, bfs_breadth, node
    queue.push({root, (param_.max_match_window_size - depth) * bfs_breadth_scale + param_.min_bfs_breadth, node});
    while (queue.size() && cursor <= draft_token_num) {
      auto front = queue.front();
      queue.pop();

      auto parent = std::get<0>(front);
      auto cur_breadth = std::get<1>(front);
      auto iter = std::get<2>(front)->lru.begin();

      auto breadth = std::max(1, int32_t(cur_breadth));
      for (int i = 0; i < breadth && iter != std::get<2>(front)->lru.end() && cursor <= draft_token_num; ++i, ++iter) {
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

  return fillResult(tokens.back(), draft_token_num + 1, tree, root);
}

Ngram::Result Ngram::matchProb(const std::vector<int32_t>& tokens, size_t batch_size) const {
  std::vector<std::pair<TrieNode*, int32_t>> nodes = match(tokens, batch_size);
  auto draft_token_num = param_.get_draft_token_num(batch_size);

  struct CompareByLastDouble {
    bool operator()(
        const std::tuple<double, const TrieNode*, double>& a,  // parent_pos,  node, final_prob
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
  int top_k = param_.max_bfs_breadth;

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

  for (auto [node, _] : nodes) {
    addToHeap(root, node, 1.0);

    while (!heap.empty() && cursor <= draft_token_num) {
      auto [parent, trie_node, prob] = heap.top();  // parent_pos, node, final_prob
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

  return fillResult(tokens.back(), draft_token_num + 1, tree, root);
}

Ngram::Result Ngram::batchMatch(const std::vector<std::vector<int32_t>>& tokens) const {
  std::unique_lock<std::mutex> lock(mutex_);
  Result merged_result;
  auto match_func = param_.match_type == "BFS" ? &Ngram::matchBFS : &Ngram::matchProb;
  for (const auto& tks : tokens) {
    Result res = (this->*match_func)(tks, tokens.size());
    merged_result.token.insert(merged_result.token.end(), res.token.begin(), res.token.end());
    merged_result.mask.insert(merged_result.mask.end(), res.mask.begin(), res.mask.end());
  }
  return merged_result;
}

void Ngram::Result::truncate(size_t n) {
  if (n < token.size()) {
    int full_n = token.size();
    for (int i = 1; i < n; ++i) {
      memcpy(&mask[i * n], &mask[i * full_n], sizeof(mask[0]) * n);
    }
    token.resize(n);
    mask.resize(n * n);
  }
}

}  // namespace ngram
