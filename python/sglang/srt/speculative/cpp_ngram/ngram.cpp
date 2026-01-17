#include "ngram.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <list>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
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

  for (auto const& [token, next] : tree[root].next) {
    queue.emplace(token, next, 0);
  }
  while (queue.size()) {
    auto [token, next, prev] = queue.front();
    queue.pop();
    info.token.emplace_back(token);
    prevs.emplace_back(prev);
    for (auto const& [t, n] : tree[next].next) {
      if (info.token.size() < draft_token_num) {
        queue.emplace(t, n, info.token.size() - 1);
      }
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

std::vector<std::pair<SAMNode*, int32_t>> Ngram::match(const std::vector<int32_t>& tokens, size_t batch_size) const {
  if (tokens.empty()) return {};
  size_t max_k = std::min(tokens.size(), param_.max_match_window_size);
  size_t min_k = param_.get_min_match_window_size(batch_size);

  SAMNode* curr = root_;
  int matched_len = 0;
  for (size_t i = tokens.size() - max_k; i < tokens.size(); ++i) {
    int32_t t = tokens[i];
    while (curr != root_ && !curr->next.count(t)) {
      curr = curr->fail;
      matched_len = curr->len;
    }
    if (curr->next.count(t)) {
      curr = curr->next.at(t).target;
      matched_len++;
    }
  }

  std::vector<std::pair<SAMNode*, int32_t>> results;
  SAMNode* temp = curr;
  int current_len = matched_len;

  while (temp && current_len >= min_k) {
    int next_len = temp->fail ? temp->fail->len : 0;
    for (int l = current_len; l > next_len; --l) {
      if (l >= min_k && l <= max_k) {
        results.push_back({temp, l});
      }
    }
    temp = temp->fail;
    current_len = next_len;
  }
  return results;
}

void Ngram::squeeze(size_t count) {
  reset();
}

void Ngram::extend(int32_t t, SAMNode*& last) {
  SAMNode* old_last = last;
  SAMNode* p = last;

  if (p->next.count(t)) {
    SAMNode* q = p->next[t].target;
    if (p->len + 1 == q->len) {
      last = q;
    } else {
      SAMNode* clone = getNode();
      clone->len = p->len + 1;
      clone->next = q->next;
      clone->fail = q->fail;
      clone->token = q->token;
      clone->lru = q->lru;
      for (int32_t tid : clone->lru) {
        clone->next[tid].lru_it = std::find(clone->lru.begin(), clone->lru.end(), tid);
        clone->sorted_children.insert(tid);
      }
      while (p && p->next.count(t) && p->next[t].target == q) {
        p->next[t].target = clone;
        p = p->fail;
      }
      q->fail = clone;
      last = clone;
    }
  } else {
    SAMNode* cur = getNode();
    cur->len = last->len + 1;
    cur->token = t;
    while (p && !p->next.count(t)) {
      p->lru.push_front(t);
      p->next[t] = {cur, 0, p->lru.begin()};
      p->sorted_children.insert(t);
      p = p->fail;
    }
    if (!p) {
      cur->fail = root_;
    } else {
      SAMNode* q = p->next[t].target;
      if (p->len + 1 == q->len) {
        cur->fail = q;
      } else {
        SAMNode* clone = getNode();
        clone->len = p->len + 1;
        clone->next = q->next;
        clone->fail = q->fail;
        clone->token = q->token;
        clone->lru = q->lru;
        for (int32_t tid : clone->lru) {
          clone->next[tid].lru_it = std::find(clone->lru.begin(), clone->lru.end(), tid);
          clone->sorted_children.insert(tid);
        }
        while (p && p->next.count(t) && p->next[t].target == q) {
          p->next[t].target = clone;
          p = p->fail;
        }
        q->fail = cur->fail = clone;
      }
    }
    last = cur;
  }

  SAMNode* path_ptr = old_last;
  int steps = 0;
  while (path_ptr && steps < param_.branch_length) {
    path_ptr->update_freq(t, 1);
    path_ptr = path_ptr->fail;
    steps++;
  }
}

void Ngram::insert() {
  while (!quit_flag_) {
    std::vector<int32_t> data;
    if (!insert_queue_.dequeue(data)) continue;
    std::unique_lock<std::mutex> lock(mutex_);

    SAMNode* last = root_;
    for (int32_t t : data) {
      if (free_node_count_ < 10) {
        lock.unlock();
        reset();
        lock.lock();
        last = root_;
      }
      extend(t, last);
    }
  }
}

void Ngram::synchronize() const {
  while (!insert_queue_.empty())
    std::this_thread::sleep_for(std::chrono::microseconds(10));
}

void Ngram::asyncInsert(std::vector<std::vector<int32_t>>&& tokens) {
  for (auto&& t : tokens)
    insert_queue_.enqueue(std::move(t));
}

Ngram::Result Ngram::matchBFS(const std::vector<int32_t>& tokens, size_t batch_size) const {
  auto draft_token_num = param_.get_draft_token_num(batch_size);
  std::vector<std::pair<SAMNode*, int32_t>> nodes = match(tokens, batch_size);
  double bfs_breadth_scale = double(param_.max_bfs_breadth - param_.min_bfs_breadth) /
                             (param_.max_match_window_size - param_.min_match_window_size + 1);

  std::vector<Node> tree(draft_token_num + 1);
  int root = 0, cursor = 1;

  for (auto [node, depth] : nodes) {
    std::queue<std::tuple<int32_t, double, const SAMNode*>> queue;
    queue.push({root, (param_.max_match_window_size - depth) * bfs_breadth_scale + param_.min_bfs_breadth, node});
    while (!queue.empty() && cursor <= draft_token_num) {
      auto [parent, cur_breadth, sam_node] = queue.front();
      queue.pop();
      int breadth = std::max(1, cur_breadth);
      int count = 0;
      auto iter = sam_node->lru.begin();
      for (; count < breadth && iter != sam_node->lru.end() && cursor <= draft_token_num; ++count, ++iter) {
        int32_t t_id = *iter;
        int pos;
        if (tree[parent].next.count(t_id))
          pos = tree[parent].next[t_id];
        else
          pos = tree[parent].next[t_id] = cursor++;
        queue.emplace(pos, cur_breadth - bfs_breadth_scale, sam_node->next.at(t_id).target);
      }
    }
  }

  return fillResult(tokens.back(), draft_token_num + 1, tree, root);
}

Ngram::Result Ngram::matchProb(const std::vector<int32_t>& tokens, size_t batch_size) const {
  auto draft_token_num = param_.get_draft_token_num(batch_size);
  std::vector<std::pair<SAMNode*, int32_t>> nodes = match(tokens, batch_size);
  struct HeapNode {
    int parent;
    const SAMNode* sam_node;
    double prob;
    bool operator<(const HeapNode& o) const {
      return prob < o.prob;
    }
  };
  std::vector<Node> tree(draft_token_num + 1);
  int root = 0, cursor = 1, top_k = param_.max_bfs_breadth;

  auto addToHeap = [&](std::priority_queue<HeapNode>& heap, int parent, const SAMNode* sam_node, double prob) {
    double sum_freq = 0;
    int count_sum = 0;
    std::vector<int32_t> topk_tokens;
    for (int32_t tid : sam_node->sorted_children) {
      sum_freq += sam_node->next.at(tid).freq;
      topk_tokens.push_back(tid);
      if (++count_sum >= top_k) break;
    }
    if (sum_freq <= 0) return;
    for (int32_t tid : topk_tokens) {
      double norm_freq = (double)sam_node->next.at(tid).freq / sum_freq * prob;
      heap.push({parent, sam_node->next.at(tid).target, norm_freq});
    }
  };

  for (auto const& res : nodes) {
    std::priority_queue<HeapNode> heap;
    addToHeap(heap, root, res.first, 1.0);
    while (!heap.empty() && cursor <= draft_token_num) {
      HeapNode top = heap.top();
      heap.pop();
      int32_t t_id = top.sam_node->token;
      int pos;
      if (tree[top.parent].next.count(t_id))
        pos = tree[top.parent].next[t_id];
      else
        pos = tree[top.parent].next[t_id] = cursor++;
      addToHeap(heap, pos, top.sam_node, top.prob);
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
    for (int i = 1; i < n; ++i)
      memcpy(&mask[i * n], &mask[i * full_n], sizeof(mask[0]) * n);
    token.resize(n);
    mask.resize(n * n);
  }
}

}  // namespace ngram
