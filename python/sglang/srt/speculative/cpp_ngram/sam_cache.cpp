#include "sam_cache.h"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace ngram {

namespace {

struct CompareByProb {
  bool operator()(
      const std::tuple<double, int32_t, int32_t, int32_t>& a,
      const std::tuple<double, int32_t, int32_t, int32_t>& b) const {
    return std::get<0>(a) < std::get<0>(b);
  }
};

}  // namespace

SAMCache::SAMCache(size_t capacity, const Param& param) : capacity_(std::max<size_t>(capacity, 2)), param_(param) {
  resetStates();
}

void SAMCache::resetStates() {
  states_.clear();
  states_.emplace_back();
  init_ = 0;
  last_ = init_;
  next_position_ = 0;
}

void SAMCache::extend(int32_t token) {
  int32_t cur = static_cast<int32_t>(states_.size());
  states_.emplace_back();
  states_[cur].len = states_[last_].len + 1;
  states_[cur].base_occ = 1;
  states_[cur].base_lastpos = next_position_++;

  int32_t p = last_;
  while (p != -1 && states_[p].next.find(token) == states_[p].next.end()) {
    states_[p].next[token] = cur;
    p = states_[p].fail;
  }

  if (p == -1) {
    states_[cur].fail = init_;
    last_ = cur;
    return;
  }

  int32_t q = states_[p].next[token];
  if (states_[p].len + 1 == states_[q].len) {
    states_[cur].fail = q;
    last_ = cur;
    return;
  }

  int32_t clone = static_cast<int32_t>(states_.size());
  states_.push_back(states_[q]);
  states_[clone].len = states_[p].len + 1;
  states_[clone].base_occ = 0;
  states_[clone].base_lastpos = -1;
  states_[clone].occ = 0;
  states_[clone].lastpos = -1;
  states_[clone].by_occ.clear();
  states_[clone].by_lastpos.clear();

  while (p != -1) {
    auto it = states_[p].next.find(token);
    if (it == states_[p].next.end() || it->second != q) {
      break;
    }
    it->second = clone;
    p = states_[p].fail;
  }

  states_[q].fail = clone;
  states_[cur].fail = clone;
  last_ = cur;
}

void SAMCache::rebuild() {
  resetStates();
  for (const auto& window : windows_) {
    last_ = init_;
    for (int32_t token : window) {
      extend(token);
    }
  }

  finalizeStateMetadata();
}

void SAMCache::finalizeStateMetadata() {
  std::vector<int32_t> order(states_.size());
  for (int32_t i = 0; i < static_cast<int32_t>(states_.size()); ++i) {
    order[i] = i;
    states_[i].occ = states_[i].base_occ;
    states_[i].lastpos = states_[i].base_lastpos;
    states_[i].by_occ.clear();
    states_[i].by_lastpos.clear();
  }

  std::sort(order.begin(), order.end(), [this](int32_t a, int32_t b) { return states_[a].len > states_[b].len; });

  for (int32_t v : order) {
    int32_t link = states_[v].fail;
    if (link == -1) {
      continue;
    }
    states_[link].occ += states_[v].occ;
    states_[link].lastpos = std::max(states_[link].lastpos, states_[v].lastpos);
  }

  for (auto& state : states_) {
    for (const auto& [token, target] : state.next) {
      state.by_occ.insert(std::make_pair(-states_[target].occ, token));
      state.by_lastpos.insert(std::make_pair(-states_[target].lastpos, token));
    }
  }
}

void SAMCache::insert(const int32_t* tokens, size_t len) {
  for (size_t i = 0; i + param_.min_match_window_size < len; ++i) {
    auto start = tokens + i;
    auto end = start + std::min(len - i, param_.branch_length);
    windows_.emplace_back(start, end);

    last_ = init_;
    for (auto it = start; it != end; ++it) {
      extend(*it);
    }
  }

  while (states_.size() > capacity_ && !windows_.empty()) {
    windows_.pop_front();
    rebuild();
  }

  if (states_.size() <= capacity_) {
    finalizeStateMetadata();
  }
}

void SAMCache::squeeze(size_t count) {
  // NOTE(kpham-sgl): SAM squeeze is intentionally not trie-identical. Trie can
  // evict individual least-recent leaf nodes, while SAM states are shared across
  // many substrings and are not cheap to delete in place. The goal here is to
  // control CPU memory overhead, using state count as a proxy, not to preserve
  // the exact same local eviction behavior as TrieCache.
  while (count-- && !windows_.empty()) {
    windows_.pop_front();
  }
  rebuild();
}

void SAMCache::reset() {
  windows_.clear();
  resetStates();
}

bool SAMCache::walk(const int32_t* tokens, size_t len, int32_t& state) const {
  state = init_;
  for (size_t i = 0; i < len; ++i) {
    auto it = states_[state].next.find(tokens[i]);
    if (it == states_[state].next.end()) {
      return false;
    }
    state = it->second;
  }
  return true;
}

std::vector<SAMCache::MatchAnchor>
SAMCache::match(const int32_t* context, size_t len, size_t min_window, size_t max_window) const {
  std::vector<MatchAnchor> anchors;
  for (int32_t depth = std::min(len, max_window); depth >= static_cast<int32_t>(min_window); --depth) {
    int32_t state = init_;
    if (walk(context + len - depth, depth, state)) {
      anchors.push_back({state, depth});
    }
  }
  return anchors;
}

Result SAMCache::buildRecency(
    const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const {
  auto anchors = match(context, len, param.min_match_window_size, param.max_match_window_size);

  double bfs_breadth_scale = double(param.max_bfs_breadth - param.min_bfs_breadth) /
                             (param.max_match_window_size - param.min_match_window_size + 1);

  std::vector<Node> tree(draft_token_num + 1);
  int root = 0;
  int cursor = 1;

  for (auto [state, depth] : anchors) {
    std::queue<std::tuple<int32_t, double, int32_t>> queue;
    queue.push({root, (param.max_match_window_size - depth) * bfs_breadth_scale + param.min_bfs_breadth, state});
    while (!queue.empty() && cursor <= static_cast<int>(draft_token_num)) {
      auto [parent, cur_breadth, cur_state] = queue.front();
      queue.pop();

      int breadth = std::max(1, static_cast<int32_t>(cur_breadth));
      int taken = 0;
      for (const auto& [neg_lastpos, token] : states_[cur_state].by_lastpos) {
        (void)neg_lastpos;
        int pos = -1;
        if (auto tit = tree[parent].next.find(token); tit != tree[parent].next.end()) {
          pos = tit->second;
        } else {
          pos = tree[parent].next.insert(std::make_pair(token, cursor++)).first->second;
        }
        queue.emplace(pos, cur_breadth - bfs_breadth_scale, states_[cur_state].next.at(token));
        if (++taken >= breadth || cursor > static_cast<int>(draft_token_num)) {
          break;
        }
      }
    }
  }

  return fillResult(last_token, draft_token_num + 1, tree, root);
}

Result SAMCache::buildFrequency(
    const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const {
  auto anchors = match(context, len, param.min_match_window_size, param.max_match_window_size);

  std::priority_queue<
      std::tuple<double, int32_t, int32_t, int32_t>,
      std::vector<std::tuple<double, int32_t, int32_t, int32_t>>,
      CompareByProb>
      heap;

  std::vector<Node> tree(draft_token_num + 1);
  int root = 0;
  int cursor = 1;
  int top_k = param.max_bfs_breadth;

  auto addToHeap = [this, &heap, top_k](int parent, int32_t state, double prob) -> void {
    double sum_cnt = 0.0;
    int count = 0;
    std::vector<std::pair<int32_t, int32_t>> top_children;
    top_children.reserve(top_k);
    for (const auto& [neg_occ, token] : states_[state].by_occ) {
      int32_t occ = -neg_occ;
      sum_cnt += static_cast<double>(occ);
      top_children.emplace_back(token, occ);
      if (++count >= top_k) {
        break;
      }
    }
    if (sum_cnt <= 0.0) {
      sum_cnt = 1.0;
    }
    for (const auto& [token, occ] : top_children) {
      double next_prob = prob * static_cast<double>(occ) / sum_cnt;
      heap.emplace(next_prob, parent, token, states_[state].next.at(token));
    }
  };

  for (auto [state, _] : anchors) {
    addToHeap(root, state, 1.0);
    while (!heap.empty() && cursor <= static_cast<int>(draft_token_num)) {
      auto [prob, parent, token, target] = heap.top();
      heap.pop();

      int pos = -1;
      if (auto tit = tree[parent].next.find(token); tit != tree[parent].next.end()) {
        pos = tit->second;
      } else {
        pos = cursor++;
        tree[parent].next[token] = pos;
      }
      addToHeap(pos, target, prob);
    }
  }

  return fillResult(last_token, draft_token_num + 1, tree, root);
}

}  // namespace ngram
