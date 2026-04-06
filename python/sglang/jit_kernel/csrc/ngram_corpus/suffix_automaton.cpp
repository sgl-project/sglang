#include "suffix_automaton.h"

#include <algorithm>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>

namespace ngram {

SuffixAutomaton::SuffixAutomaton() {
  reset_();
}

void SuffixAutomaton::reset_() {
  states_.clear();
  states_.emplace_back();
  last_ = 0;
  pos_ = 0;
  saw_token_ = false;
  finalized_ = false;
  loaded_ = false;
}

void SuffixAutomaton::appendTokens(const std::vector<int32_t>& tokens) {
  if (finalized_) {
    throw std::runtime_error("Cannot append tokens after finalizing the SAM.");
  }
  if (tokens.empty()) {
    return;
  }

  for (const auto token : tokens) {
    extend_(token, pos_++);
    saw_token_ = true;
  }
}

void SuffixAutomaton::finalize() {
  if (finalized_) {
    return;
  }
  finalized_ = true;
  if (!saw_token_) {
    return;
  }

  propagateOccurrencesAndRecency_();
  loaded_ = true;
}

void SuffixAutomaton::extend_(int32_t token, int64_t pos) {
  const int cur = static_cast<int>(states_.size());
  states_.emplace_back();
  states_[cur].max_len = states_[last_].max_len + 1;
  states_[cur].occ_count = 1;
  states_[cur].max_end_pos = pos;

  int p = last_;
  while (p != -1 && !states_[p].next.contains(token)) {
    states_[p].next[token] = cur;
    p = states_[p].link;
  }

  if (p == -1) {
    states_[cur].link = 0;
    last_ = cur;
    return;
  }

  const int q = states_[p].next[token];
  if (states_[p].max_len + 1 == states_[q].max_len) {
    states_[cur].link = q;
    last_ = cur;
    return;
  }

  const int clone = static_cast<int>(states_.size());
  states_.push_back(states_[q]);
  states_[clone].max_len = states_[p].max_len + 1;
  states_[clone].occ_count = 0;
  states_[clone].children_by_freq.clear();
  states_[clone].children_by_recency.clear();

  while (p != -1 && states_[p].next[token] == q) {
    states_[p].next[token] = clone;
    p = states_[p].link;
  }

  states_[q].link = clone;
  states_[cur].link = clone;
  last_ = cur;
}

void SuffixAutomaton::propagateOccurrencesAndRecency_() {
  std::vector<int> order(states_.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(
      order.begin(), order.end(), [this](int lhs, int rhs) { return states_[lhs].max_len < states_[rhs].max_len; });

  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    const int state = *it;
    const int link = states_[state].link;
    if (link < 0) {
      continue;
    }
    states_[link].occ_count += states_[state].occ_count;
    states_[link].max_end_pos = std::max(states_[link].max_end_pos, states_[state].max_end_pos);
  }

  for (auto& state : states_) {
    state.children_by_freq.clear();
    state.children_by_recency.clear();
    state.children_by_freq.reserve(state.next.size());
    state.children_by_recency.reserve(state.next.size());
    for (const auto& [token, child_state] : state.next) {
      if (token == kSeparatorToken) {
        continue;
      }
      state.children_by_freq.emplace_back(token, child_state);
      state.children_by_recency.emplace_back(token, child_state);
    }

    std::sort(state.children_by_freq.begin(), state.children_by_freq.end(), [this](const auto& lhs, const auto& rhs) {
      const auto lhs_freq = states_[lhs.second].occ_count;
      const auto rhs_freq = states_[rhs.second].occ_count;
      return std::tie(rhs_freq, lhs.first, lhs.second) < std::tie(lhs_freq, rhs.first, rhs.second);
    });
    std::sort(
        state.children_by_recency.begin(), state.children_by_recency.end(), [this](const auto& lhs, const auto& rhs) {
          const auto lhs_recency = states_[lhs.second].max_end_pos;
          const auto rhs_recency = states_[rhs.second].max_end_pos;
          return std::tie(rhs_recency, lhs.first, lhs.second) < std::tie(lhs_recency, rhs.first, rhs.second);
        });
  }
}

std::vector<SamAnchor> SuffixAutomaton::match(const int32_t* context, size_t len, size_t max_depth) const {
  if (empty() || len == 0) {
    return {};
  }

  const auto start = len > max_depth ? len - max_depth : 0;
  int state = 0;
  int32_t matched_len = 0;
  for (size_t i = start; i < len; ++i) {
    const auto token = context[i];
    while (state != 0 && !states_[state].next.contains(token)) {
      state = states_[state].link;
      matched_len = std::min<int32_t>(matched_len, states_[state].max_len);
    }
    if (auto iter = states_[state].next.find(token); iter != states_[state].next.end()) {
      state = iter->second;
      ++matched_len;
    } else if (auto root_iter = states_[0].next.find(token); root_iter != states_[0].next.end()) {
      state = root_iter->second;
      matched_len = 1;
    } else {
      state = 0;
      matched_len = 0;
    }
  }

  std::vector<SamAnchor> anchors;
  while (state > 0 && matched_len > 0) {
    if (!states_[state].children_by_freq.empty()) {
      anchors.push_back({state, matched_len});
    }
    state = states_[state].link;
    if (state <= 0) {
      break;
    }
    matched_len = std::min<int32_t>(matched_len, states_[state].max_len);
  }
  return anchors;
}

Result SuffixAutomaton::buildRecency(
    const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const {
  auto anchors = match(context, len, param.max_trie_depth);
  const auto max_match_depth = std::max<int32_t>(1, static_cast<int32_t>(param.max_trie_depth - 1));
  const double bfs_breadth_scale = double(param.max_bfs_breadth - param.min_bfs_breadth) / max_match_depth;
  std::vector<Node> tree(draft_token_num + 1);
  int root = 0;
  int cursor = 1;

  for (const auto& anchor : anchors) {
    std::queue<std::tuple<int, double, int>> queue;
    queue.push(
        {root, (max_match_depth - anchor.matched_len) * bfs_breadth_scale + param.min_bfs_breadth, anchor.state});
    while (!queue.empty() && cursor <= static_cast<int>(draft_token_num)) {
      auto [parent, cur_breadth, state] = queue.front();
      queue.pop();

      const auto& children = states_[state].children_by_recency;
      const auto breadth = std::max(1, static_cast<int32_t>(cur_breadth));
      for (int i = 0;
           i < breadth && i < static_cast<int>(children.size()) && cursor <= static_cast<int>(draft_token_num);
           ++i) {
        const auto [token, child_state] = children[i];
        int pos = -1;
        if (auto iter = tree[parent].next.find(token); iter != tree[parent].next.end()) {
          pos = iter->second;
        } else {
          pos = tree[parent].next.insert({token, cursor++}).first->second;
        }
        queue.emplace(pos, cur_breadth - bfs_breadth_scale, child_state);
      }
    }
  }
  return fillResult(last_token, draft_token_num + 1, tree, root);
}

Result SuffixAutomaton::buildFrequency(
    const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const {
  auto anchors = match(context, len, param.max_trie_depth);
  struct CompareByProb {
    bool operator()(
        const std::tuple<int, int32_t, int, double>& lhs, const std::tuple<int, int32_t, int, double>& rhs) const {
      return std::get<3>(lhs) < std::get<3>(rhs);
    }
  };

  std::priority_queue<
      std::tuple<int, int32_t, int, double>,
      std::vector<std::tuple<int, int32_t, int, double>>,
      CompareByProb>
      heap;
  std::vector<Node> tree(draft_token_num + 1);
  int root = 0;
  int cursor = 1;
  const int top_k = static_cast<int>(param.max_bfs_breadth);

  auto addToHeap = [this, &heap, top_k](int parent, int state, double prob) {
    if (top_k <= 0) {
      return;
    }
    const auto& children = states_[state].children_by_freq;
    if (children.empty()) {
      return;
    }
    double sum_freq = 0.0;
    int count = 0;
    for (const auto& [_, child_state] : children) {
      sum_freq += static_cast<double>(states_[child_state].occ_count);
      if (++count >= top_k) {
        break;
      }
    }
    if (sum_freq <= 0) {
      sum_freq = 1.0;
    }
    count = 0;
    for (const auto& [token, child_state] : children) {
      const auto scaled_prob = static_cast<double>(states_[child_state].occ_count) / sum_freq * prob;
      heap.emplace(parent, token, child_state, scaled_prob);
      if (++count >= top_k) {
        break;
      }
    }
  };

  for (const auto& anchor : anchors) {
    addToHeap(root, anchor.state, 1.0);
    while (!heap.empty() && cursor <= static_cast<int>(draft_token_num)) {
      auto [parent, token, child_state, prob] = heap.top();
      heap.pop();

      int pos = -1;
      if (auto iter = tree[parent].next.find(token); iter != tree[parent].next.end()) {
        pos = iter->second;
      } else {
        pos = cursor++;
        tree[parent].next[token] = pos;
      }
      addToHeap(pos, child_state, prob);
    }
  }
  return fillResult(last_token, draft_token_num + 1, tree, root);
}

}  // namespace ngram
