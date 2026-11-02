#include "suffix_automaton.h"

#include <algorithm>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>

namespace sglang {

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
    std::vector<SamState>().swap(states_);
    return;
  }

  propagateOccurrencesAndRecency_();
  freeze_();
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
}

void SuffixAutomaton::freeze_() {
  const size_t num_states = states_.size();
  size_t num_edges = 0;
  for (const auto& state : states_) {
    num_edges += state.next.size();
  }

  link_.resize(num_states);
  max_len_.resize(num_states);
  occ_count_.resize(num_states);
  edge_begin_.resize(num_states + 1);
  edge_token_.resize(num_edges);
  edge_target_.resize(num_edges);
  by_recency_.resize(num_edges);
  by_freq_.resize(num_edges);

  // Drafting orders: most frequent / most recent child first, ties broken by
  // token and then by state.
  auto more_frequent = [this](uint32_t lhs, uint32_t rhs) {
    const auto lhs_freq = states_[edge_target_[lhs]].occ_count;
    const auto rhs_freq = states_[edge_target_[rhs]].occ_count;
    return std::tie(rhs_freq, edge_token_[lhs], edge_target_[lhs]) <
           std::tie(lhs_freq, edge_token_[rhs], edge_target_[rhs]);
  };
  auto more_recent = [this](uint32_t lhs, uint32_t rhs) {
    const auto lhs_recency = states_[edge_target_[lhs]].max_end_pos;
    const auto rhs_recency = states_[edge_target_[rhs]].max_end_pos;
    return std::tie(rhs_recency, edge_token_[lhs], edge_target_[lhs]) <
           std::tie(lhs_recency, edge_token_[rhs], edge_target_[rhs]);
  };

  std::vector<std::pair<int32_t, int>> edges;
  uint32_t cursor = 0;
  for (size_t s = 0; s < num_states; ++s) {
    auto& state = states_[s];
    link_[s] = state.link;
    max_len_[s] = state.max_len;
    occ_count_[s] = static_cast<uint32_t>(state.occ_count);

    const uint32_t begin = cursor;
    edge_begin_[s] = begin;
    edges.assign(state.next.begin(), state.next.end());
    std::sort(edges.begin(), edges.end());
    for (const auto& [token, target] : edges) {
      edge_token_[cursor] = token;
      edge_target_[cursor] = target;
      ++cursor;
    }

    // Separator edges sort first and are never drafted.
    const uint32_t first_child = (begin != cursor && edge_token_[begin] == kSeparatorToken) ? begin + 1 : begin;
    const uint32_t num_children = cursor - first_child;
    std::iota(by_freq_.begin() + begin, by_freq_.begin() + begin + num_children, first_child);
    std::sort(by_freq_.begin() + begin, by_freq_.begin() + begin + num_children, more_frequent);
    std::iota(by_recency_.begin() + begin, by_recency_.begin() + begin + num_children, first_child);
    std::sort(by_recency_.begin() + begin, by_recency_.begin() + begin + num_children, more_recent);

    // The comparators above only read occ_count / max_end_pos, so this state's
    // transitions can be released as soon as they have been copied.
    std::unordered_map<int32_t, int>().swap(state.next);
  }
  edge_begin_[num_states] = cursor;

  std::vector<SamState>().swap(states_);
}

int SuffixAutomaton::transition_(int state, int32_t token) const {
  const auto first = edge_token_.begin() + edge_begin_[state];
  const auto last = edge_token_.begin() + edge_begin_[state + 1];
  const auto iter = std::lower_bound(first, last, token);
  if (iter == last || *iter != token) {
    return -1;
  }
  return edge_target_[iter - edge_token_.begin()];
}

int SuffixAutomaton::numChildren_(int state) const {
  const uint32_t begin = edge_begin_[state];
  const uint32_t end = edge_begin_[state + 1];
  const bool has_separator = begin != end && edge_token_[begin] == kSeparatorToken;
  return static_cast<int>(end - begin) - (has_separator ? 1 : 0);
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
    int next_state = transition_(state, token);
    while (state != 0 && next_state < 0) {
      state = link_[state];
      matched_len = std::min<int32_t>(matched_len, max_len_[state]);
      next_state = transition_(state, token);
    }
    // The loop stops at a state that has the transition, or at the root, so a
    // miss here means the root has no transition on `token` either.
    if (next_state >= 0) {
      state = next_state;
      ++matched_len;
    } else {
      state = 0;
      matched_len = 0;
    }
  }

  std::vector<SamAnchor> anchors;
  while (state > 0 && matched_len > 0) {
    if (numChildren_(state) > 0) {
      anchors.push_back({state, matched_len});
    }
    state = link_[state];
    if (state <= 0) {
      break;
    }
    matched_len = std::min<int32_t>(matched_len, max_len_[state]);
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

      const uint32_t children = edge_begin_[state];
      const int num_children = numChildren_(state);
      const auto breadth = std::max(1, static_cast<int32_t>(cur_breadth));
      for (int i = 0; i < breadth && i < num_children && cursor <= static_cast<int>(draft_token_num); ++i) {
        const uint32_t edge = by_recency_[children + i];
        const int32_t token = edge_token_[edge];
        const int child_state = edge_target_[edge];
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
    const uint32_t children = edge_begin_[state];
    const int num_children = std::min(numChildren_(state), top_k);
    if (num_children == 0) {
      return;
    }
    double sum_freq = 0.0;
    for (int i = 0; i < num_children; ++i) {
      sum_freq += static_cast<double>(occ_count_[edge_target_[by_freq_[children + i]]]);
    }
    if (sum_freq <= 0) {
      sum_freq = 1.0;
    }
    for (int i = 0; i < num_children; ++i) {
      const uint32_t edge = by_freq_[children + i];
      const int child_state = edge_target_[edge];
      const auto scaled_prob = static_cast<double>(occ_count_[child_state]) / sum_freq * prob;
      heap.emplace(parent, edge_token_[edge], child_state, scaled_prob);
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

}  // namespace sglang
