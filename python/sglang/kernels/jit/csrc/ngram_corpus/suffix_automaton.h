#pragma once

#include "param.h"
#include "result.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace sglang {

namespace ngram {

struct SamAnchor {
  int state = 0;
  int32_t matched_len = 0;
};

// Build-time state. Only lives between the first appendTokens() and finalize().
struct SamState {
  int link = -1;
  int32_t max_len = 0;
  std::unordered_map<int32_t, int> next;
  uint64_t occ_count = 0;
  int64_t max_end_pos = -1;
};

class SuffixAutomaton {
 public:
  static constexpr int32_t kSeparatorToken = std::numeric_limits<int32_t>::min();

  SuffixAutomaton();

  void appendTokens(const std::vector<int32_t>& tokens);

  // Computes occurrence counts and recency, then freezes the automaton into the
  // flat arrays below and releases the build-time states. Read-only afterwards.
  void finalize();

  bool empty() const {
    return !loaded_;
  }

  int64_t tokenCount() const {
    return pos_;
  }

  Result buildRecency(
      const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const;

  Result buildFrequency(
      const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const;

 private:
  void reset_();
  void extend_(int32_t token, int64_t pos);
  void propagateOccurrencesAndRecency_();
  void freeze_();
  // Target state of `state` on `token`, or -1 if there is no such transition.
  int transition_(int state, int32_t token) const;
  // Number of draftable children of `state`, i.e. edges other than separators.
  int numChildren_(int state) const;
  std::vector<SamAnchor> match(const int32_t* context, size_t len, size_t max_depth) const;

  // Build-time representation, released by finalize().
  std::vector<SamState> states_;
  int last_ = 0;
  int64_t pos_ = 0;
  bool saw_token_ = false;
  bool finalized_ = false;
  bool loaded_ = false;

  // Frozen representation, valid once loaded_ is set. State indices are
  // unchanged from the build, and 0 is the root.
  std::vector<int32_t> link_;
  std::vector<int32_t> max_len_;
  // Bounded by the corpus length, which int state indices already keep < 2^31.
  std::vector<uint32_t> occ_count_;
  // The edges of state s are [edge_begin_[s], edge_begin_[s + 1]), sorted by
  // token, so a separator edge (kSeparatorToken is INT32_MIN) always comes first.
  std::vector<uint32_t> edge_begin_;
  std::vector<int32_t> edge_token_;
  std::vector<int32_t> edge_target_;
  // Edge indices in drafting order, separators excluded. They share
  // edge_begin_ with the edges: the children of state s in recency order are
  // by_recency_[edge_begin_[s] + i] for i < numChildren_(s).
  std::vector<uint32_t> by_recency_;
  std::vector<uint32_t> by_freq_;
};

}  // namespace ngram

}  // namespace sglang
