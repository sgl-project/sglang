#pragma once

#include "param.h"
#include "result.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace ngram {

struct SamAnchor {
  int state = 0;
  int32_t matched_len = 0;
};

struct SamState {
  int link = -1;
  int32_t max_len = 0;
  std::unordered_map<int32_t, int> next;
  uint64_t occ_count = 0;
  int64_t max_end_pos = -1;
  std::vector<std::pair<int32_t, int>> children_by_freq;
  std::vector<std::pair<int32_t, int>> children_by_recency;
};

class SuffixAutomaton {
 public:
  static constexpr int32_t kSeparatorToken = std::numeric_limits<int32_t>::min();

  SuffixAutomaton();

  void appendTokens(const std::vector<int32_t>& tokens);

  void finalize();

  bool empty() const {
    return !loaded_;
  }

  Result buildRecency(
      const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const;

  Result buildFrequency(
      const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const;

 private:
  void reset_();
  void extend_(int32_t token, int64_t pos);
  void propagateOccurrencesAndRecency_();
  std::vector<SamAnchor> match(const int32_t* context, size_t len, size_t max_depth) const;

  std::vector<SamState> states_;
  int last_ = 0;
  int64_t pos_ = 0;
  bool saw_token_ = false;
  bool finalized_ = false;
  bool loaded_ = false;
};

}  // namespace ngram
