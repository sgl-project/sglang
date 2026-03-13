#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <numeric>
#include <set>
#include <unordered_map>
#include <vector>

#include "param.h"
#include "result.h"

namespace ngram {

struct SAMNode {
  std::unordered_map<int32_t, int32_t> next;
  int32_t fail = -1;
  int32_t len = 0;
  int32_t base_occ = 0;
  int32_t base_lastpos = -1;
  int32_t occ = 0;
  int32_t lastpos = -1;

  std::set<std::pair<int32_t, int32_t>> by_occ;
  std::set<std::pair<int32_t, int32_t>> by_lastpos;
};

class SAMCache {
 public:
  SAMCache(size_t capacity, const Param& param);

  void insert(const int32_t* tokens, size_t len);

  Result buildRecency(
      const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const;

  Result buildFrequency(
      const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const;

  void squeeze(size_t count);

  void reset();

 private:
  struct MatchAnchor {
    int32_t state;
    int32_t depth;
  };

  void resetStates();
  void rebuild();
  void finalizeStateMetadata();
  void extend(int32_t token);
  std::vector<MatchAnchor> match(const int32_t* context, size_t len, size_t min_window, size_t max_window) const;
  bool walk(const int32_t* tokens, size_t len, int32_t& state) const;

  size_t capacity_;
  Param param_;
  std::deque<std::vector<int32_t>> windows_;
  std::vector<SAMNode> states_;
  int32_t last_ = 0;
  int32_t init_ = 0;
  int32_t next_position_ = 0;
};

}  // namespace ngram
