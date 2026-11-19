#pragma once

#include <cstddef>
#include <string>

namespace sglang {

namespace ngram {

struct Param {
  size_t min_bfs_breadth;
  size_t max_bfs_breadth;
  size_t max_trie_depth;
  size_t draft_token_num;
  size_t external_sam_budget = 0;
  size_t external_corpus_max_tokens = 10000000;
  std::string match_type;
};

}  // namespace ngram

}  // namespace sglang
