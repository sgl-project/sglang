#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sglang {

namespace ngram {

struct Result {
  std::vector<int32_t> token;
  std::vector<uint8_t> mask;
  // Number of REAL nodes in `token` (the anchor plus every matched node), i.e.
  // `token[0 : num_valid)` is the tree and `token[num_valid : )` is padding.
  //
  // Only fillResult can know this. Once the tail is zero-padded the padding is
  // indistinguishable from a genuine match: a pad gets `prev = 0`, so its mask
  // row is exactly `{0, i}` -- byte-identical to a real depth-1 child whose
  // token happens to be 0 (a legal id; it is <|begin_of_sentence|> in DSV4, and
  // it IS in the trie whenever the prompt is indexed). Consumers that count
  // non-zero tokens, or walk leaf paths, therefore report a no-match result as
  // a length-1 continuation of token 0.
  int32_t num_valid = 0;

  void truncate(size_t n);
};

struct Node {
  std::unordered_map<int32_t, int32_t> next;
};

Result fillResult(int last_token, int draft_token_num, std::vector<Node>& tree, int root);
std::vector<std::vector<int32_t>> extractLeafPaths_(const Result& result);
Result buildResultFromLeafPaths_(int last_token, int draft_token_num, const std::vector<std::vector<int32_t>>& paths);
Result combineRootResults_(int last_token, int draft_token_num, const Result& primary, const Result& secondary);

}  // namespace ngram

}  // namespace sglang
