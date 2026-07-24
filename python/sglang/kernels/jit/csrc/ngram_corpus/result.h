#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ngram {

struct Result {
  std::vector<int32_t> token;
  std::vector<uint8_t> mask;

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
