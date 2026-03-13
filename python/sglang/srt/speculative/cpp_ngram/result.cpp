#include "result.h"

#include <cstring>
#include <queue>
#include <tuple>

namespace ngram {

Result fillResult(int last_token, int draft_token_num, std::vector<Node>& tree, int root) {
  Result info;
  std::vector<int32_t> prevs;
  info.token.reserve(draft_token_num);
  prevs.reserve(draft_token_num);
  std::queue<std::tuple<int32_t, int32_t, int32_t>> queue;
  info.token.emplace_back(last_token);
  prevs.emplace_back(-1);

  for (auto [token, next] : tree[root].next) {
    queue.emplace(token, next, 0);
  }
  while (queue.size()) {
    auto [token, next, prev] = queue.front();
    queue.pop();
    info.token.emplace_back(token);
    prevs.emplace_back(prev);
    for (auto [t, n] : tree[next].next) {
      queue.emplace(t, n, info.token.size() - 1);
    }
  }

  // zero padding to length
  while (info.token.size() < static_cast<size_t>(draft_token_num)) {
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

void Result::truncate(size_t n) {
  if (n < token.size()) {
    int full_n = token.size();
    for (size_t i = 1; i < n; ++i) {
      memcpy(&mask[i * n], &mask[i * full_n], sizeof(mask[0]) * n);
    }
    token.resize(n);
    mask.resize(n * n);
  }
}

}  // namespace ngram
