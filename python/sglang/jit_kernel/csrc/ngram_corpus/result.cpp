#include "result.h"

#include <algorithm>
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

std::vector<std::vector<int32_t>> extractLeafPaths_(const Result& result) {
  const auto n = static_cast<int>(result.token.size());
  if (n <= 1) {
    return {};
  }

  std::vector<int> parent(n, -1);
  std::vector<bool> has_child(n, false);
  for (int i = 1; i < n; ++i) {
    for (int j = i - 1; j >= 0; --j) {
      if (result.mask[i * n + j]) {
        parent[i] = j;
        has_child[j] = true;
        break;
      }
    }
  }

  std::vector<std::vector<int32_t>> paths;
  for (int leaf = 1; leaf < n; ++leaf) {
    if (has_child[leaf]) {
      continue;
    }
    std::vector<int32_t> path;
    for (int cursor = leaf; cursor > 0; cursor = parent[cursor]) {
      path.emplace_back(result.token[cursor]);
    }
    std::reverse(path.begin(), path.end());
    if (path.size() == 1 && path.front() == 0) {
      continue;
    }
    paths.emplace_back(std::move(path));
  }
  return paths;
}

Result buildResultFromLeafPaths_(int last_token, int draft_token_num, const std::vector<std::vector<int32_t>>& paths) {
  std::vector<Node> tree(draft_token_num);
  const int root = 0;
  int cursor = 1;
  for (const auto& path : paths) {
    int parent = root;
    for (const auto token : path) {
      auto iter = tree[parent].next.find(token);
      if (iter == tree[parent].next.end()) {
        if (cursor >= draft_token_num) {
          parent = -1;
          break;
        }
        iter = tree[parent].next.insert({token, cursor++}).first;
      }
      parent = iter->second;
    }
    if (cursor >= draft_token_num) {
      break;
    }
  }
  return fillResult(last_token, draft_token_num, tree, root);
}

Result combineRootResults_(int last_token, int draft_token_num, const Result& primary, const Result& secondary) {
  auto primary_paths = extractLeafPaths_(primary);
  auto secondary_paths = extractLeafPaths_(secondary);
  std::vector<std::vector<int32_t>> merged_paths = std::move(primary_paths);
  merged_paths.reserve(merged_paths.size() + secondary_paths.size());
  for (const auto& path : secondary_paths) {
    if (path.empty()) {
      continue;
    }
    merged_paths.emplace_back(path);
  }

  return buildResultFromLeafPaths_(last_token, draft_token_num, merged_paths);
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
