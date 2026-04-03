#pragma once

#include "param.h"
#include "result.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <new>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace ngram {

struct TrieNode {
  std::unordered_map<int32_t, TrieNode*> child;
  std::list<TrieNode*>::const_iterator global_lru_pos;
  std::list<TrieNode*>::const_iterator parent_lru_pos;
  int32_t token;
  TrieNode* parent;
  std::list<TrieNode*> lru;
  int32_t freq = 0;

  struct CompareByFreq {
    bool operator()(TrieNode* a, TrieNode* b) const {
      return std::tie(b->freq, a->token, a) < std::tie(a->freq, b->token, b);
    }
  };
  std::multiset<TrieNode*, CompareByFreq> sorted_children;
};

class Trie {
 public:
  Trie(size_t capacity, const Param& param);

  void insert(const int32_t* tokens, size_t len);

  Result buildRecency(
      const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const;

  Result buildFrequency(
      const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const;

  void squeeze(size_t count);

  void reset();

 private:
  std::vector<std::pair<TrieNode*, int32_t>> match(const int32_t* context, size_t len) const;

  TrieNode* getNode() {
    auto node = node_pool_[--free_node_count_];
    node->~TrieNode();
    new (node) TrieNode();
    return node;
  }

  std::vector<TrieNode> nodes_;
  std::vector<TrieNode*> node_pool_;
  size_t free_node_count_;
  std::list<TrieNode*> global_lru_;
  TrieNode* root_;
  std::vector<TrieNode*> path_;
  Param param_;
};

}  // namespace ngram
