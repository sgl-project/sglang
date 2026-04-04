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
  // Logical generation of this TrieNode. retireNode() bumps it before the node
  // goes back to the pool so stale NodeRefs fail validation after reuse.
  uint64_t version = 1;

  struct CompareByFreq {
    bool operator()(TrieNode* a, TrieNode* b) const {
      return std::tie(b->freq, a->token, a) < std::tie(a->freq, b->token, b);
    }
  };
  std::multiset<TrieNode*, CompareByFreq> sorted_children;
};

// By-value handle to a logical trie location, cached in MatchState.
// We cannot cache TrieNode* alone across decode steps: squeeze() may evict a
// node, and getNode() may later recycle the same address for a different node.
struct NodeRef {
  TrieNode* ptr = nullptr;
  uint64_t version = 0;
};

// Per-request cached anchors. anchors[d - 1] caches the trie match for the
// length-d suffix ending at the current last token; processed_total_len records
// the full request length covered by those cached anchors.
struct MatchState {
  uint64_t trie_epoch = 0;
  size_t processed_total_len = 0;
  std::vector<NodeRef> anchors;
};

class Trie {
 public:
  Trie(size_t capacity, const Param& param);

  void insert(const int32_t* tokens, size_t len);

  Result buildRecency(
      const int32_t* context,
      size_t len,
      int32_t last_token,
      size_t draft_token_num,
      const Param& param,
      MatchState& state,
      size_t total_len) const;

  Result buildFrequency(
      const int32_t* context,
      size_t len,
      int32_t last_token,
      size_t draft_token_num,
      const Param& param,
      MatchState& state,
      size_t total_len) const;

  void squeeze(size_t count);

  void reset();

 private:
  // Stateful suffix matcher. If `state` still represents the previous step for
  // this request, infer the newly appended suffix from (`context`, `total_len`)
  // and advance anchors incrementally; otherwise rebuild the cached anchors from
  // `context`. Returns only the suffix matches that are currently expandable.
  std::vector<std::pair<const TrieNode*, int32_t>>
  match(const int32_t* context, size_t len, MatchState& state, size_t total_len) const;
  // Recompute all cached anchors from the current tail. After this, for every
  // d in [1, min(len, max_trie_depth)], anchors[d - 1] represents the suffix of
  // length d ending at context[len - 1].
  void rebuildMatchState_(const int32_t* context, size_t len, MatchState& state, size_t total_len) const;
  // Advance the cached anchors by consuming the newly appended suffix one
  // token at a time, without re-walking all suffixes from root.
  bool advanceMatchState_(MatchState& state, const int32_t* tokens, size_t len, size_t total_len) const;
  // Check that every non-empty cached NodeRef in MatchState still resolves to
  // the same logical trie node under the current trie_epoch_.
  bool validateMatchState_(const MatchState& state) const;
  // MatchState keeps all live suffix matches, including leaves. This helper
  // filters the cached anchors down to the suffixes that currently have children and
  // therefore can seed BFS / PROB draft construction.
  std::vector<std::pair<const TrieNode*, int32_t>> getExpandableAnchors_(const MatchState& state) const;
  // Resolve a cached NodeRef back to a live trie node. nullptr means the
  // cached location went stale and the caller should rebuild from context.
  const TrieNode* resolve(const MatchState& state, const NodeRef& ref) const;
  NodeRef rootRef() const {
    return NodeRef{root_, root_->version};
  }
  NodeRef capture(TrieNode* node) const {
    if (node == nullptr) {
      return {};
    }
    return NodeRef{node, node->version};
  }
  void retireNode(TrieNode* node) {
    if (node != nullptr) {
      ++node->version;
    }
  }

  TrieNode* getNode() {
    auto node = node_pool_[--free_node_count_];
    auto version = node->version;
    node->~TrieNode();
    new (node) TrieNode();
    node->version = version;
    return node;
  }

  std::vector<TrieNode> nodes_;
  std::vector<TrieNode*> node_pool_;
  size_t free_node_count_;
  std::list<TrieNode*> global_lru_;
  TrieNode* root_;
  std::vector<TrieNode*> path_;
  Param param_;
  uint64_t trie_epoch_ = 1;
};

}  // namespace ngram
