#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "param.h"
#include "queue.h"

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

class Ngram {
  std::vector<TrieNode> nodes_;
  std::vector<TrieNode*> node_pool_;
  size_t free_node_count_;
  std::list<TrieNode*> global_lru_;
  TrieNode* root_;
  std::vector<TrieNode*> path_;
  Param param_;

  std::vector<std::pair<TrieNode*, int32_t>> match(const std::vector<int32_t>& tokens, size_t batch_size) const;

  void squeeze(size_t count);

  TrieNode* getNode() {
    auto node = node_pool_[--free_node_count_];
    node->~TrieNode();
    new (node) TrieNode();
    return node;
  }

  mutable std::mutex mutex_;
  bool quit_flag_;
  utils::Queue<std::vector<int32_t>> insert_queue_;
  std::thread insert_worker_;
  std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t>> match_tmp_data_;

 public:
  Ngram(size_t capacity, const Param& param);
  Ngram() = default;
  ~Ngram();

  static Ngram& instance() {
    static Ngram instance;
    return instance;
  }

  void synchronize() const;

  void asyncInsert(std::vector<std::vector<int32_t>>&& tokens);

  struct Result {
    std::vector<int32_t> token;
    std::vector<uint8_t> mask;

    void truncate(size_t n);
  };

  Result batchMatch(const std::vector<std::vector<int32_t>>& tokens) const;

  void reset() {
    std::unique_lock<std::mutex> lock(mutex_);

    global_lru_.clear();
    path_.clear();
    node_pool_.clear();
    for (auto& node : nodes_) {
      node_pool_.emplace_back(&node);
    }
    free_node_count_ = node_pool_.size();
    root_ = getNode();
  }

  const Param& param() const {
    return param_;
  }

 private:
  Result matchBFS(const std::vector<int32_t>& tokens, size_t batch_size) const;
  Result matchProb(const std::vector<int32_t>& tokens, size_t batch_size) const;

  void insert();
};

}  // namespace ngram
