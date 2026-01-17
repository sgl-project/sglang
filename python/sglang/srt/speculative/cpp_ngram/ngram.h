#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <new>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "param.h"
#include "queue.h"

namespace ngram {

struct SAMNode;

struct Transition {
  SAMNode* target;
  int32_t freq;
  std::list<int32_t>::iterator lru_it;
};

struct SAMNode {
  std::unordered_map<int32_t, Transition> next;
  std::list<SAMNode*>::const_iterator global_lru_pos;
  int32_t token;
  SAMNode* parent = nullptr;
  std::list<int32_t> lru;

  SAMNode* fail = nullptr;
  int32_t len = 0;

  struct CompareByFreq {
    const std::unordered_map<int32_t, Transition>* next_map;
    bool operator()(int32_t a, int32_t b) const {
      const auto& ta = next_map->at(a);
      const auto& tb = next_map->at(b);
      if (ta.freq != tb.freq) return ta.freq > tb.freq;
      return a < b;
    }
  };
  std::set<int32_t, CompareByFreq> sorted_children;

  SAMNode() : sorted_children(CompareByFreq{&next}) {}

  void update_freq(int32_t t, int32_t delta) {
    auto it = next.find(t);
    if (it != next.end()) {
      sorted_children.erase(t);
      it->second.freq += delta;
      sorted_children.insert(t);
      lru.splice(lru.begin(), lru, it->second.lru_it);
    }
  }
};

class Ngram {
  std::vector<SAMNode> nodes_;
  std::vector<SAMNode*> node_pool_;
  size_t free_node_count_;
  std::list<SAMNode*> global_lru_;
  SAMNode* root_;
  std::vector<SAMNode*> path_;
  Param param_;

  std::vector<std::pair<SAMNode*, int32_t>> match(const std::vector<int32_t>& tokens, size_t batch_size) const;

  void squeeze(size_t count);

  SAMNode* getNode() {
    auto node = node_pool_[--free_node_count_];
    node->~SAMNode();
    new (node) SAMNode();
    return node;
  }

  mutable std::mutex mutex_;
  bool quit_flag_;
  utils::Queue<std::vector<int32_t>> insert_queue_;
  std::thread insert_worker_;

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
    root_->len = 0;
    root_->fail = nullptr;
  }

  const Param& param() const {
    return param_;
  }

 private:
  Result matchBFS(const std::vector<int32_t>& tokens, size_t batch_size) const;
  Result matchProb(const std::vector<int32_t>& tokens, size_t batch_size) const;

  void insert();
  void extend(int32_t token, SAMNode*& last);
};

}  // namespace ngram
