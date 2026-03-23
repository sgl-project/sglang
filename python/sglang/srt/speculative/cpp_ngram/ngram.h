#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "param.h"
#include "queue.h"
#include "result.h"
#include "trie.h"

namespace ngram {

class Ngram {
  std::unique_ptr<Trie> trie_;
  Param param_;

  mutable std::mutex mutex_;
  bool quit_flag_ = false;
  utils::Queue<std::vector<int32_t>> insert_queue_;
  std::thread insert_worker_;

 public:
  Ngram(size_t capacity, const Param& param);
  ~Ngram();

  void synchronize() const;

  void asyncInsert(std::vector<std::vector<int32_t>>&& tokens);

  Result batchMatch(const std::vector<std::vector<int32_t>>& tokens) const;

  void reset() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (trie_) {
      trie_->reset();
    }
  }

  const Param& param() const {
    return param_;
  }

 private:
  void insertWorker();
};

}  // namespace ngram
