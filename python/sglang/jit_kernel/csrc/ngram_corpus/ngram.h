#pragma once

#include "param.h"
#include "queue.h"
#include "result.h"
#include "trie.h"
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ngram {

class Ngram {
  std::unique_ptr<Trie> trie_;
  Param param_;

  // NOTE: protects trie_ and pending_count_. Ensures batchMatch never reads
  // trie_ while insertWorker is writing. After synchronize(), no pending
  // inserts remain so mutex_ contention is effectively zero.
  mutable std::mutex mutex_;
  mutable std::condition_variable sync_cv_;
  // NOTE: tracks inserts from enqueue through trie_->insert() completion,
  // not just queue occupancy. A dequeued item may still be mid-insert.
  size_t pending_count_ = 0;
  utils::Queue<std::vector<int32_t>> insert_queue_;
  std::thread insert_worker_;
  std::unordered_map<int64_t, MatchState> match_state_;

 public:
  Ngram(size_t capacity, const Param& param);
  ~Ngram();

  void synchronize() const;

  void asyncInsert(std::vector<std::vector<int32_t>>&& tokens);

  Result batchMatch(const std::vector<std::vector<int32_t>>& tokens);

  Result batchMatch(
      const std::vector<int64_t>& state_ids,
      const std::vector<std::vector<int32_t>>& tokens,
      const std::vector<size_t>& total_lens);

  void eraseMatchState(const std::vector<int64_t>& state_ids);

  void reset() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (trie_) {
      trie_->reset();
    }
    match_state_.clear();
  }

  const Param& param() const {
    return param_;
  }

 private:
  void insertWorker();
};

}  // namespace ngram
