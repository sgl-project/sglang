#pragma once

#include "param.h"
#include "queue.h"
#include "result.h"
#include "suffix_automaton.h"
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
  std::unordered_map<std::string, std::unique_ptr<SuffixAutomaton>> sams_;
  // FIXME: single staging slot — only one corpus can be loaded at a time.
  // To support concurrent loads, move staging into a per-load local variable.
  std::unique_ptr<SuffixAutomaton> staging_sam_;
  Param param_;

  // NOTE: protects trie_, sams_, and pending_count_. staging_sam_ is NOT
  // protected by mutex_ — it is only accessed from the corpus loading thread.
  // finishExternalCorpusLoad briefly acquires mutex_ to move the completed
  // SAM into sams_.
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

  void startExternalCorpusLoad();

  void appendExternalCorpusTokens(const std::vector<int32_t>& tokens);

  void finishExternalCorpusLoad(const std::string& corpus_id);

  void removeExternalCorpus(const std::string& corpus_id);

  void clearExternalCorpus();

  std::vector<std::string> listExternalCorpora() const;

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
