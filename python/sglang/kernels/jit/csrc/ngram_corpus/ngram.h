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

namespace sglang {

namespace ngram {

class Ngram {
  struct ExternalCorpus {
    std::string id;
    std::unique_ptr<SuffixAutomaton> sam;
  };

  std::unique_ptr<Trie> trie_;
  // SAM ownership is keyed by the request-facing handle. A single secondary
  // id index supports duplicate checks and removal without raw-pointer indexes.
  std::unordered_map<int64_t, ExternalCorpus> sams_;
  std::unordered_map<std::string, int64_t> sam_handle_by_id_;
  // A finalized staged SAM remains outside every query index until commit.
  // ExternalCorpusManager serializes loads, so one staging slot is sufficient.
  std::unique_ptr<SuffixAutomaton> staging_sam_;
  std::string staging_corpus_id_;
  int64_t staging_corpus_handle_ = 0;
  Param param_;

  // NOTE: protects trie_, published SAM indexes, and pending_count_. Staging
  // is built by one loading thread and is only finalized/committed/cancelled
  // after that thread has stopped.
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

  // Finalizes the staged corpus but does not make it queryable.
  void finishExternalCorpusLoad(const std::string& corpus_id, int64_t corpus_handle);

  // Atomically installs the finalized staged corpus into all query indexes.
  void commitExternalCorpusLoad(const std::string& corpus_id);

  void removeExternalCorpus(const std::string& corpus_id);

  void resetStagingSam();

  void clearExternalCorpus();

  std::vector<std::pair<std::string, int64_t>> listExternalCorpora() const;

  Result batchMatch(
      const std::vector<int64_t>& state_ids,
      const std::vector<std::vector<int32_t>>& tokens,
      const std::vector<size_t>& total_lens,
      const std::vector<int64_t>& corpus_handles = {});

  void eraseMatchState(const std::vector<int64_t>& state_ids);

  // Resets the online trie and match state but preserves external corpora
  // (sams_). External corpora are user-managed via add/remove APIs and
  // should not be affected by cache flushes.
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

}  // namespace sglang
