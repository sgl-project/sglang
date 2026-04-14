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

struct InsertWorkItem {
  int64_t state_id = -1;
  std::vector<int32_t> tokens;
};

class Ngram {
  std::unique_ptr<TrieArena> trie_arena_;
  std::unique_ptr<Trie> global_trie_;
  std::unordered_map<int64_t, std::unique_ptr<Trie>> request_tries_;
  std::unordered_map<std::string, std::unique_ptr<SuffixAutomaton>> sams_;
  // FIXME: single staging slot — only one corpus can be loaded at a time.
  // To support concurrent loads, move staging into a per-load local variable.
  std::unique_ptr<SuffixAutomaton> staging_sam_;
  Param param_;

  // NOTE: protects trie_arena_, global_trie_, request_tries_, sams_, and
  // pending_count_. staging_sam_ is NOT
  // protected by mutex_ — it is only accessed from the corpus loading thread.
  // finishExternalCorpusLoad briefly acquires mutex_ to move the completed
  // SAM into sams_.
  mutable std::mutex mutex_;
  mutable std::condition_variable sync_cv_;
  // NOTE: tracks inserts from enqueue through trie insert completion,
  // not just queue occupancy. A dequeued item may still be mid-insert.
  size_t pending_count_ = 0;
  utils::Queue<InsertWorkItem> insert_queue_;
  std::thread insert_worker_;
  std::unordered_map<int64_t, MatchState> match_state_;

 public:
  Ngram(size_t capacity, const Param& param);
  ~Ngram();

  void synchronize() const;

  void asyncInsert(std::vector<InsertWorkItem>&& items);

  void startExternalCorpusLoad();

  void appendExternalCorpusTokens(const std::vector<int32_t>& tokens);

  // Publishes the staged corpus. Duplicate corpus_id is rejected.
  void finishExternalCorpusLoad(const std::string& corpus_id);

  void removeExternalCorpus(const std::string& corpus_id);

  void resetStagingSam();

  void clearExternalCorpus();

  std::vector<std::pair<std::string, int64_t>> listExternalCorpora() const;

  Result batchMatch(
      const std::vector<int64_t>& state_ids,
      const std::vector<std::vector<int32_t>>& tokens,
      const std::vector<size_t>& total_lens);

  void eraseMatchState(const std::vector<int64_t>& state_ids);

  void eraseRequestState(const std::vector<int64_t>& state_ids);

  // Resets the online trie and match state but preserves external corpora
  // (sams_). External corpora are user-managed via add/remove APIs and
  // should not be affected by cache flushes.
  void reset();

  const Param& param() const {
    return param_;
  }

 private:
  void insertWorker();
  Trie* getOrCreateTrie_(int64_t state_id);
};

}  // namespace ngram
