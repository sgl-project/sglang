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

struct PrecomputeDraftsStats {
  int64_t num_paths = 0;
  int64_t num_phase2_contexts = 0;
  int64_t num_cache_entries = 0;
};

struct SelectPrecomputedDraftsResult {
  Result result;
  std::vector<uint8_t> bonus_prediction_hit;
  std::vector<uint8_t> precomputed_cache_hit;
  int64_t bonus_prediction_hit_ct = 0;
  int64_t bonus_prediction_total_ct = 0;
  int64_t precomputed_cache_hit_ct = 0;
  int64_t precomputed_cache_total_ct = 0;
};

class Ngram {
  struct PathKey {
    int64_t state_id = 0;
    std::vector<int32_t> path_cols;

    bool operator==(const PathKey& other) const {
      return state_id == other.state_id && path_cols == other.path_cols;
    }
  };

  struct PathBonusKey {
    int64_t state_id = 0;
    std::vector<int32_t> path_cols;
    int32_t bonus_token = 0;

    bool operator==(const PathBonusKey& other) const {
      return state_id == other.state_id && bonus_token == other.bonus_token && path_cols == other.path_cols;
    }
  };

  struct PathKeyHash {
    size_t operator()(const PathKey& key) const;
  };

  struct PathBonusKeyHash {
    size_t operator()(const PathBonusKey& key) const;
  };

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
  std::unordered_map<PathBonusKey, Result, PathBonusKeyHash> precomputed_cache_;
  std::unordered_map<PathKey, std::vector<int32_t>, PathKeyHash> precomputed_bonus_candidates_;

 public:
  Ngram(size_t capacity, const Param& param);
  ~Ngram();

  void synchronize() const;

  void asyncInsert(std::vector<std::vector<int32_t>>&& tokens);

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

  PrecomputeDraftsStats precomputeDrafts(
      const std::vector<int64_t>& state_ids,
      const std::vector<std::vector<int32_t>>& base_tokens,
      const std::vector<size_t>& base_total_lens,
      const std::vector<int32_t>& draft_tokens,
      const std::vector<uint8_t>& tree_mask,
      size_t bonus_topk,
      size_t max_trie_depth);

  SelectPrecomputedDraftsResult selectPrecomputedDrafts(
      const std::vector<int64_t>& state_ids,
      const std::vector<int32_t>& accept_tokens,
      const std::vector<int64_t>& accept_lens,
      const std::vector<int64_t>& accept_index,
      const std::vector<std::vector<int32_t>>& fallback_tokens,
      const std::vector<size_t>& fallback_total_lens);

  std::vector<int32_t> precomputedRootBonusTokens(const std::vector<int64_t>& state_ids) const;

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
    precomputed_cache_.clear();
    precomputed_bonus_candidates_.clear();
  }

  const Param& param() const {
    return param_;
  }

 private:
  void insertWorker();
  Result buildMatchUnlocked(
      const std::vector<int32_t>& suffix, size_t total_len, MatchState& state, size_t batch_size_for_budget) const;
  std::vector<int32_t> buildRootCandidatesUnlocked(
      const std::vector<int32_t>& suffix, size_t total_len, MatchState& state, size_t max_candidates) const;
};

}  // namespace ngram
