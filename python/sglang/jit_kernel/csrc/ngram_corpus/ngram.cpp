#include "ngram.h"

#include "trie.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ngram {

// Implemented source-importance formula:
//   score = source_prior * (w_specificity * specificity + w_confidence * confidence)
//
// where:
// - specificity is normalized match depth / matched_len for the best anchor
// - confidence is normalized top-1 next-token mass at that anchor
// - the w_* terms are normalized from the user-provided match weights below
namespace {
double computeSourceScore(const MatchQuality& quality, double source_prior, const Param& param) {
  if (!quality.has_match) {
    return 0.0;
  }
  const double total_weight = param.match_specificity_weight + param.match_confidence_weight;
  if (total_weight <= 0.0) {
    return 0.0;
  }
  const double specificity_weight = param.match_specificity_weight / total_weight;
  const double confidence_weight = param.match_confidence_weight / total_weight;
  return source_prior * (specificity_weight * quality.specificity + confidence_weight * quality.confidence);
}

double effectiveTrieSourcePrior(double source_prior) {
  return source_prior > 0.0 ? source_prior : 1.0;
}

Result buildEmptyResult(int32_t last_token, int result_token_num) {
  return buildResultFromLeafPaths_(last_token, result_token_num, {});
}

}  // namespace
Ngram::Ngram(size_t capacity, const Param& param) : trie_capacity_(capacity), param_(param) {
  if (!(param_.max_trie_depth > 1)) {
    throw std::runtime_error(
        "param_.max_trie_depth must be greater than 1, current value: " + std::to_string(param_.max_trie_depth));
  }
  if (!(param_.min_bfs_breadth > 0)) {
    throw std::runtime_error(
        "min_bfs_breadth must be greater than 0, current value: " + std::to_string(param_.min_bfs_breadth));
  }
  if (!(param_.min_bfs_breadth <= param_.max_bfs_breadth)) {
    throw std::runtime_error(
        "min_bfs_breadth must be less than or equal to max_bfs_breadth, "
        "current min_bfs_breadth: " +
        std::to_string(param_.min_bfs_breadth) + ", max_bfs_breadth: " + std::to_string(param_.max_bfs_breadth));
  }
  if (!(param_.draft_token_num > 0)) {
    throw std::runtime_error(
        "draft_token_num must be greater than 0, current value: " + std::to_string(param_.draft_token_num));
  }
  if (param_.trie_source_prior < 0.0) {
    throw std::runtime_error(
        "trie_source_prior must be greater than or equal to 0, current value: " +
        std::to_string(param_.trie_source_prior));
  }
  if (param_.match_specificity_weight < 0.0) {
    throw std::runtime_error(
        "match_specificity_weight must be greater than or equal to 0, current value: " +
        std::to_string(param_.match_specificity_weight));
  }
  if (param_.match_confidence_weight < 0.0) {
    throw std::runtime_error(
        "match_confidence_weight must be greater than or equal to 0, current value: " +
        std::to_string(param_.match_confidence_weight));
  }
  if (param_.match_specificity_weight + param_.match_confidence_weight <= 0.0) {
    throw std::runtime_error("match quality weights must sum to a positive value");
  }
  if (!(trie_capacity_ > 0)) {
    throw std::runtime_error("trie capacity must be greater than 0, current value: " + std::to_string(trie_capacity_));
  }
  if (param_.request_trie_mode && trie_capacity_ < param_.max_trie_depth) {
    throw std::runtime_error(
        "request trie capacity must be greater than or equal to max_trie_depth, current trie capacity: " +
        std::to_string(trie_capacity_) + ", max_trie_depth: " + std::to_string(param_.max_trie_depth));
  }
  for (auto config : param_.batch_draft_token_num) {
    if (config != std::numeric_limits<decltype(config)>::max()) {
      if (!(config <= param_.draft_token_num)) {
        throw std::runtime_error(
            "batch_draft_token_num config value " + std::to_string(config) +
            " must be less than or equal to draft_token_num: " + std::to_string(param_.draft_token_num));
      }
    }
  }

  if (!param_.request_trie_mode) {
    global_trie_ = std::make_unique<Trie>(trie_capacity_, param_);
  }

  insert_worker_ = std::thread(&Ngram::insertWorker, this);
}

Ngram::~Ngram() {
  insert_queue_.close();
  if (insert_worker_.joinable()) {
    insert_worker_.join();
  }
}

void Ngram::synchronize() const {
  std::unique_lock<std::mutex> lock(mutex_);
  sync_cv_.wait(lock, [this] { return pending_count_ == 0; });
}

void Ngram::asyncInsert(std::vector<InsertWorkItem>&& items) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_count_ += items.size();
  }
  for (auto&& item : items) {
    insert_queue_.enqueue(std::move(item));
  }
}

// NOTE: staging operations (start/append/finish) are called from a background
// thread during async corpus loading. They do NOT hold mutex_ because
// staging_sam_ is disjoint from sams_ / online trie state. Only
// finishExternalCorpusLoad briefly acquires mutex_ when moving the completed
// SAM into sams_.
void Ngram::startExternalCorpusLoad() {
  if (staging_sam_) {
    throw std::runtime_error("startExternalCorpusLoad called while another load is in progress");
  }
  staging_sam_ = std::make_unique<SuffixAutomaton>();
}

void Ngram::appendExternalCorpusTokens(const std::vector<int32_t>& tokens) {
  if (!staging_sam_) {
    throw std::runtime_error("appendExternalCorpusTokens called without startExternalCorpusLoad");
  }
  staging_sam_->appendTokens(tokens);
}

void Ngram::finishExternalCorpusLoad(const std::string& corpus_id) {
  if (!staging_sam_) {
    throw std::runtime_error("finishExternalCorpusLoad called without startExternalCorpusLoad");
  }
  staging_sam_->finalize();
  if (staging_sam_->empty()) {
    staging_sam_.reset();
    throw std::runtime_error("External corpus is empty — no tokens were loaded.");
  }
  // Only lock briefly to install the completed SAM.
  std::unique_lock<std::mutex> lock(mutex_);
  if (sams_.find(corpus_id) != sams_.end()) {
    throw std::runtime_error(
        "External corpus '" + corpus_id + "' already exists. Remove it before adding a new corpus with the same id.");
  }
  sams_.emplace(corpus_id, std::move(staging_sam_));
}

void Ngram::removeExternalCorpus(const std::string& corpus_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  sams_.erase(corpus_id);
}

void Ngram::resetStagingSam() {
  // staging_sam_ is only accessed from the loading thread — no lock needed.
  staging_sam_.reset();
}

void Ngram::clearExternalCorpus() {
  std::unique_lock<std::mutex> lock(mutex_);
  sams_.clear();
  staging_sam_.reset();
}

std::vector<std::pair<std::string, int64_t>> Ngram::listExternalCorpora() const {
  std::unique_lock<std::mutex> lock(mutex_);
  std::vector<std::pair<std::string, int64_t>> entries;
  entries.reserve(sams_.size());
  for (const auto& [id, sam] : sams_) {
    entries.emplace_back(id, sam->tokenCount());
  }
  return entries;
}

void Ngram::insertWorker() {
  for (;;) {
    InsertWorkItem item;
    if (!insert_queue_.dequeue(item)) {
      break;
    }
    // Still a per-Ngram lock because of contention with batchMatch() and
    // request-trie lifecycle changes.
    std::unique_lock<std::mutex> lock(mutex_);
    auto* trie = getOrCreateTrieForInsert_(item.state_id);
    trie->insert(item.tokens.data(), item.tokens.size());
    --pending_count_;
    lock.unlock();
    sync_cv_.notify_all();
  }
}

// See C++ Core Guidelines F.7 and R.3 for why the return type is a raw pointer.
Trie* Ngram::getTrieForMatch_(int64_t state_id) {
  if (!param_.request_trie_mode) {
    return global_trie_.get();
  }

  if (state_id < 0) {
    throw std::runtime_error("request_trie_mode requires non-negative state ids");
  }
  auto iter = request_tries_.find(state_id);
  return iter == request_tries_.end() ? nullptr : iter->second.get();
}

// See C++ Core Guidelines F.7 and R.3 for why the return type is a raw pointer.
Trie* Ngram::getOrCreateTrieForInsert_(int64_t state_id) {
  if (!param_.request_trie_mode) {
    if (!global_trie_) {
      global_trie_ = std::make_unique<Trie>(trie_capacity_, param_);
    }
    return global_trie_.get();
  }

  if (state_id < 0) {
    throw std::runtime_error("request_trie_mode requires non-negative state ids");
  }
  auto& trie = request_tries_[state_id];
  if (!trie) {
    trie = std::make_unique<Trie>(trie_capacity_, param_);
  }
  return trie.get();
}

Result Ngram::batchMatch(
    const std::vector<int64_t>& state_ids,
    const std::vector<std::vector<int32_t>>& tokens,
    const std::vector<size_t>& total_lens) {
  if (state_ids.size() != tokens.size() || state_ids.size() != total_lens.size()) {
    throw std::runtime_error("batchMatch expects state_ids, tokens, and total_lens to match in size");
  }

  std::unique_lock<std::mutex> lock(mutex_);

  using TrieAnchoredBuildFn =
      Result (Trie::*)(const std::vector<std::pair<const TrieNode*, int32_t>>&, int32_t, size_t, const Param&) const;
  using SamAnchoredBuildFn =
      Result (SuffixAutomaton::*)(const std::vector<SamAnchor>&, int32_t, size_t, const Param&) const;
  TrieAnchoredBuildFn trie_anchored_build_fn;
  SamAnchoredBuildFn sam_anchored_build_fn;
  if (param_.match_type == "BFS") {
    trie_anchored_build_fn = &Trie::buildRecencyFromAnchors;
    sam_anchored_build_fn = &SuffixAutomaton::buildRecencyFromAnchors;
  } else if (param_.match_type == "PROB") {
    trie_anchored_build_fn = &Trie::buildFrequencyFromAnchors;
    sam_anchored_build_fn = &SuffixAutomaton::buildFrequencyFromAnchors;
  } else {
    throw std::runtime_error("Unknown match_type: '" + param_.match_type + "'. Must be 'BFS' or 'PROB'.");
  }

  const auto total_draft_token_num = param_.get_draft_token_num(tokens.size());
  const auto result_token_num = static_cast<int>(total_draft_token_num + 1);
  std::vector<std::pair<std::string, const SuffixAutomaton*>> ordered_sams;
  ordered_sams.reserve(sams_.size());
  for (const auto& [corpus_id, sam] : sams_) {
    ordered_sams.emplace_back(corpus_id, sam.get());
  }
  std::sort(
      ordered_sams.begin(), ordered_sams.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  Result merged;
  for (size_t i = 0; i < state_ids.size(); ++i) {
    const auto& suffix = tokens[i];
    if (suffix.empty()) {
      throw std::runtime_error("batchMatch received an empty token tail");
    }

    std::vector<std::pair<const TrieNode*, int32_t>> trie_anchors;
    MatchQuality trie_quality;
    auto* trie = getTrieForMatch_(state_ids[i]);
    if (trie != nullptr) {
      auto& state = match_state_[state_ids[i]];
      trie_anchors = trie->match(suffix.data(), suffix.size(), state, total_lens[i]);
      trie_quality = trie->summarizeMatchQuality(trie_anchors, param_);
    }

    if (ordered_sams.empty()) {
      auto res = trie != nullptr
                     ? (trie->*trie_anchored_build_fn)(trie_anchors, suffix.back(), total_draft_token_num, param_)
                     : buildEmptyResult(suffix.back(), result_token_num);
      merged.token.insert(merged.token.end(), res.token.begin(), res.token.end());
      merged.mask.insert(merged.mask.end(), res.mask.begin(), res.mask.end());
      continue;
    }

    struct RankedSourceResult {
      double score = 0.0;
      Result result;
    };

    std::vector<RankedSourceResult> source_results;
    source_results.reserve(ordered_sams.size() + 1);
    if (trie_quality.has_match) {
      source_results.push_back(
          RankedSourceResult{
              computeSourceScore(trie_quality, effectiveTrieSourcePrior(param_.trie_source_prior), param_),
              (trie->*trie_anchored_build_fn)(trie_anchors, suffix.back(), total_draft_token_num, param_)});
    }

    for (const auto& [_, sam] : ordered_sams) {
      auto anchors = sam->match(suffix.data(), suffix.size(), param_.max_trie_depth);
      auto quality = sam->summarizeMatchQuality(anchors, param_);
      const auto score = computeSourceScore(quality, 1.0, param_);
      if (score <= 0.0) {
        continue;
      }
      source_results.push_back(
          RankedSourceResult{
              score, (sam->*sam_anchored_build_fn)(anchors, suffix.back(), total_draft_token_num, param_)});
    }

    Result combined;
    if (source_results.empty()) {
      combined = trie != nullptr
                     ? (trie->*trie_anchored_build_fn)(trie_anchors, suffix.back(), total_draft_token_num, param_)
                     : buildEmptyResult(suffix.back(), result_token_num);
    } else {
      // Merge stronger sources first so the final cap keeps their branches when the tree saturates.
      std::stable_sort(source_results.begin(), source_results.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.score > rhs.score;
      });
      combined = std::move(source_results.front().result);
      for (size_t source_idx = 1; source_idx < source_results.size(); ++source_idx) {
        combined = combineRootResults_(suffix.back(), result_token_num, combined, source_results[source_idx].result);
      }
    }

    merged.token.insert(merged.token.end(), combined.token.begin(), combined.token.end());
    merged.mask.insert(merged.mask.end(), combined.mask.begin(), combined.mask.end());
  }
  return merged;
}

void Ngram::eraseMatchState(const std::vector<int64_t>& state_ids) {
  std::unique_lock<std::mutex> lock(mutex_);
  for (const auto& sid : state_ids) {
    match_state_.erase(sid);
  }
}

void Ngram::eraseRequestState(const std::vector<int64_t>& state_ids) {
  std::unique_lock<std::mutex> lock(mutex_);
  for (const auto& sid : state_ids) {
    match_state_.erase(sid);
    if (param_.request_trie_mode) {
      request_tries_.erase(sid);
    }
  }
}

void Ngram::reset() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (global_trie_) {
    global_trie_->reset();
  }
  request_tries_.clear();
  match_state_.clear();
}

}  // namespace ngram
