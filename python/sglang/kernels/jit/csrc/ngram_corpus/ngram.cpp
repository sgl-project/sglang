#include "ngram.h"

#include "trie.h"
#include <limits>
#include <stdexcept>
#include <string>

namespace sglang {

namespace ngram {

namespace {
constexpr int64_t kAllCorporaHandle = -1;
}

Ngram::Ngram(size_t capacity, const Param& param) : param_(param) {
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
  for (auto config : param_.batch_draft_token_num) {
    if (config != std::numeric_limits<decltype(config)>::max()) {
      if (!(config <= param_.draft_token_num)) {
        throw std::runtime_error(
            "batch_draft_token_num config value " + std::to_string(config) +
            " must be less than or equal to draft_token_num: " + std::to_string(param_.draft_token_num));
      }
    }
  }

  trie_ = std::make_unique<Trie>(capacity, param_);

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

void Ngram::asyncInsert(std::vector<std::vector<int32_t>>&& tokens) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_count_ += tokens.size();
  }
  for (auto&& token : tokens) {
    insert_queue_.enqueue(std::move(token));
  }
}

// NOTE: staging operations (start/append/finish) are called from one background
// thread during async corpus loading. The staged SAM is disjoint from every
// query-visible index until commitExternalCorpusLoad.
void Ngram::startExternalCorpusLoad() {
  if (staging_sam_) {
    throw std::runtime_error("startExternalCorpusLoad called while another load is in progress");
  }
  staging_corpus_id_.clear();
  staging_corpus_handle_ = 0;
  staging_sam_ = std::make_unique<SuffixAutomaton>();
}

void Ngram::appendExternalCorpusTokens(const std::vector<int32_t>& tokens) {
  if (!staging_sam_) {
    throw std::runtime_error("appendExternalCorpusTokens called without startExternalCorpusLoad");
  }
  if (!staging_corpus_id_.empty()) {
    throw std::runtime_error("appendExternalCorpusTokens called after finishExternalCorpusLoad");
  }
  staging_sam_->appendTokens(tokens);
}

void Ngram::finishExternalCorpusLoad(const std::string& corpus_id, int64_t corpus_handle) {
  if (!staging_sam_) {
    throw std::runtime_error("finishExternalCorpusLoad called without startExternalCorpusLoad");
  }
  if (corpus_id.empty()) {
    throw std::runtime_error("External corpus id must be non-empty");
  }
  if (corpus_id.find_first_of("\t\n\r") != std::string::npos) {
    throw std::runtime_error("External corpus id must not contain tabs or line breaks");
  }
  if (corpus_handle <= 0) {
    throw std::runtime_error("External corpus handle must be a positive int64");
  }
  staging_sam_->finalize();
  if (staging_sam_->empty()) {
    resetStagingSam();
    throw std::runtime_error("External corpus is empty — no tokens were loaded.");
  }
  // Validate against the current published indexes, but deliberately keep the
  // finalized SAM in staging. Distributed callers publish only after every
  // participating rank reports a successful build.
  std::unique_lock<std::mutex> lock(mutex_);
  if (sam_handle_by_id_.find(corpus_id) != sam_handle_by_id_.end()) {
    throw std::runtime_error(
        "External corpus '" + corpus_id + "' already exists. Remove it before adding a new corpus with the same id.");
  }
  if (sams_.find(corpus_handle) != sams_.end()) {
    throw std::runtime_error("External corpus handle " + std::to_string(corpus_handle) + " already exists");
  }
  staging_corpus_id_ = corpus_id;
  staging_corpus_handle_ = corpus_handle;
}

void Ngram::commitExternalCorpusLoad(const std::string& corpus_id) {
  if (!staging_sam_ || staging_corpus_id_.empty()) {
    throw std::runtime_error("commitExternalCorpusLoad called without a finalized staged corpus");
  }
  if (corpus_id != staging_corpus_id_) {
    throw std::runtime_error(
        "commitExternalCorpusLoad corpus id mismatch: expected '" + staging_corpus_id_ + "', got '" + corpus_id +
        "'");
  }

  std::unique_lock<std::mutex> lock(mutex_);
  const int64_t corpus_handle = staging_corpus_handle_;
  sams_.emplace(corpus_handle, ExternalCorpus{corpus_id, std::move(staging_sam_)});
  sam_handle_by_id_.emplace(corpus_id, corpus_handle);
  staging_corpus_id_.clear();
  staging_corpus_handle_ = 0;
}

void Ngram::removeExternalCorpus(const std::string& corpus_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto handle_it = sam_handle_by_id_.find(corpus_id);
  if (handle_it != sam_handle_by_id_.end()) {
    sams_.erase(handle_it->second);
    sam_handle_by_id_.erase(handle_it);
  }
}

void Ngram::resetStagingSam() {
  staging_sam_.reset();
  staging_corpus_id_.clear();
  staging_corpus_handle_ = 0;
}

void Ngram::clearExternalCorpus() {
  std::unique_lock<std::mutex> lock(mutex_);
  sam_handle_by_id_.clear();
  sams_.clear();
  resetStagingSam();
}

std::vector<std::pair<std::string, int64_t>> Ngram::listExternalCorpora() const {
  std::unique_lock<std::mutex> lock(mutex_);
  std::vector<std::pair<std::string, int64_t>> entries;
  entries.reserve(sams_.size());
  for (const auto& [_, corpus] : sams_) {
    entries.emplace_back(corpus.id, corpus.sam->tokenCount());
  }
  return entries;
}

void Ngram::insertWorker() {
  for (;;) {
    std::vector<int32_t> data;
    if (!insert_queue_.dequeue(data)) {
      break;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    trie_->insert(data.data(), data.size());
    --pending_count_;
    lock.unlock();
    sync_cv_.notify_all();
  }
}

Result Ngram::batchMatch(
    const std::vector<int64_t>& state_ids,
    const std::vector<std::vector<int32_t>>& tokens,
    const std::vector<size_t>& total_lens,
    const std::vector<int64_t>& corpus_handles) {
  if (state_ids.size() != tokens.size() || state_ids.size() != total_lens.size()) {
    throw std::runtime_error("batchMatch expects state_ids, tokens, and total_lens to match in size");
  }
  // An empty vector preserves the legacy all-SAM behavior. In an aligned
  // per-request vector, -1 selects all SAMs, 0 (or any unknown handle) selects
  // no external SAM, and a known positive handle selects exactly one SAM.
  const bool per_request = !corpus_handles.empty();
  if (per_request && corpus_handles.size() != state_ids.size()) {
    throw std::runtime_error("batchMatch expects corpus_handles to match state_ids in size");
  }

  std::unique_lock<std::mutex> lock(mutex_);

  using TrieResultBuildFn =
      Result (Trie::*)(const int32_t*, size_t, int32_t, size_t, const Param&, MatchState&, size_t) const;
  using SamResultBuildFn = Result (SuffixAutomaton::*)(const int32_t*, size_t, int32_t, size_t, const Param&) const;
  TrieResultBuildFn trie_result_build_fn;
  SamResultBuildFn sam_result_build_fn;
  if (param_.match_type == "BFS") {
    trie_result_build_fn = &Trie::buildRecency;
    sam_result_build_fn = &SuffixAutomaton::buildRecency;
  } else if (param_.match_type == "PROB") {
    trie_result_build_fn = &Trie::buildFrequency;
    sam_result_build_fn = &SuffixAutomaton::buildFrequency;
  } else {
    throw std::runtime_error("Unknown match_type: '" + param_.match_type + "'. Must be 'BFS' or 'PROB'.");
  }

  // All budget values are loop-invariant (mutex_ held, sams_ won't change).
  const size_t num_sams = sams_.size();
  const auto total_draft_token_num = param_.get_draft_token_num(tokens.size());
  const size_t total_sam_budget =
      num_sams > 0 ? std::min(param_.external_sam_budget, total_draft_token_num) : size_t{0};
  const size_t per_sam_budget = num_sams > 0 ? total_sam_budget / num_sams : size_t{0};
  const size_t trie_budget = total_draft_token_num - (per_sam_budget * num_sams);

  Result merged;
  for (size_t i = 0; i < state_ids.size(); ++i) {
    const auto& suffix = tokens[i];
    if (suffix.empty()) {
      throw std::runtime_error("batchMatch received an empty token tail");
    }

    auto& state = match_state_[state_ids[i]];

    const bool use_all_sams = !per_request || corpus_handles[i] == kAllCorporaHandle;
    if (!use_all_sams) {
      const SuffixAutomaton* sam = nullptr;
      const int64_t handle = corpus_handles[i];
      if (handle > 0) {
        auto it = sams_.find(handle);
        if (it != sams_.end()) {
          sam = it->second.sam.get();
        }
      }
      const size_t sam_budget =
          sam ? std::min(param_.external_sam_budget, total_draft_token_num) : size_t{0};
      const size_t req_trie_budget = total_draft_token_num - sam_budget;
      Result res = (trie_.get()->*trie_result_build_fn)(
          suffix.data(), suffix.size(), suffix.back(), req_trie_budget, param_, state, total_lens[i]);
      if (sam_budget > 0) {
        auto sam_res =
            (sam->*sam_result_build_fn)(suffix.data(), suffix.size(), suffix.back(), sam_budget, param_);
        res = combineRootResults_(suffix.back(), static_cast<int>(total_draft_token_num + 1), res, sam_res);
      }
      merged.token.insert(merged.token.end(), res.token.begin(), res.token.end());
      merged.mask.insert(merged.mask.end(), res.mask.begin(), res.mask.end());
      continue;
    }

    if (total_sam_budget == 0 || per_sam_budget == 0) {
      auto res = (trie_.get()->*trie_result_build_fn)(
          suffix.data(), suffix.size(), suffix.back(), total_draft_token_num, param_, state, total_lens[i]);
      merged.token.insert(merged.token.end(), res.token.begin(), res.token.end());
      merged.mask.insert(merged.mask.end(), res.mask.begin(), res.mask.end());
      continue;
    }

    auto combined = (trie_.get()->*trie_result_build_fn)(
        suffix.data(), suffix.size(), suffix.back(), trie_budget, param_, state, total_lens[i]);

    for (const auto& [_, corpus] : sams_) {
      const auto& sam = corpus.sam;
      auto sam_res =
          (sam.get()->*sam_result_build_fn)(suffix.data(), suffix.size(), suffix.back(), per_sam_budget, param_);
      combined = combineRootResults_(suffix.back(), static_cast<int>(total_draft_token_num + 1), combined, sam_res);
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

}  // namespace ngram

}  // namespace sglang
