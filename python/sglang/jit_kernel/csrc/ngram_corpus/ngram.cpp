#include "ngram.h"

#include "trie.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace ngram {

namespace {

size_t hashCombine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

std::vector<int32_t>
appendAndTrim(const std::vector<int32_t>& base, const std::vector<int32_t>& extra, size_t max_len) {
  std::vector<int32_t> out;
  out.reserve(std::min(max_len, base.size() + extra.size()));
  const auto total_len = base.size() + extra.size();
  const auto skip = total_len > max_len ? total_len - max_len : 0;
  for (size_t i = skip; i < base.size(); ++i) {
    out.emplace_back(base[i]);
  }
  const auto extra_skip = skip > base.size() ? skip - base.size() : 0;
  for (size_t i = extra_skip; i < extra.size(); ++i) {
    out.emplace_back(extra[i]);
  }
  return out;
}

std::vector<int32_t> appendTokenAndTrim(const std::vector<int32_t>& base, int32_t token, size_t max_len) {
  std::vector<int32_t> out;
  out.reserve(std::min(max_len, base.size() + size_t{1}));
  const auto total_len = base.size() + size_t{1};
  const auto skip = total_len > max_len ? total_len - max_len : 0;
  for (size_t i = skip; i < base.size(); ++i) {
    out.emplace_back(base[i]);
  }
  if (skip <= base.size()) {
    out.emplace_back(token);
  }
  return out;
}

std::vector<int32_t> directChildTokens(
    const std::vector<int32_t>& draft_tokens,
    const std::vector<uint8_t>& tree_mask,
    size_t row_offset,
    size_t d,
    size_t parent) {
  std::vector<int32_t> child_tokens;
  std::unordered_set<int32_t> seen_tokens;
  for (size_t child = parent + 1; child < d; ++child) {
    const auto child_row = row_offset + child * d;
    if (tree_mask[child_row + child] == 0 || tree_mask[child_row + parent] == 0) {
      continue;
    }
    int last_ancestor = -1;
    for (size_t col = 0; col < child; ++col) {
      if (tree_mask[child_row + col] != 0) {
        last_ancestor = static_cast<int>(col);
      }
    }
    if (last_ancestor != static_cast<int>(parent)) {
      continue;
    }
    const auto token = draft_tokens[child];
    if (seen_tokens.insert(token).second) {
      child_tokens.emplace_back(token);
    }
  }
  return child_tokens;
}

bool containsToken(const std::vector<int32_t>& tokens, int32_t token) {
  return std::find(tokens.begin(), tokens.end(), token) != tokens.end();
}

size_t wideBonusTokenCount(size_t draft_token_num, double wide_bonus_ratio) {
  if (draft_token_num == 0) {
    return 0;
  }
  if (!std::isfinite(wide_bonus_ratio)) {
    wide_bonus_ratio = 0.0;
  }
  const auto ratio = std::clamp(wide_bonus_ratio, 0.0, 1.0);
  return std::min(draft_token_num, static_cast<size_t>(std::ceil(static_cast<double>(draft_token_num) * ratio)));
}

}  // namespace

size_t Ngram::PathKeyHash::operator()(const PathKey& key) const {
  size_t seed = std::hash<int64_t>{}(key.state_id);
  for (const auto col : key.path_cols) {
    seed = hashCombine(seed, std::hash<int32_t>{}(col));
  }
  return seed;
}

size_t Ngram::PathBonusKeyHash::operator()(const PathBonusKey& key) const {
  size_t seed = std::hash<int64_t>{}(key.state_id);
  seed = hashCombine(seed, std::hash<int32_t>{}(key.bonus_token));
  for (const auto col : key.path_cols) {
    seed = hashCombine(seed, std::hash<int32_t>{}(col));
  }
  return seed;
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

// NOTE: staging operations (start/append/finish) are called from a background
// thread during async corpus loading. They do NOT hold mutex_ because
// staging_sam_ is disjoint from sams_ / trie_. Only finishExternalCorpusLoad
// briefly acquires mutex_ when moving the completed SAM into sams_.
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

Result Ngram::buildMatchUnlocked(
    const std::vector<int32_t>& suffix, size_t total_len, MatchState& state, size_t batch_size_for_budget) const {
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

  const size_t num_sams = sams_.size();
  const auto total_draft_token_num = param_.get_draft_token_num(batch_size_for_budget);
  const size_t total_sam_budget =
      num_sams > 0 ? std::min(param_.external_sam_budget, total_draft_token_num) : size_t{0};
  const size_t per_sam_budget = num_sams > 0 ? total_sam_budget / num_sams : size_t{0};
  const size_t trie_budget = total_draft_token_num - (per_sam_budget * num_sams);

  if (total_sam_budget == 0 || per_sam_budget == 0) {
    return (trie_.get()->*trie_result_build_fn)(
        suffix.data(), suffix.size(), suffix.back(), total_draft_token_num, param_, state, total_len);
  }

  auto combined = (trie_.get()->*trie_result_build_fn)(
      suffix.data(), suffix.size(), suffix.back(), trie_budget, param_, state, total_len);

  for (const auto& [_, sam] : sams_) {
    auto sam_res =
        (sam.get()->*sam_result_build_fn)(suffix.data(), suffix.size(), suffix.back(), per_sam_budget, param_);
    combined = combineRootResults_(suffix.back(), static_cast<int>(total_draft_token_num + 1), combined, sam_res);
  }
  return combined;
}

std::vector<int32_t> Ngram::buildRootCandidatesUnlocked(
    const std::vector<int32_t>& suffix, size_t total_len, MatchState& state, size_t max_candidates) const {
  using TrieCandidateBuildFn =
      std::vector<int32_t> (Trie::*)(const int32_t*, size_t, size_t, const Param&, MatchState&, size_t) const;
  using SamCandidateBuildFn =
      std::vector<int32_t> (SuffixAutomaton::*)(const int32_t*, size_t, size_t, const Param&) const;
  TrieCandidateBuildFn trie_candidate_build_fn;
  SamCandidateBuildFn sam_candidate_build_fn;
  if (param_.match_type == "BFS") {
    trie_candidate_build_fn = &Trie::getRootCandidatesRecency;
    sam_candidate_build_fn = &SuffixAutomaton::getRootCandidatesRecency;
  } else if (param_.match_type == "PROB") {
    trie_candidate_build_fn = &Trie::getRootCandidatesFrequency;
    sam_candidate_build_fn = &SuffixAutomaton::getRootCandidatesFrequency;
  } else {
    throw std::runtime_error("Unknown match_type: '" + param_.match_type + "'. Must be 'BFS' or 'PROB'.");
  }

  std::vector<int32_t> merged_candidates;
  merged_candidates.reserve(max_candidates);
  std::unordered_set<int32_t> seen_tokens;
  auto append_candidates = [&merged_candidates, &seen_tokens, max_candidates](const std::vector<int32_t>& src) {
    for (const auto token : src) {
      if (merged_candidates.size() >= max_candidates) {
        return;
      }
      if (seen_tokens.insert(token).second) {
        merged_candidates.emplace_back(token);
      }
    }
  };

  auto trie_candidates =
      (trie_.get()->*trie_candidate_build_fn)(suffix.data(), suffix.size(), max_candidates, param_, state, total_len);
  append_candidates(trie_candidates);

  for (const auto& [_, sam] : sams_) {
    if (merged_candidates.size() >= max_candidates) {
      break;
    }
    auto sam_candidates = (sam.get()->*sam_candidate_build_fn)(suffix.data(), suffix.size(), max_candidates, param_);
    append_candidates(sam_candidates);
  }
  return merged_candidates;
}

Result Ngram::batchMatch(
    const std::vector<int64_t>& state_ids,
    const std::vector<std::vector<int32_t>>& tokens,
    const std::vector<size_t>& total_lens) {
  if (state_ids.size() != tokens.size() || state_ids.size() != total_lens.size()) {
    throw std::runtime_error("batchMatch expects state_ids, tokens, and total_lens to match in size");
  }

  std::unique_lock<std::mutex> lock(mutex_);

  Result merged;
  for (size_t i = 0; i < state_ids.size(); ++i) {
    const auto& suffix = tokens[i];
    if (suffix.empty()) {
      throw std::runtime_error("batchMatch received an empty token tail");
    }

    auto& state = match_state_[state_ids[i]];
    auto res = buildMatchUnlocked(suffix, total_lens[i], state, tokens.size());
    merged.token.insert(merged.token.end(), res.token.begin(), res.token.end());
    merged.mask.insert(merged.mask.end(), res.mask.begin(), res.mask.end());
  }
  return merged;
}

PrecomputeDraftsStats Ngram::precomputeDrafts(
    const std::vector<int64_t>& state_ids,
    const std::vector<std::vector<int32_t>>& base_tokens,
    const std::vector<size_t>& base_total_lens,
    const std::vector<int32_t>& draft_tokens,
    const std::vector<uint8_t>& tree_mask,
    size_t bonus_topk,
    size_t max_trie_depth,
    double wide_bonus_ratio) {
  if (state_ids.size() != base_tokens.size() || state_ids.size() != base_total_lens.size()) {
    throw std::runtime_error("precomputeDrafts expects state_ids, base_tokens, and base_total_lens to match in size");
  }
  const size_t bs = state_ids.size();
  const size_t d = param_.draft_token_num;
  if (draft_tokens.size() != bs * d) {
    throw std::runtime_error("precomputeDrafts received draft_tokens with unexpected size");
  }
  if (tree_mask.size() != bs * d * d) {
    throw std::runtime_error("precomputeDrafts received tree_mask with unexpected size");
  }

  std::unique_lock<std::mutex> lock(mutex_);
  precomputed_cache_.clear();
  precomputed_bonus_candidates_.clear();
  const auto wide_bonus_token_count = wideBonusTokenCount(d, wide_bonus_ratio);
  const auto max_phase2_entries = bs * (wide_bonus_token_count * bonus_topk + (d - wide_bonus_token_count));
  precomputed_cache_.reserve(max_phase2_entries);
  precomputed_bonus_candidates_.reserve(bs);

  struct Phase2Entry {
    int64_t state_id;
    std::vector<int32_t> path_cols;
    int32_t bonus_token;
    std::vector<int32_t> tokens;
    size_t total_len;
    MatchState state;
  };
  std::vector<Phase2Entry> phase2_entries;
  phase2_entries.reserve(max_phase2_entries);
  PrecomputeDraftsStats stats;

  for (size_t req_idx = 0; req_idx < bs; ++req_idx) {
    const auto state_id = state_ids[req_idx];
    const auto row_token_offset = req_idx * d;
    const auto row_mask_offset = req_idx * d * d;
    std::vector<int32_t> req_draft_tokens(d);
    std::copy_n(draft_tokens.begin() + row_token_offset, d, req_draft_tokens.begin());

    for (size_t node = 0; node < d; ++node) {
      const auto node_row = row_mask_offset + node * d;
      if (tree_mask[node_row + node] == 0) {
        continue;
      }

      std::vector<int32_t> path_cols;
      path_cols.reserve(d);
      for (size_t col = 0; col < d; ++col) {
        if (tree_mask[node_row + col] != 0) {
          path_cols.emplace_back(static_cast<int32_t>(col));
        }
      }
      if (path_cols.empty() || path_cols.front() != 0) {
        continue;
      }
      ++stats.num_paths;

      std::vector<int32_t> path_draft_tokens;
      path_draft_tokens.reserve(path_cols.size());
      for (const auto col : path_cols) {
        if (col != 0) {
          path_draft_tokens.emplace_back(req_draft_tokens[col]);
        }
      }

      auto current_child_tokens = directChildTokens(req_draft_tokens, tree_mask, row_mask_offset, d, node);
      std::unordered_set<int32_t> seen_bonus;
      std::vector<int32_t> path_bonus_candidates;
      const auto path_bonus_topk = node < wide_bonus_token_count ? bonus_topk : std::min<size_t>(bonus_topk, 1);
      if (path_bonus_topk == 0) {
        continue;
      }

      const auto path_max_bonus_candidates = path_bonus_topk + current_child_tokens.size();
      auto check_tokens = appendAndTrim(base_tokens[req_idx], path_draft_tokens, max_trie_depth);
      const auto check_total_len = base_total_lens[req_idx] + path_draft_tokens.size();
      MatchState path_state;
      auto bonus_candidates =
          buildRootCandidatesUnlocked(check_tokens, check_total_len, path_state, path_max_bonus_candidates);

      for (const auto bonus_token : bonus_candidates) {
        if (containsToken(current_child_tokens, bonus_token)) {
          continue;
        }
        if (!seen_bonus.insert(bonus_token).second) {
          continue;
        }
        path_bonus_candidates.emplace_back(bonus_token);
        auto draft_check_tokens = appendTokenAndTrim(check_tokens, bonus_token, max_trie_depth);
        phase2_entries.emplace_back(
            Phase2Entry{
                state_id, path_cols, bonus_token, std::move(draft_check_tokens), check_total_len + 1, path_state});
        if (path_bonus_candidates.size() >= path_bonus_topk) {
          break;
        }
      }

      if (!path_bonus_candidates.empty() && path_cols.size() == 1) {
        precomputed_bonus_candidates_[PathKey{state_id, path_cols}] = std::move(path_bonus_candidates);
      }
    }
  }

  stats.num_phase2_contexts = static_cast<int64_t>(phase2_entries.size());
  const auto phase2_batch_size = std::max<size_t>(phase2_entries.size(), 1);
  for (auto& entry : phase2_entries) {
    auto res = buildMatchUnlocked(entry.tokens, entry.total_len, entry.state, phase2_batch_size);
    PathBonusKey key{entry.state_id, entry.path_cols, entry.bonus_token};
    if (precomputed_cache_.find(key) == precomputed_cache_.end()) {
      precomputed_cache_.emplace(std::move(key), std::move(res));
    }
  }
  stats.num_cache_entries = static_cast<int64_t>(precomputed_cache_.size());
  return stats;
}

SelectPrecomputedDraftsResult Ngram::selectPrecomputedDrafts(
    const std::vector<int64_t>& state_ids,
    const std::vector<int32_t>& accept_tokens,
    const std::vector<int64_t>& accept_lens,
    const std::vector<int64_t>& accept_index,
    const std::vector<std::vector<int32_t>>& fallback_tokens,
    const std::vector<size_t>& fallback_total_lens) {
  if (state_ids.size() != accept_lens.size() || state_ids.size() != fallback_tokens.size() ||
      state_ids.size() != fallback_total_lens.size()) {
    throw std::runtime_error(
        "selectPrecomputedDrafts expects state_ids, accept_lens, fallback_tokens, and fallback_total_lens to match");
  }
  const size_t bs = state_ids.size();
  const size_t d = param_.draft_token_num;
  if (accept_tokens.size() != bs * d) {
    throw std::runtime_error("selectPrecomputedDrafts received accept_tokens with unexpected size");
  }
  if (accept_index.size() != bs * d) {
    throw std::runtime_error("selectPrecomputedDrafts received accept_index with unexpected size");
  }

  std::unique_lock<std::mutex> lock(mutex_);
  SelectPrecomputedDraftsResult out;
  out.result.token.assign(bs * d, 0);
  out.result.mask.assign(bs * d * d, 0);
  out.bonus_prediction_hit.assign(bs, 0);
  out.precomputed_cache_hit.assign(bs, 0);
  out.bonus_prediction_total_ct = static_cast<int64_t>(bs);
  out.precomputed_cache_total_ct = static_cast<int64_t>(bs);

  std::vector<size_t> miss_indices;
  miss_indices.reserve(bs);
  for (size_t i = 0; i < bs; ++i) {
    const auto accept_len = accept_lens[i];
    if (accept_len <= 0 || accept_len > static_cast<int64_t>(d)) {
      miss_indices.emplace_back(i);
      continue;
    }

    std::vector<int32_t> path_cols;
    const auto num_path_slots = std::min<size_t>(static_cast<size_t>(accept_len), d);
    path_cols.reserve(num_path_slots);
    bool valid_path = true;
    for (size_t j = 0; j < num_path_slots; ++j) {
      const auto global_idx = accept_index[i * d + j];
      if (global_idx != -1) {
        const auto local_idx = global_idx - static_cast<int64_t>(i * d);
        if (local_idx < 0 || local_idx >= static_cast<int64_t>(d)) {
          valid_path = false;
          break;
        }
        path_cols.emplace_back(static_cast<int32_t>(local_idx));
      }
    }
    if (!valid_path) {
      miss_indices.emplace_back(i);
      continue;
    }

    const auto bonus_token = accept_tokens[i * d + static_cast<size_t>(accept_len - 1)];
    const PathBonusKey cache_key{state_ids[i], path_cols, bonus_token};
    auto cache_iter = precomputed_cache_.find(cache_key);
    if (cache_iter != precomputed_cache_.end()) {
      out.bonus_prediction_hit[i] = 1;
      out.precomputed_cache_hit[i] = 1;
      ++out.bonus_prediction_hit_ct;
      ++out.precomputed_cache_hit_ct;
      std::copy(cache_iter->second.token.begin(), cache_iter->second.token.end(), out.result.token.begin() + i * d);
      std::copy(cache_iter->second.mask.begin(), cache_iter->second.mask.end(), out.result.mask.begin() + i * d * d);
    } else {
      miss_indices.emplace_back(i);
    }
  }

  const auto miss_batch_size = std::max<size_t>(miss_indices.size(), 1);
  for (const auto i : miss_indices) {
    if (fallback_tokens[i].empty()) {
      throw std::runtime_error("selectPrecomputedDrafts received an empty fallback token tail");
    }
    auto& state = match_state_[state_ids[i]];
    auto res = buildMatchUnlocked(fallback_tokens[i], fallback_total_lens[i], state, miss_batch_size);
    std::copy(res.token.begin(), res.token.end(), out.result.token.begin() + i * d);
    std::copy(res.mask.begin(), res.mask.end(), out.result.mask.begin() + i * d * d);
  }

  return out;
}

std::vector<int32_t> Ngram::precomputedRootBonusTokens(const std::vector<int64_t>& state_ids) const {
  std::unique_lock<std::mutex> lock(mutex_);
  std::vector<int32_t> out;
  out.reserve(state_ids.size());
  const std::vector<int32_t> root_path_cols{0};

  for (const auto state_id : state_ids) {
    const PathKey key{state_id, root_path_cols};
    auto iter = precomputed_bonus_candidates_.find(key);
    if (iter == precomputed_bonus_candidates_.end() || iter->second.empty()) {
      out.emplace_back(-1);
    } else {
      out.emplace_back(iter->second.front());
    }
  }
  return out;
}

void Ngram::eraseMatchState(const std::vector<int64_t>& state_ids) {
  std::unique_lock<std::mutex> lock(mutex_);
  for (const auto& sid : state_ids) {
    match_state_.erase(sid);
  }
}

}  // namespace ngram
