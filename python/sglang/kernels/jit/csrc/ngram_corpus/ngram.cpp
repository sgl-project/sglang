#include "ngram.h"

#include "trie.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace sglang {

namespace ngram {

namespace {

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

std::vector<int32_t>
buildParentNodes(const std::vector<uint8_t>& tree_mask, size_t row_offset, size_t draft_token_num) {
  std::vector<int32_t> parents(draft_token_num, -1);
  for (size_t node = 1; node < draft_token_num; ++node) {
    const auto node_row = row_offset + node * draft_token_num;
    if (tree_mask[node_row + node] == 0) {
      continue;
    }
    for (size_t col = 0; col < node; ++col) {
      if (tree_mask[node_row + col] != 0) {
        parents[node] = static_cast<int32_t>(col);
      }
    }
  }
  return parents;
}

std::vector<int32_t> pathDraftTokens(
    const std::vector<int32_t>& draft_tokens,
    const std::vector<uint8_t>& tree_mask,
    size_t token_offset,
    size_t node_row,
    size_t draft_token_num) {
  std::vector<int32_t> path_tokens;
  path_tokens.reserve(draft_token_num);
  for (size_t col = 1; col < draft_token_num; ++col) {
    if (tree_mask[node_row + col] != 0) {
      path_tokens.emplace_back(draft_tokens[token_offset + col]);
    }
  }
  return path_tokens;
}

std::vector<std::unordered_set<int32_t>> buildDirectChildTokenSets(
    const std::vector<int32_t>& draft_tokens, const std::vector<int32_t>& parents, size_t token_offset) {
  std::vector<std::unordered_set<int32_t>> child_token_sets(parents.size());
  for (size_t child = 1; child < parents.size(); ++child) {
    const auto parent = parents[child];
    if (parent >= 0) {
      child_token_sets[parent].emplace(draft_tokens[token_offset + child]);
    }
  }
  return child_token_sets;
}

size_t wideBonusPathCount(size_t draft_token_num, double wide_bonus_ratio) {
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

PrecomputeDraftsStats Ngram::precomputeDraftsDense(
    const std::vector<std::vector<int32_t>>& base_tokens,
    const std::vector<size_t>& base_total_lens,
    const std::vector<int32_t>& draft_tokens,
    const std::vector<uint8_t>& tree_mask,
    size_t bonus_topk,
    size_t max_trie_depth,
    double wide_bonus_ratio,
    PrecomputeDraftsDenseCache& dense_cache) {
  if (base_tokens.size() != base_total_lens.size()) {
    throw std::runtime_error("precomputeDraftsDense expects base_tokens and base_total_lens to match in size");
  }
  const size_t bs = base_tokens.size();
  const size_t d = param_.draft_token_num;
  if (draft_tokens.size() != bs * d) {
    throw std::runtime_error("precomputeDraftsDense received draft_tokens with unexpected size");
  }
  if (tree_mask.size() != bs * d * d) {
    throw std::runtime_error("precomputeDraftsDense received tree_mask with unexpected size");
  }

  std::unique_lock<std::mutex> lock(mutex_);
  const auto wide_bonus_path_count = wideBonusPathCount(d, wide_bonus_ratio);
  const auto max_phase2_entries = bs * (wide_bonus_path_count * bonus_topk + (d - wide_bonus_path_count));
  dense_cache.bonus_tokens.assign(bs * d * bonus_topk, -1);
  dense_cache.draft_tokens.assign(bs * d * bonus_topk * d, 0);
  dense_cache.tree_masks.assign(bs * d * bonus_topk * d * d, 0);

  struct Phase2Entry {
    size_t req_idx;
    size_t path_node;
    size_t bonus_slot;
    std::vector<int32_t> tokens;
    size_t total_len;
    MatchState state;
  };
  std::vector<Phase2Entry> phase2_entries;
  phase2_entries.reserve(max_phase2_entries);
  PrecomputeDraftsStats stats;

  for (size_t req_idx = 0; req_idx < bs; ++req_idx) {
    const auto row_token_offset = req_idx * d;
    const auto row_mask_offset = req_idx * d * d;
    const auto parent_nodes = buildParentNodes(tree_mask, row_mask_offset, d);
    const auto direct_child_token_sets = buildDirectChildTokenSets(draft_tokens, parent_nodes, row_token_offset);

    for (size_t node = 0; node < d; ++node) {
      const auto node_row = row_mask_offset + node * d;
      if (tree_mask[node_row + node] == 0 || tree_mask[node_row] == 0) {
        continue;
      }
      ++stats.num_paths;

      auto path_draft_tokens = pathDraftTokens(draft_tokens, tree_mask, row_token_offset, node_row, d);
      const auto& current_child_tokens = direct_child_token_sets[node];
      size_t num_bonus_candidates = 0;
      const auto path_bonus_topk = node < wide_bonus_path_count ? bonus_topk : std::min<size_t>(bonus_topk, 1);
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
        if (current_child_tokens.contains(bonus_token)) {
          continue;
        }
        const auto bonus_slot = num_bonus_candidates++;
        const auto dense_bonus_offset = (req_idx * d + node) * bonus_topk + bonus_slot;
        dense_cache.bonus_tokens[dense_bonus_offset] = bonus_token;
        auto draft_check_tokens = appendTokenAndTrim(check_tokens, bonus_token, max_trie_depth);
        phase2_entries.emplace_back(
            Phase2Entry{req_idx, node, bonus_slot, std::move(draft_check_tokens), check_total_len + 1, path_state});
        if (num_bonus_candidates >= path_bonus_topk) {
          break;
        }
      }
    }
  }

  stats.num_phase2_contexts = static_cast<int64_t>(phase2_entries.size());
  const auto phase2_batch_size = std::max<size_t>(phase2_entries.size(), 1);
  for (auto& entry : phase2_entries) {
    auto res = buildMatchUnlocked(entry.tokens, entry.total_len, entry.state, phase2_batch_size);
    const auto dense_token_offset = ((entry.req_idx * d + entry.path_node) * bonus_topk + entry.bonus_slot) * d;
    const auto dense_mask_offset = dense_token_offset * d;
    const auto result_size = std::min(d, res.token.size());
    std::copy_n(res.token.begin(), result_size, dense_cache.draft_tokens.begin() + dense_token_offset);
    for (size_t row = 0; row < result_size; ++row) {
      std::copy_n(
          res.mask.begin() + row * res.token.size(),
          result_size,
          dense_cache.tree_masks.begin() + dense_mask_offset + row * d);
    }
  }
  stats.num_cache_entries = static_cast<int64_t>(phase2_entries.size());
  return stats;
}

void Ngram::eraseMatchState(const std::vector<int64_t>& state_ids) {
  std::unique_lock<std::mutex> lock(mutex_);
  for (const auto& sid : state_ids) {
    match_state_.erase(sid);
  }
}

}  // namespace ngram

}  // namespace sglang
