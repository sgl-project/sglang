#include "ngram.h"

#include "trie.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ngram {

namespace {

struct WeightedBudgetSource {
  double score = 0.0;
  size_t cap = 0;
  size_t budget = 0;
};

double computeSourceScore(const MatchQuality& quality, double source_prior, const Param& param) {
  if (!quality.has_match) {
    return 0.0;
  }
  const double total_weight = param.match_specificity_weight + param.match_confidence_weight;
  if (total_weight <= 0.0) {
    return 0.0;
  }
  // Implemented source-importance formula:
  //   score = source_prior * (w_specificity * specificity + w_confidence * confidence)
  //
  // where:
  // - specificity is normalized match depth / matched_len for the best anchor
  // - confidence is normalized top-1 next-token mass at that anchor
  // - the w_* terms are normalized from the user-provided match weights below
  const double specificity_weight = param.match_specificity_weight / total_weight;
  const double confidence_weight = param.match_confidence_weight / total_weight;
  return source_prior * (specificity_weight * quality.specificity + confidence_weight * quality.confidence);
}

size_t ceilShare(double share, size_t total_budget) {
  if (share <= 0.0 || total_budget == 0) {
    return 0;
  }
  return std::min(total_budget, static_cast<size_t>(std::ceil(share * static_cast<double>(total_budget) - 1e-12)));
}

void allocateLargestRemainder(size_t total_budget, std::vector<WeightedBudgetSource>* sources) {
  size_t remaining_budget = total_budget;
  while (remaining_budget > 0) {
    double total_score = 0.0;
    std::vector<size_t> active;
    active.reserve(sources->size());
    for (size_t i = 0; i < sources->size(); ++i) {
      const auto& source = (*sources)[i];
      if (source.score > 0.0 && source.budget < source.cap) {
        active.push_back(i);
        total_score += source.score;
      }
    }
    if (active.empty() || total_score <= 0.0) {
      break;
    }

    std::vector<std::pair<double, size_t>> remainders;
    remainders.reserve(active.size());
    size_t distributed = 0;
    for (const auto idx : active) {
      auto& source = (*sources)[idx];
      const auto remaining_cap = source.cap - source.budget;
      const double ideal = static_cast<double>(remaining_budget) * source.score / total_score;
      const auto whole = std::min(remaining_cap, static_cast<size_t>(ideal));
      source.budget += whole;
      distributed += whole;
      if (source.budget < source.cap) {
        remainders.emplace_back(ideal - static_cast<double>(whole), idx);
      }
    }

    remaining_budget -= distributed;
    if (remaining_budget == 0 || remainders.empty()) {
      break;
    }

    std::sort(remainders.begin(), remainders.end(), [sources](const auto& lhs, const auto& rhs) {
      if (lhs.first != rhs.first) {
        return lhs.first > rhs.first;
      }
      const auto& lhs_source = (*sources)[lhs.second];
      const auto& rhs_source = (*sources)[rhs.second];
      if (lhs_source.score != rhs_source.score) {
        return lhs_source.score > rhs_source.score;
      }
      return lhs.second < rhs.second;
    });

    size_t assigned = 0;
    for (const auto& [_, idx] : remainders) {
      if (remaining_budget == 0) {
        break;
      }
      auto& source = (*sources)[idx];
      if (source.budget >= source.cap) {
        continue;
      }
      ++source.budget;
      --remaining_budget;
      ++assigned;
    }
    if (assigned == 0) {
      break;
    }
  }
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
  if (param_.trie_source_prior < 0.0) {
    throw std::runtime_error(
        "trie_source_prior must be greater than or equal to 0, current value: " +
        std::to_string(param_.trie_source_prior));
  }
  if (!(param_.min_trie_share >= 0.0 && param_.min_trie_share <= 1.0)) {
    throw std::runtime_error(
        "min_trie_share must be between 0 and 1, current value: " + std::to_string(param_.min_trie_share));
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

  // All budget values are loop-invariant (mutex_ held, sams_ won't change).
  const size_t num_sams = sams_.size();
  const auto total_draft_token_num = param_.get_draft_token_num(tokens.size());
  const size_t max_total_sam_budget =
      num_sams > 0 ? std::min(param_.external_sam_budget, total_draft_token_num) : size_t{0};
  const size_t trie_base_budget = total_draft_token_num - max_total_sam_budget;
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

    auto& state = match_state_[state_ids[i]];
    auto trie_anchors = trie_->match(suffix.data(), suffix.size(), state, total_lens[i]);
    auto trie_quality = trie_->summarizeMatchQuality(trie_anchors, param_);

    if (max_total_sam_budget == 0) {
      auto res = (trie_.get()->*trie_anchored_build_fn)(trie_anchors, suffix.back(), total_draft_token_num, param_);
      merged.token.insert(merged.token.end(), res.token.begin(), res.token.end());
      merged.mask.insert(merged.mask.end(), res.mask.begin(), res.mask.end());
      continue;
    }

    struct SamMatchView {
      const SuffixAutomaton* sam = nullptr;
      std::vector<SamAnchor> anchors;
      MatchQuality quality;
      size_t budget = 0;
    };

    std::vector<SamMatchView> sam_matches;
    sam_matches.reserve(ordered_sams.size());
    for (const auto& [_, sam] : ordered_sams) {
      auto anchors = sam->match(suffix.data(), suffix.size(), param_.max_trie_depth);
      auto quality = sam->summarizeMatchQuality(anchors, param_);
      sam_matches.push_back(SamMatchView{sam, std::move(anchors), quality, 0});
    }

    size_t trie_budget = trie_base_budget;
    size_t flexible_budget = max_total_sam_budget;
    if (trie_quality.has_match && flexible_budget > 0) {
      const auto trie_floor_budget =
          std::max(trie_base_budget, ceilShare(param_.min_trie_share, total_draft_token_num));
      const auto reserved_for_trie = std::min(flexible_budget, trie_floor_budget - trie_base_budget);
      trie_budget += reserved_for_trie;
      flexible_budget -= reserved_for_trie;
    }

    if (flexible_budget > 0) {
      std::vector<WeightedBudgetSource> sources;
      const auto no_source = std::numeric_limits<size_t>::max();
      size_t trie_source_idx = no_source;
      if (trie_quality.has_match) {
        trie_source_idx = sources.size();
        sources.push_back(
            WeightedBudgetSource{
                computeSourceScore(trie_quality, param_.trie_source_prior, param_), flexible_budget, 0});
      }

      std::vector<size_t> sam_source_indices(sam_matches.size(), no_source);
      for (size_t sam_idx = 0; sam_idx < sam_matches.size(); ++sam_idx) {
        const auto score = computeSourceScore(sam_matches[sam_idx].quality, 1.0, param_);
        if (score <= 0.0) {
          continue;
        }
        sam_source_indices[sam_idx] = sources.size();
        sources.push_back(WeightedBudgetSource{score, flexible_budget, 0});
      }

      allocateLargestRemainder(flexible_budget, &sources);

      if (trie_source_idx != no_source) {
        trie_budget += sources[trie_source_idx].budget;
      }
      for (size_t sam_idx = 0; sam_idx < sam_matches.size(); ++sam_idx) {
        const auto source_idx = sam_source_indices[sam_idx];
        if (source_idx != no_source) {
          sam_matches[sam_idx].budget = sources[source_idx].budget;
        }
      }
    }

    auto combined = (trie_.get()->*trie_anchored_build_fn)(trie_anchors, suffix.back(), trie_budget, param_);

    for (const auto& sam_match : sam_matches) {
      if (sam_match.budget == 0) {
        continue;
      }
      auto sam_res =
          (sam_match.sam->*sam_anchored_build_fn)(sam_match.anchors, suffix.back(), sam_match.budget, param_);
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
