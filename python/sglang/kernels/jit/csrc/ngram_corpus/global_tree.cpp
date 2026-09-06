#include "global_tree.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sglang {

namespace ngram {

GlobalTree::GlobalTree(
    size_t proposal_budget,
    size_t max_breadth,
    size_t max_match_length,
    GlobalTreeMode score_mode,
    std::optional<TrieAnchor> trie_anchor,
    std::vector<SamAnchor> sam_anchors)
    : proposal_budget_(proposal_budget),
      max_breadth_(max_breadth),
      max_match_length_(max_match_length),
      score_mode_(score_mode),
      trie_anchor_(std::move(trie_anchor)),
      sam_anchors_(std::move(sam_anchors)) {
  const size_t source_count = static_cast<size_t>(trie_anchor_.has_value()) + sam_anchors_.size();
  const size_t candidates_per_expansion = source_count * max_breadth_;
  const size_t max_generated_candidates = proposal_budget_ * candidates_per_expansion;
  nodes_.reserve(proposal_budget_ + 1);
  cursor_arena_.reserve(source_count + max_generated_candidates);
  frontier_.reserve(max_generated_candidates);
  candidate_buffer_.reserve(candidates_per_expansion);
  trie_transitions_.reserve(max_breadth_);
  sam_transitions_.reserve(max_breadth_);
  nodes_.emplace_back();
}

void GlobalTree::generate() {
  seedFrontier();
  for (size_t proposal_index = 0; proposal_index < proposal_budget_ && !frontier_.empty(); ++proposal_index) {
    selectNextNode();
  }
}

double GlobalTree::initialContribution(int32_t matched_length) const {
  switch (score_mode_) {
    case GlobalTreeMode::PATH_PROBABILITY:
      return 1.0;
    case GlobalTreeMode::SPECIFICITY_PATH_PROBABILITY:
      return static_cast<double>(matched_length) / static_cast<double>(max_match_length_);
    case GlobalTreeMode::DISABLED:
      throw std::runtime_error("GlobalTree cannot use the disabled score mode");
  }
  throw std::runtime_error("GlobalTree received an unknown score mode");
}

void GlobalTree::seedFrontier() {
  const auto cursor_begin = static_cast<uint32_t>(cursor_arena_.size());
  if (trie_anchor_) {
    cursor_arena_.push_back(TrieCursor{trie_anchor_->state, initialContribution(trie_anchor_->matched_length)});
  }
  for (size_t sam_index = 0; sam_index < sam_anchors_.size(); ++sam_index) {
    const auto& anchor = sam_anchors_[sam_index];
    cursor_arena_.push_back(
        SamCursor{static_cast<uint32_t>(sam_index), anchor.state, initialContribution(anchor.matched_length)});
  }
  expandSelectCursors(0, cursor_begin, static_cast<uint32_t>(cursor_arena_.size() - cursor_begin));
}

template <typename TypedCursor>
void GlobalTree::appendCandidate(
    const TypedCursor& cursor, int32_t token, TypedCursor successor, uint64_t mass, uint64_t total_mass) {
  if (mass == 0) {
    return;
  }

  const double conditional_probability = static_cast<double>(mass) / static_cast<double>(total_mass);
  const double contribution_score = cursor.contribution_score * conditional_probability;
  successor.contribution_score = contribution_score;
  candidate_buffer_.push_back(
      ScratchCandidate{
          .successor = std::move(successor),
          .contribution_score = contribution_score,
          .token = token,
          .insertion_sequence = next_insertion_sequence_++,
      });
}

void GlobalTree::appendCursorTransitionsCandidates(const Cursor& cursor) {
  if (const auto* trie_cursor = std::get_if<TrieCursor>(&cursor)) {
    const auto total_mass =
        trie_anchor_->trie->frequencyTransitions(trie_cursor->state, max_breadth_, trie_transitions_);
    for (const auto& transition : trie_transitions_) {
      appendCandidate(*trie_cursor, transition.token, TrieCursor{transition.state}, transition.mass, total_mass);
    }
  } else {
    const auto& sam_cursor = std::get<SamCursor>(cursor);
    const auto& anchor = sam_anchors_[sam_cursor.sam_index];
    const auto total_mass = anchor.sam->frequencyTransitions(sam_cursor.state, max_breadth_, sam_transitions_);
    for (const auto& transition : sam_transitions_) {
      appendCandidate(
          sam_cursor, transition.token, SamCursor{sam_cursor.sam_index, transition.state}, transition.mass, total_mass);
    }
  }
}

void GlobalTree::expandSelectCursors(uint32_t global_parent_id, uint32_t cursor_begin, uint32_t cursor_count) {
  candidate_buffer_.clear();
  for (uint32_t offset = 0; offset < cursor_count; ++offset) {
    appendCursorTransitionsCandidates(cursor_arena_[cursor_begin + offset]);
  }
  if (candidate_buffer_.empty()) {
    return;
  }

  std::sort(candidate_buffer_.begin(), candidate_buffer_.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.token != rhs.token) {
      return lhs.token < rhs.token;
    }
    return rhs < lhs;
  });

  for (size_t group_begin = 0; group_begin < candidate_buffer_.size();) {
    size_t group_end = group_begin + 1;
    while (group_end < candidate_buffer_.size() &&
           candidate_buffer_[group_end].token == candidate_buffer_[group_begin].token) {
      ++group_end;
    }

    const auto& best = candidate_buffer_[group_begin];
    const auto cursor_begin = static_cast<uint32_t>(cursor_arena_.size());
    for (size_t candidate_index = group_begin; candidate_index < group_end; ++candidate_index) {
      cursor_arena_.push_back(candidate_buffer_[candidate_index].successor);
    }
    frontier_.push_back(
        FrontierEdge{
            .contribution_score = best.contribution_score,
            .parent_id = global_parent_id,
            .cursor_begin = cursor_begin,
            .cursor_count = static_cast<uint32_t>(group_end - group_begin),
            .token = best.token,
            .insertion_sequence = best.insertion_sequence,
        });
    group_begin = group_end;
  }
}

void GlobalTree::selectNextNode() {
  const auto best = std::max_element(frontier_.begin(), frontier_.end());
  const auto edge = *best;
  *best = frontier_.back();
  frontier_.pop_back();

  const auto node_id = static_cast<uint32_t>(nodes_.size());
  nodes_.push_back(GlobalNode{edge.parent_id, edge.token});
  if (nodes_.size() - 1 < proposal_budget_) {
    expandSelectCursors(node_id, edge.cursor_begin, edge.cursor_count);
  }
}

Result GlobalTree::materialize(int32_t last_token) const {
  std::vector<int32_t> proposal_tokens;
  std::vector<int32_t> proposal_parents;
  proposal_tokens.reserve(nodes_.size() - 1);
  proposal_parents.reserve(nodes_.size() - 1);

  for (size_t node_id = 1; node_id < nodes_.size(); ++node_id) {
    proposal_tokens.push_back(nodes_[node_id].token);
    proposal_parents.push_back(static_cast<int32_t>(nodes_[node_id].parent_id));
  }

  return fillResultWithParentArray(last_token, proposal_budget_ + 1, proposal_tokens, proposal_parents);
}

}  // namespace ngram

}  // namespace sglang
