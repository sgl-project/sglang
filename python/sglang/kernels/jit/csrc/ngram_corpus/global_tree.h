#pragma once

#include "param.h"
#include "result.h"
#include "suffix_automaton.h"
#include "trie.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace sglang {

namespace ngram {

class GlobalTree {
 public:
  GlobalTree(
      size_t proposal_budget,
      size_t max_breadth,
      size_t max_match_length,
      GlobalTreeMode score_mode,
      std::optional<TrieAnchor> trie_anchor,
      std::vector<SamAnchor> sam_anchors);

  void generate();
  Result materialize(int32_t last_token) const;

 private:
  struct TrieCursor {
    const TrieNode* state = nullptr;
    double contribution_score = 1.0;
  };

  struct SamCursor {
    uint32_t sam_index = 0;
    int state = 0;
    double contribution_score = 1.0;
  };

  using Cursor = std::variant<TrieCursor, SamCursor>;

  struct ScratchCandidate {
    Cursor successor;
    double contribution_score = 0.0;
    int32_t token = 0;
    uint32_t insertion_sequence = 0;

    bool operator<(const ScratchCandidate& other) const noexcept {
      if (contribution_score != other.contribution_score) {
        return contribution_score < other.contribution_score;
      }
      return insertion_sequence > other.insertion_sequence;
    }
  };

  struct FrontierEdge {
    double contribution_score = 0.0;
    uint32_t parent_id = 0;
    uint32_t cursor_begin = 0;
    uint32_t cursor_count = 0;
    int32_t token = 0;
    uint32_t insertion_sequence = 0;

    bool operator<(const FrontierEdge& other) const noexcept {
      if (contribution_score != other.contribution_score) {
        return contribution_score < other.contribution_score;
      }
      return insertion_sequence > other.insertion_sequence;
    }
  };

  struct GlobalNode {
    uint32_t parent_id = 0;
    int32_t token = 0;
  };

  double initialContribution(int32_t matched_length) const;
  void seedFrontier();
  template <typename TypedCursor>
  void
  appendCandidate(const TypedCursor& cursor, int32_t token, TypedCursor successor, uint64_t mass, uint64_t total_mass);
  void appendCursorTransitionsCandidates(const Cursor& cursor);
  void expandSelectCursors(uint32_t global_parent_id, uint32_t cursor_begin, uint32_t cursor_count);
  void selectNextNode();

  size_t proposal_budget_;
  size_t max_breadth_;
  size_t max_match_length_;
  GlobalTreeMode score_mode_;
  std::optional<TrieAnchor> trie_anchor_;
  std::vector<SamAnchor> sam_anchors_;
  std::vector<GlobalNode> nodes_;
  std::vector<Cursor> cursor_arena_;
  std::vector<FrontierEdge> frontier_;
  std::vector<ScratchCandidate> candidate_buffer_;
  std::vector<TrieFrequencyTransition> trie_transitions_;
  std::vector<SamFrequencyTransition> sam_transitions_;
  uint32_t next_insertion_sequence_ = 0;
};

}  // namespace ngram

}  // namespace sglang
