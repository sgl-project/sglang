"""Validate host-corpus parameters before converting signed values to size_t."""


def validate_ngram_config(
    *,
    min_bfs_breadth: int,
    max_bfs_breadth: int,
    max_trie_depth: int,
    capacity: int,
    draft_token_num: int,
    external_sam_budget: int,
    external_corpus_max_tokens: int,
) -> None:
    if not 1 <= min_bfs_breadth <= max_bfs_breadth <= 2**31 - 1:
        raise ValueError(
            "NGRAM requires 1 <= speculative_ngram_min_bfs_breadth <= "
            "speculative_ngram_max_bfs_breadth <= 2147483647."
        )
    if not 2 <= max_trie_depth <= 2**31 - 1:
        raise ValueError("speculative_ngram_max_trie_depth must be in [2, 2147483647].")
    if capacity < max_trie_depth + 1:
        raise ValueError(
            "speculative_ngram_capacity must be at least "
            "speculative_ngram_max_trie_depth + 1 (including the trie root)."
        )
    if draft_token_num < 1:
        raise ValueError("speculative_num_draft_tokens must be positive for NGRAM.")
    if not 0 <= external_sam_budget <= draft_token_num - 1:
        raise ValueError(
            "speculative_ngram_external_sam_budget must be in "
            f"[0, speculative_num_draft_tokens - 1 ({draft_token_num - 1})]."
        )
    if external_corpus_max_tokens <= 0:
        raise ValueError(
            "speculative_ngram_external_corpus_max_tokens must be positive."
        )
