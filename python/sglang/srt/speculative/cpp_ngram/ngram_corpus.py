# -*- coding: utf-8 -*-

from collections.abc import Iterable, Sequence
from typing import Dict, List, NamedTuple, Tuple

import numpy as np

from sglang.kernels.ops.speculative.ngram_corpus import get_ngram_corpus_cls


class NgramPrecomputeResult(NamedTuple):
    stats: tuple[int, int, int]
    bonus_tokens: np.ndarray
    draft_tokens: np.ndarray
    tree_masks: np.ndarray


class NgramCorpus:
    def __init__(
        self,
        max_trie_depth=18,
        min_bfs_breadth=1,
        max_bfs_breadth=8,
        draft_token_num=8,
        match_type="BFS",
        capacity=1000000,
        external_sam_budget=0,
        external_corpus_max_tokens=10000000,
    ) -> None:
        cls = get_ngram_corpus_cls()
        self._obj = cls(
            capacity=capacity,
            max_trie_depth=max_trie_depth,
            min_bfs_breadth=min_bfs_breadth,
            max_bfs_breadth=max_bfs_breadth,
            draft_token_num=draft_token_num,
            match_type=match_type,
            external_sam_budget=external_sam_budget,
            external_corpus_max_tokens=external_corpus_max_tokens,
        )
        self.external_corpus_max_tokens = external_corpus_max_tokens
        self.draft_token_num = draft_token_num
        self._req_id_to_state_id: Dict[str, int] = {}
        self._next_state_id: int = 0
        self._corpus_token_counts: Dict[str, int] = {}
        self._total_loaded_tokens: int = 0

    def _get_state_id(self, req_id: str) -> int:
        sid = self._req_id_to_state_id.get(req_id)
        if sid is None:
            sid = self._next_state_id
            self._next_state_id += 1
            self._req_id_to_state_id[req_id] = sid
        return sid

    def batch_put(self, batch_tokens: List[List[int]]):
        self._obj.insert(batch_tokens)

    def synchronize(self):
        self._obj.synchronize()  # type: ignore

    @property
    def remaining_token_budget(self) -> int:
        return self.external_corpus_max_tokens - self._total_loaded_tokens

    def load_external_corpus_named(
        self, corpus_id: str, chunks: Iterable[Sequence[int]]
    ) -> int:
        if corpus_id in self._corpus_token_counts:
            raise ValueError(
                f"External corpus '{corpus_id}' already exists. Remove it before "
                f"adding a new corpus with the same id."
            )
        # Note(kpham-sgl): remaining_token_budget is stale (e.g if there are removes
        # during the load), which makes the budget more conservative than it should be.
        # This is acceptable because otherwise load_external_corpus_named would need to check the budget after each chunk,
        # which would be inefficient.
        _, loaded_token_count = self._obj.load_external_corpus_named(
            corpus_id, chunks, self.remaining_token_budget
        )
        return loaded_token_count

    # Commit corpus bookkeeping after successful load. Call only at background thread join.
    # (or after synchronous load_external_corpus_named returns)
    def commit_external_corpus_load(
        self, corpus_id: str, loaded_token_count: int
    ) -> None:
        self._corpus_token_counts[corpus_id] = loaded_token_count
        self._total_loaded_tokens += loaded_token_count

    def remove_external_corpus(self, corpus_id: str) -> None:
        self._obj.remove_corpus(corpus_id)
        old_count = self._corpus_token_counts.pop(corpus_id, 0)
        self._total_loaded_tokens -= old_count

    def list_external_corpora(self) -> Dict[str, int]:
        return self._obj.list_corpora()

    def reset(self):
        self._obj.reset()  # type: ignore
        self._req_id_to_state_id.clear()
        self._next_state_id = 0

    def batch_get(
        self,
        req_ids: List[str],
        batch_tokens: List[List[int]],
        total_lens: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        state_ids = [self._get_state_id(rid) for rid in req_ids]
        return self._obj.match_stateful(state_ids, batch_tokens, total_lens)

    def precompute_drafts_dense(
        self,
        *,
        base_tokens: List[List[int]],
        total_lens: List[int],
        draft_tokens,
        tree_mask,
        bonus_topk: int,
        max_trie_depth: int,
        wide_bonus_ratio: float = 0.5,
    ) -> NgramPrecomputeResult:
        stats, bonus_tokens, next_draft_tokens, tree_masks = (
            self._obj.precompute_drafts_dense_host(
                base_tokens=base_tokens,
                total_lens=total_lens,
                draft_tokens=draft_tokens,
                tree_mask=tree_mask,
                bonus_topk=bonus_topk,
                max_trie_depth=max_trie_depth,
                wide_bonus_ratio=wide_bonus_ratio,
            )
        )
        batch_size = len(base_tokens)
        draft_token_num = self.draft_token_num
        return NgramPrecomputeResult(
            stats=stats,
            bonus_tokens=bonus_tokens.reshape(batch_size, draft_token_num, bonus_topk),
            draft_tokens=next_draft_tokens.reshape(
                batch_size,
                draft_token_num,
                bonus_topk,
                draft_token_num,
            ),
            tree_masks=tree_masks.reshape(
                batch_size,
                draft_token_num,
                bonus_topk,
                draft_token_num,
                draft_token_num,
            ),
        )

    def erase_match_state(self, req_ids: List[str]):
        state_ids = []
        for rid in req_ids:
            sid = self._req_id_to_state_id.pop(rid, None)
            if sid is not None:
                state_ids.append(sid)
        if state_ids:
            self._obj.erase_states(state_ids)

    def leaf_paths_from_mask(
        self, tokens: List[int], tree_mask: List[List[int]]
    ) -> List[List[int]]:
        """
        Find all leaf paths according to the binary tree_mask (i.e., paths that are not prefixes of any other path).

        Args:
            mask   : List[List[int]]   # nxn binary matrix
            tokens : List[int]         # token list corresponding to columns

        Returns:
            List[List[int]]            # token lists of only the leaf paths, preserving their order of appearance
        """

        row_sets = [
            (i, {idx for idx, v in enumerate(row) if v == 1})
            for i, row in enumerate(tree_mask)
        ]
        leaf_sets = []
        leaf_rows = []

        for i, cur_set in reversed(row_sets):
            if any(cur_set <= kept for kept in leaf_sets):
                continue
            leaf_sets.append(cur_set)
            leaf_rows.append(i)

        leaf_rows.reverse()
        result = []
        for r in leaf_rows:
            path = [tokens[col] for col in range(len(tokens)) if tree_mask[r][col] == 1]
            result.append(path)

        return result
