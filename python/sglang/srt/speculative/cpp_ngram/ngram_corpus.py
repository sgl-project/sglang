# -*- coding: utf-8 -*-

import logging
from collections.abc import Iterable, Sequence
from typing import Dict, List, Optional, Tuple

import numpy as np

from sglang.jit_kernel.ngram_corpus import get_ngram_corpus_cls

logger = logging.getLogger(__name__)


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
        self.draft_token_num = draft_token_num
        self.external_corpus_max_tokens = external_corpus_max_tokens
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

    @staticmethod
    def direct_child_tokens(
        tokens: np.ndarray,
        tree_mask: np.ndarray,
        parent: int,
        max_candidates: Optional[int] = None,
    ) -> List[int]:
        child_tokens = []
        seen_tokens = set()
        for child in range(parent + 1, len(tokens)):
            if tree_mask[child, child] == 0 or tree_mask[child, parent] == 0:
                continue
            ancestor_cols = np.nonzero(tree_mask[child, :child])[0]
            if len(ancestor_cols) == 0 or ancestor_cols[-1] != parent:
                continue
            token = int(tokens[child])
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            child_tokens.append(token)
            if max_candidates is not None and len(child_tokens) >= max_candidates:
                break
        return child_tokens

    def precompute_drafts(
        self,
        req_ids: List[str],
        base_tokens: List[List[int]],
        total_lens: List[int],
        draft_tokens,
        tree_mask,
        bonus_topk: int,
        max_trie_depth: int,
    ) -> Tuple[int, int, int]:
        state_ids = [self._get_state_id(rid) for rid in req_ids]
        return self._obj.precompute_drafts_stateful_wrapper(
            state_ids,
            base_tokens,
            total_lens,
            draft_tokens,
            tree_mask,
            bonus_topk,
            max_trie_depth,
        )

    def precomputed_root_bonus_tokens(self, req_ids: List[str]) -> List[int]:
        state_ids = [self._get_state_id(rid) for rid in req_ids]
        return self._obj.precomputed_root_bonus_tokens_stateful_wrapper(state_ids)

    def select_precomputed_drafts(
        self,
        req_ids: List[str],
        accept_tokens,
        accept_lens,
        accept_index,
        fallback_tokens: List[List[int]],
        fallback_total_lens: List[int],
    ):
        state_ids = [self._get_state_id(rid) for rid in req_ids]
        return self._obj.select_precomputed_drafts_stateful_wrapper(
            state_ids,
            accept_tokens,
            accept_lens,
            accept_index,
            fallback_tokens,
            fallback_total_lens,
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

    def debug_result(
        self, decoding_ids: np.ndarray, decoding_masks: np.ndarray, tokenizer=None
    ):
        decoding_ids = decoding_ids.reshape(-1, self.draft_token_num)
        decoding_masks = decoding_masks.reshape(
            -1, self.draft_token_num, self.draft_token_num
        )
        logger.info(f"\n{decoding_ids=}\n{decoding_masks=}")
        for i in range(decoding_ids.shape[0]):
            leaf_paths = self.leaf_paths_from_mask(
                decoding_ids[i].tolist(), decoding_masks[i].tolist()
            )
            if tokenizer is None:
                logger.info(f"draft path {i}: {leaf_paths}")
            else:
                logger.info(f"result {i}:")
                for leaf_path in leaf_paths:
                    logger.info(
                        f"draft path {i}: {leaf_path} -> {tokenizer.decode(leaf_path, ensure_ascii=False)}"
                    )


# main function
if __name__ == "__main__":
    format = f"%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    token_ids = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 44, 55, 66, 77, 88, 99, 100],
    ]
    corpus = NgramCorpus(max_trie_depth=12, draft_token_num=8)
    corpus.batch_put(token_ids)

    corpus.synchronize()
    queries = [[1, 2, 3], [3, 44], [3, 6, 999]]
    decoding_ids, decoding_masks = corpus.batch_get(
        req_ids=[f"query-{i}" for i in range(len(queries))],
        batch_tokens=queries,
        total_lens=[len(q) for q in queries],
    )

    corpus.debug_result(decoding_ids, decoding_masks)
