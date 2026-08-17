# -*- coding: utf-8 -*-

import logging
from collections.abc import Iterable, Sequence
from typing import Dict, List, Optional, Tuple

import numpy as np

from sglang.kernels.ops.speculative.ngram_corpus import get_ngram_corpus_cls

logger = logging.getLogger(__name__)

_ALL_CORPORA_HANDLE = -1
_NO_CORPUS_HANDLE = 0


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
        # Published id -> (request-facing handle, token count).
        self._corpora: Dict[str, Tuple[int, int]] = {}
        self._next_corpus_handle: int = 1
        self._total_loaded_tokens: int = 0
        self._staging_corpus_id: Optional[str] = None
        self._staging_corpus_handle: Optional[int] = None
        self._staging_token_count: int = 0

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
        if not corpus_id:
            raise ValueError("External corpus id must be non-empty.")
        if any(delimiter in corpus_id for delimiter in "\t\n\r"):
            raise ValueError(
                "External corpus id must not contain tabs or line breaks."
            )
        if corpus_id in self._corpora:
            raise ValueError(
                f"External corpus '{corpus_id}' already exists. Remove it before "
                f"adding a new corpus with the same id."
            )
        if self._staging_corpus_id is not None:
            raise RuntimeError(
                f"External corpus '{self._staging_corpus_id}' is still staged. "
                "Commit or cancel it before starting another load."
            )
        corpus_handle = self._next_corpus_handle
        self._next_corpus_handle += 1
        # A concurrent removal can only make this snapshot conservative.
        _, loaded_token_count = self._obj.load_external_corpus_named(
            corpus_id, corpus_handle, chunks, self.remaining_token_budget
        )
        self._staging_corpus_id = corpus_id
        self._staging_corpus_handle = corpus_handle
        self._staging_token_count = loaded_token_count
        return loaded_token_count

    def commit_external_corpus_load(
        self, corpus_id: str, loaded_token_count: int
    ) -> None:
        """Atomically publish the finalized staging corpus to local readers."""
        if self._staging_corpus_id != corpus_id:
            raise RuntimeError(
                "No staged external corpus matches "
                f"'{corpus_id}' (staged: {self._staging_corpus_id!r})."
            )
        if loaded_token_count != self._staging_token_count:
            raise RuntimeError(
                f"Staged corpus '{corpus_id}' has {self._staging_token_count} "
                f"tokens, not {loaded_token_count}."
            )
        assert self._staging_corpus_handle is not None
        # Install the single Python metadata entry first. It has no observers
        # during this scheduler-thread call, and can be removed if the C++
        # commit reports a normal validation error. After C++ publication only
        # non-failing attribute assignments remain.
        new_total = self._total_loaded_tokens + loaded_token_count
        self._corpora[corpus_id] = (
            self._staging_corpus_handle,
            loaded_token_count,
        )
        try:
            self._obj.commit_corpus(corpus_id)
        except Exception:
            self._corpora.pop(corpus_id, None)
            raise
        self._total_loaded_tokens = new_total
        self._clear_staging_metadata()

    def cancel_external_corpus_load(self, corpus_id: Optional[str] = None) -> None:
        """Discard an unpublished staging corpus; published corpora are untouched."""
        if (
            corpus_id is not None
            and self._staging_corpus_id is not None
            and corpus_id != self._staging_corpus_id
        ):
            raise RuntimeError(
                f"Cannot cancel staged corpus '{self._staging_corpus_id}' as "
                f"'{corpus_id}'."
            )
        self._obj.cancel_corpus_load()
        self._clear_staging_metadata()

    def _clear_staging_metadata(self) -> None:
        self._staging_corpus_id = None
        self._staging_corpus_handle = None
        self._staging_token_count = 0

    def remove_external_corpus(self, corpus_id: str) -> None:
        self._obj.remove_corpus(corpus_id)
        metadata = self._corpora.pop(corpus_id, None)
        if metadata is not None:
            self._total_loaded_tokens -= metadata[1]

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
        corpus_ids: Optional[List[Optional[str]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(req_ids) != len(batch_tokens) or len(req_ids) != len(total_lens):
            raise ValueError(
                "req_ids, batch_tokens, and total_lens must have the same batch size"
            )
        if corpus_ids is not None and len(corpus_ids) != len(req_ids):
            raise ValueError(
                "corpus_ids must be omitted or match the request batch size "
                f"({len(corpus_ids)} != {len(req_ids)})"
            )
        if corpus_ids is not None and any(
            corpus_id is not None and not isinstance(corpus_id, str)
            for corpus_id in corpus_ids
        ):
            raise ValueError("Every corpus_id must be a string or None.")
        state_ids = [self._get_state_id(rid) for rid in req_ids]
        corpus_handles = None
        if corpus_ids is not None:
            corpus_handles = []
            for corpus_id in corpus_ids:
                if corpus_id is None:
                    corpus_handles.append(_ALL_CORPORA_HANDLE)
                elif not corpus_id:
                    corpus_handles.append(_NO_CORPUS_HANDLE)
                else:
                    corpus_handles.append(
                        self._corpora.get(
                            corpus_id, (_NO_CORPUS_HANDLE, 0)
                        )[0]
                    )
        return self._obj.match_stateful(
            state_ids, batch_tokens, total_lens, corpus_handles
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
