# -*- coding: utf-8 -*-

import logging
from collections.abc import Iterable, Iterator, Sequence
from typing import Dict, List, Optional, Tuple

import numpy as np

from sglang.jit_kernel.ngram_corpus import get_ngram_corpus_cls
from sglang.srt.speculative.cpp_ngram.external_corpus import SEPARATOR_TOKEN

logger = logging.getLogger(__name__)


# Convenience path for pre-tokenized in-memory documents. The main serving
# startup path loads file-backed corpora through `iter_external_corpus_chunks()`.
def _documents_to_chunks(
    documents: Iterable[Sequence[int]],
    max_tokens: Optional[int] = None,
) -> Iterator[list[int]]:
    total_tokens = 0
    has_previous = False
    for doc in documents:
        if not doc:
            continue
        doc_tokens = list(doc)
        separator_cost = 1 if has_previous else 0
        next_total_tokens = total_tokens + separator_cost + len(doc_tokens)
        if max_tokens is not None and next_total_tokens > max_tokens:
            raise ValueError(
                "External ngram corpus exceeds the configured token limit "
                f"({max_tokens}) after loading {total_tokens} tokens."
            )
        total_tokens = next_total_tokens
        if has_previous:
            yield [SEPARATOR_TOKEN] + doc_tokens
        else:
            yield doc_tokens
        has_previous = True


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
        external_corpus_documents: Optional[Iterable[Sequence[int]]] = None,
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
        self.default_mask = np.ones((1, 1), dtype=np.int64)
        self.draft_token_num = draft_token_num
        self._req_id_to_state_id: Dict[str, int] = {}
        self._next_state_id: int = 0
        self.external_corpus_token_count = 0
        if external_corpus_documents is not None:
            self.load_external_corpus(
                _documents_to_chunks(
                    external_corpus_documents, external_corpus_max_tokens
                )
            )

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

    def load_external_corpus(self, chunks: Iterable[Sequence[int]]) -> int:
        """Load pre-chunked external corpus tokens.

        Callers passing raw chunk iterables must enforce any token budget before
        calling this method. Python-side helpers such as
        `iter_external_corpus_chunks()` and `external_corpus_documents=` handle
        `external_corpus_max_tokens` validation before handing chunks to C++.
        """
        _, loaded_token_count = self._obj.load_external_corpus(chunks)
        self.external_corpus_token_count = loaded_token_count
        return loaded_token_count

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
