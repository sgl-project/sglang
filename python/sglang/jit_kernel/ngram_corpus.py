from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Dict, List, Tuple

import numpy as np
import torch
import tvm_ffi

from sglang.jit_kernel.utils import cache_once, load_jit

_MATCH_TYPE_MAP = {"BFS": 0, "PROB": 1}
_TRIE_MODE_MAP = {"global": 0, "request": 1}


def _to_csr(batch_tokens: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = []
    offsets = [0]
    for seq in batch_tokens:
        flat.extend(seq)
        offsets.append(len(flat))
    tokens_flat = torch.tensor(flat, dtype=torch.int32)
    offsets_t = torch.tensor(offsets, dtype=torch.int64)
    return tokens_flat, offsets_t


@cache_once
def get_ngram_corpus_cls():
    module = load_jit(
        "ngram_corpus",
        cpp_files=[
            "ngram_corpus/result.cpp",
            "ngram_corpus/trie.cpp",
            "ngram_corpus/suffix_automaton.cpp",
            "ngram_corpus/ngram.cpp",
            "ngram_corpus/ngram_corpus_ffi.cpp",
        ],
        header_only=False,
    )
    module.register_once()

    @tvm_ffi.register_object("sgl.NgramCorpus")
    class NgramCorpusFFI(tvm_ffi.Object):
        __slots__ = ("__dict__",)

        def __init__(
            self,
            capacity: int,
            max_trie_depth: int,
            min_bfs_breadth: int,
            max_bfs_breadth: int,
            draft_token_num: int,
            match_type: str,
            trie_mode: str = "global",
            external_corpus_max_tokens: int = 10000000,
            trie_source_prior: float = 0.0,
            match_specificity_weight: float = 0.7,
            match_confidence_weight: float = 0.3,
        ) -> None:
            mt = _MATCH_TYPE_MAP.get(match_type)
            if mt is None:
                raise ValueError(
                    f"Unknown match_type: '{match_type}'. Must be 'BFS' or 'PROB'."
                )
            tm = _TRIE_MODE_MAP.get(trie_mode)
            if tm is None:
                raise ValueError(
                    f"Unknown trie_mode: '{trie_mode}'. Must be 'global' or 'request'."
                )
            self.__ffi_init__(
                capacity,
                max_trie_depth,
                min_bfs_breadth,
                max_bfs_breadth,
                draft_token_num,
                mt,
                tm,
                external_corpus_max_tokens,
                trie_source_prior,
                match_specificity_weight,
                match_confidence_weight,
            )
            self._draft_token_num = draft_token_num
            self._trie_mode = trie_mode

        def insert(
            self, batch_tokens: List[List[int]], state_ids: List[int] | None = None
        ) -> None:
            tokens_flat, offsets = _to_csr(batch_tokens)
            if state_ids is None:
                if self._trie_mode == "request":
                    raise ValueError(
                        "request trie mode requires state_ids for insert"
                    )
                state_ids_t = torch.full(
                    (len(batch_tokens),), -1, dtype=torch.int64
                )
            else:
                state_ids_t = torch.tensor(state_ids, dtype=torch.int64)
            self.async_insert(state_ids_t, tokens_flat, offsets)  # type: ignore

        def match_stateful(
            self,
            state_ids: List[int],
            batch_tokens: List[List[int]],
            total_lens: List[int],
        ) -> Tuple[np.ndarray, np.ndarray]:
            tokens_flat, offsets = _to_csr(batch_tokens)
            batch_size = len(batch_tokens)
            d = self._draft_token_num

            state_ids_t = torch.tensor(state_ids, dtype=torch.int64)
            total_lens_t = torch.tensor(total_lens, dtype=torch.int64)
            out_tokens = torch.zeros(batch_size * d, dtype=torch.int32)
            out_mask = torch.zeros(batch_size * d * d, dtype=torch.uint8)

            self.batch_match_stateful(  # type: ignore
                state_ids_t, tokens_flat, offsets, total_lens_t, out_tokens, out_mask
            )

            return out_tokens.numpy().astype(np.int64), out_mask.numpy().astype(
                np.int64
            )

        def erase_states(self, state_ids: List[int]) -> None:
            state_ids_t = torch.tensor(state_ids, dtype=torch.int64)
            self.erase_match_state(state_ids_t)  # type: ignore

        def erase_request_states(self, state_ids: List[int]) -> None:
            state_ids_t = torch.tensor(state_ids, dtype=torch.int64)
            self.erase_request_state(state_ids_t)  # type: ignore

        def load_external_corpus_named(
            self, corpus_id: str, chunks: Iterable[Sequence[int]], max_tokens: int
        ) -> Tuple[int, int]:
            self.start_external_corpus_load()  # type: ignore
            chunk_count = 0
            loaded_token_count = 0
            try:
                for chunk in chunks:
                    tokens_t = torch.tensor(list(chunk), dtype=torch.int32)
                    if loaded_token_count + len(tokens_t) > max_tokens:
                        raise ValueError(
                            "External ngram corpus exceeds the remaining token budget "
                            f"({max_tokens}) after loading {loaded_token_count} tokens."
                        )
                    loaded_token_count += len(tokens_t)
                    self.append_external_corpus_tokens(tokens_t)  # type: ignore
                    chunk_count += 1
                self.finish_external_corpus_load(corpus_id)  # type: ignore
            except Exception:
                self.cancel_external_corpus_load()  # type: ignore
                raise
            return chunk_count, loaded_token_count

        def remove_corpus(self, corpus_id: str) -> None:
            self.remove_external_corpus(corpus_id)  # type: ignore

        def list_corpora(self) -> Dict[str, int]:
            result = self.list_external_corpora()  # type: ignore
            if not result:
                return {}
            out: Dict[str, int] = {}
            for line in result.split("\n"):
                corpus_id, token_count = line.split("\t", 1)
                out[corpus_id] = int(token_count)
            return out

    return NgramCorpusFFI
