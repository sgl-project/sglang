from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import tvm_ffi

from sglang.jit_kernel.utils import cache_once, load_jit

_MATCH_TYPE_MAP = {"BFS": 0, "PROB": 1}


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
        ) -> None:
            mt = _MATCH_TYPE_MAP.get(match_type)
            if mt is None:
                raise ValueError(
                    f"Unknown match_type: '{match_type}'. Must be 'BFS' or 'PROB'."
                )
            self.__ffi_init__(
                capacity,
                max_trie_depth,
                min_bfs_breadth,
                max_bfs_breadth,
                draft_token_num,
                mt,
            )
            self._draft_token_num = draft_token_num

        def insert(self, batch_tokens: List[List[int]]) -> None:
            tokens_flat, offsets = _to_csr(batch_tokens)
            self.async_insert(tokens_flat, offsets)  # type: ignore

        def match(
            self,
            batch_tokens: List[List[int]],
        ) -> Tuple[np.ndarray, np.ndarray]:
            tokens_flat, offsets = _to_csr(batch_tokens)
            batch_size = len(batch_tokens)
            d = self._draft_token_num

            out_tokens = torch.zeros(batch_size * d, dtype=torch.int32)
            out_mask = torch.zeros(batch_size * d * d, dtype=torch.uint8)

            self.batch_match(tokens_flat, offsets, out_tokens, out_mask)  # type: ignore

            return out_tokens.numpy().astype(np.int64), out_mask.numpy().astype(
                np.int64
            )

    return NgramCorpusFFI
