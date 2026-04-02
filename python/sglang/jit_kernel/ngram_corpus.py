from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_ngram_corpus_module() -> Module:
    return load_jit(
        "ngram_corpus",
        cpp_files=[
            "ngram_corpus/result.cpp",
            "ngram_corpus/trie.cpp",
            "ngram_corpus/ngram.cpp",
            "ngram_corpus/ngram_corpus_ffi.h",
        ],
        cpp_wrappers=[
            ("ngram_create", "&NgramCorpusFfi::create"),
            ("ngram_destroy", "&NgramCorpusFfi::destroy"),
            ("ngram_async_insert", "&NgramCorpusFfi::async_insert"),
            ("ngram_batch_match", "&NgramCorpusFfi::batch_match"),
            ("ngram_synchronize", "&NgramCorpusFfi::synchronize"),
            ("ngram_reset", "&NgramCorpusFfi::reset"),
        ],
    )


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


def ngram_create(
    capacity: int,
    max_trie_depth: int,
    min_bfs_breadth: int,
    max_bfs_breadth: int,
    draft_token_num: int,
    match_type: str,
) -> int:
    mt = _MATCH_TYPE_MAP.get(match_type)
    if mt is None:
        raise ValueError(
            f"Unknown match_type: '{match_type}'. Must be 'BFS' or 'PROB'."
        )
    out_handle = torch.zeros(1, dtype=torch.int64)
    _jit_ngram_corpus_module().ngram_create(
        capacity,
        max_trie_depth,
        min_bfs_breadth,
        max_bfs_breadth,
        draft_token_num,
        mt,
        out_handle,
    )
    return out_handle.item()


def ngram_destroy(handle: int) -> None:
    _jit_ngram_corpus_module().ngram_destroy(handle)


def ngram_async_insert(handle: int, batch_tokens: List[List[int]]) -> None:
    tokens_flat, offsets = _to_csr(batch_tokens)
    _jit_ngram_corpus_module().ngram_async_insert(handle, tokens_flat, offsets)


def ngram_batch_match(
    handle: int,
    batch_tokens: List[List[int]],
    draft_token_num: int,
) -> Tuple[np.ndarray, np.ndarray]:
    tokens_flat, offsets = _to_csr(batch_tokens)
    batch_size = len(batch_tokens)

    out_tokens = torch.zeros(batch_size * draft_token_num, dtype=torch.int32)
    out_mask = torch.zeros(
        batch_size * draft_token_num * draft_token_num, dtype=torch.uint8
    )

    _jit_ngram_corpus_module().ngram_batch_match(
        handle, tokens_flat, offsets, out_tokens, out_mask
    )

    return out_tokens.numpy().astype(np.int64), out_mask.numpy().astype(np.int64)


def ngram_synchronize(handle: int) -> None:
    _jit_ngram_corpus_module().ngram_synchronize(handle)


def ngram_reset(handle: int) -> None:
    _jit_ngram_corpus_module().ngram_reset(handle)
