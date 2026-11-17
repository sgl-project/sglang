from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np
import torch
import tvm_ffi

from sglang.kernels.jit.utils import cache_once, load_jit

_MATCH_TYPE_MAP = {"BFS": 0, "PROB": 1}


def _to_csr(batch_tokens: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    flat = []
    offsets = [0]
    for seq in batch_tokens:
        flat.extend(seq)
        offsets.append(len(flat))
    tokens_flat = torch.tensor(flat, dtype=torch.int32)
    offsets_t = torch.tensor(offsets, dtype=torch.int64)
    return tokens_flat, offsets_t


def namespace_key(namespace: str | None) -> int:
    """Map a cache namespace seed to the uint64 trie root key.

    None (or empty) is the default unnamespaced root (0). Any other string is
    hashed to a 64-bit key; 0 is remapped so it cannot collide with the
    default root.
    """
    if not namespace:
        return 0
    # Hash any seed or raw id so callers can pass storage_namespace_seed()
    # (hex) or another stable string. Torch stores signed int64; keep the
    # same bits as a C++ uint64_t key.
    value = int(hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16], 16)
    if value == 0:
        return 1
    if value >= 2**63:
        value -= 2**64
    return value


def _namespace_keys(
    batch_size: int, namespaces: list[str | None] | None
) -> torch.Tensor:
    if namespaces is None:
        return torch.zeros(batch_size, dtype=torch.int64)
    if len(namespaces) != batch_size:
        raise ValueError("namespaces must be omitted or match the batch size")
    return torch.tensor([namespace_key(ns) for ns in namespaces], dtype=torch.int64)


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
            external_sam_budget: int = 0,
            external_corpus_max_tokens: int = 10000000,
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
                external_sam_budget,
                external_corpus_max_tokens,
            )
            self._draft_token_num = draft_token_num

        def insert(
            self,
            batch_tokens: list[list[int]],
            namespaces: list[str | None] | None = None,
        ) -> None:
            tokens_flat, offsets = _to_csr(batch_tokens)
            ns_keys = _namespace_keys(len(batch_tokens), namespaces)
            self.async_insert(tokens_flat, offsets, ns_keys)  # type: ignore

        def match_stateful(
            self,
            state_ids: list[int],
            batch_tokens: list[list[int]],
            total_lens: list[int],
            namespaces: list[str | None] | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
            tokens_flat, offsets = _to_csr(batch_tokens)
            batch_size = len(batch_tokens)
            d = self._draft_token_num

            state_ids_t = torch.tensor(state_ids, dtype=torch.int64)
            total_lens_t = torch.tensor(total_lens, dtype=torch.int64)
            ns_keys = _namespace_keys(batch_size, namespaces)
            out_tokens = torch.zeros(batch_size * d, dtype=torch.int32)
            out_mask = torch.zeros(batch_size * d * d, dtype=torch.uint8)

            self.batch_match_stateful(  # type: ignore
                state_ids_t,
                tokens_flat,
                offsets,
                total_lens_t,
                ns_keys,
                out_tokens,
                out_mask,
            )

            return out_tokens.numpy().astype(np.int64), out_mask.numpy().astype(
                np.int64
            )

        def erase_states(self, state_ids: list[int]) -> None:
            state_ids_t = torch.tensor(state_ids, dtype=torch.int64)
            self.erase_match_state(state_ids_t)  # type: ignore

        def load_external_corpus_named(
            self, corpus_id: str, chunks: Iterable[Sequence[int]], max_tokens: int
        ) -> tuple[int, int]:
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

        def list_corpora(self) -> dict[str, int]:
            result = self.list_external_corpora()  # type: ignore
            if not result:
                return {}
            out: dict[str, int] = {}
            for line in result.split("\n"):
                corpus_id, token_count = line.split("\t", 1)
                out[corpus_id] = int(token_count)
            return out

    return NgramCorpusFFI
