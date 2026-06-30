from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Dict, List, Tuple

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

        def insert(self, batch_tokens: List[List[int]]) -> None:
            tokens_flat, offsets = _to_csr(batch_tokens)
            self.async_insert(tokens_flat, offsets)  # type: ignore

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

        def precompute_drafts_stateful_wrapper(
            self,
            state_ids: List[int],
            base_tokens: List[List[int]],
            total_lens: List[int],
            draft_tokens,
            tree_mask,
            bonus_topk: int,
            max_trie_depth: int,
        ) -> Tuple[int, int, int]:
            tokens_flat, offsets = _to_csr(base_tokens)
            state_ids_t = torch.tensor(state_ids, dtype=torch.int64)
            total_lens_t = torch.tensor(total_lens, dtype=torch.int64)
            draft_tokens_t = torch.as_tensor(draft_tokens, dtype=torch.int32).flatten()
            tree_mask_t = torch.as_tensor(tree_mask, dtype=torch.uint8).flatten()
            out_stats = torch.zeros(3, dtype=torch.int64)

            self.precompute_drafts_stateful(  # type: ignore
                state_ids_t,
                tokens_flat,
                offsets,
                total_lens_t,
                draft_tokens_t,
                tree_mask_t,
                bonus_topk,
                max_trie_depth,
                out_stats,
            )
            stats = out_stats.numpy().astype(np.int64).tolist()
            return int(stats[0]), int(stats[1]), int(stats[2])

        def select_precomputed_drafts_stateful_wrapper(
            self,
            state_ids: List[int],
            accept_tokens,
            accept_lens,
            accept_index,
            fallback_tokens: List[List[int]],
            fallback_total_lens: List[int],
        ):
            fallback_tokens_flat, fallback_offsets = _to_csr(fallback_tokens)
            batch_size = len(fallback_tokens)
            d = self._draft_token_num

            state_ids_t = torch.tensor(state_ids, dtype=torch.int64)
            accept_tokens_t = torch.as_tensor(
                accept_tokens, dtype=torch.int32
            ).flatten()
            accept_lens_t = torch.as_tensor(accept_lens, dtype=torch.int64).flatten()
            accept_index_t = torch.as_tensor(accept_index, dtype=torch.int64).flatten()
            fallback_total_lens_t = torch.tensor(fallback_total_lens, dtype=torch.int64)
            out_tokens = torch.zeros(batch_size * d, dtype=torch.int32)
            out_mask = torch.zeros(batch_size * d * d, dtype=torch.uint8)
            out_bonus_hit = torch.zeros(batch_size, dtype=torch.uint8)
            out_cache_hit = torch.zeros(batch_size, dtype=torch.uint8)
            out_stats = torch.zeros(4, dtype=torch.int64)

            self.select_precomputed_drafts_stateful(  # type: ignore
                state_ids_t,
                accept_tokens_t,
                accept_lens_t,
                accept_index_t,
                fallback_tokens_flat,
                fallback_offsets,
                fallback_total_lens_t,
                out_tokens,
                out_mask,
                out_bonus_hit,
                out_cache_hit,
                out_stats,
            )

            return (
                out_tokens.numpy().astype(np.int64),
                out_mask.numpy().astype(np.int64),
                out_bonus_hit.numpy().astype(np.int64).tolist(),
                out_cache_hit.numpy().astype(np.int64).tolist(),
                tuple(int(x) for x in out_stats.numpy().astype(np.int64).tolist()),
            )

        def precomputed_root_bonus_tokens_stateful_wrapper(
            self, state_ids: List[int]
        ) -> List[int]:
            if not state_ids:
                return []
            state_ids_t = torch.tensor(state_ids, dtype=torch.int64)
            out_tokens = torch.full((len(state_ids),), -1, dtype=torch.int32)

            self.precomputed_root_bonus_tokens_stateful(  # type: ignore
                state_ids_t,
                out_tokens,
            )

            return out_tokens.numpy().astype(np.int64).tolist()

        def erase_states(self, state_ids: List[int]) -> None:
            state_ids_t = torch.tensor(state_ids, dtype=torch.int64)
            self.erase_match_state(state_ids_t)  # type: ignore

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
