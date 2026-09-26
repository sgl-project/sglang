from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Dict, List, Tuple

import numpy as np
import torch
import tvm_ffi

from sglang.kernels.jit.utils import cache_once, load_jit

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

        def precompute_drafts_dense_host(
            self,
            *,
            base_tokens: List[List[int]],
            total_lens: List[int],
            draft_tokens: np.ndarray,
            tree_mask: np.ndarray,
            bonus_topk: int,
            max_trie_depth: int,
            wide_bonus_ratio: float = 0.5,
        ) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
            batch_size = len(base_tokens)
            d = self._draft_token_num

            (
                tokens_flat,
                offsets,
                total_lens_t,
                draft_tokens_t,
                tree_mask_t,
            ) = self._make_precompute_inputs(
                base_tokens=base_tokens,
                total_lens=total_lens,
                draft_tokens=draft_tokens,
                tree_mask=tree_mask,
            )
            out_bonus_tokens = torch.empty(
                batch_size * d * bonus_topk, dtype=torch.int32
            )
            out_draft_tokens = torch.empty(
                (batch_size * d * bonus_topk * d,), dtype=torch.int32
            )
            out_tree_masks = torch.empty(
                (batch_size * d * bonus_topk * d * d,), dtype=torch.uint8
            )
            out_stats = torch.empty(3, dtype=torch.int64)

            self.precompute_drafts_dense(  # type: ignore
                tokens_flat,
                offsets,
                total_lens_t,
                draft_tokens_t,
                tree_mask_t,
                bonus_topk,
                max_trie_depth,
                wide_bonus_ratio,
                out_bonus_tokens,
                out_draft_tokens,
                out_tree_masks,
                out_stats,
            )
            return (
                tuple(out_stats.tolist()),
                out_bonus_tokens.numpy(),
                out_draft_tokens.numpy().astype(np.int64),
                out_tree_masks.numpy(),
            )

        @staticmethod
        def _make_precompute_inputs(
            base_tokens: List[List[int]],
            total_lens: List[int],
            draft_tokens,
            tree_mask,
        ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            tokens_flat, offsets = _to_csr(base_tokens)
            return (
                tokens_flat,
                offsets,
                torch.tensor(total_lens, dtype=torch.int64),
                torch.as_tensor(draft_tokens, dtype=torch.int32).flatten(),
                torch.as_tensor(tree_mask, dtype=torch.uint8).flatten(),
            )

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
