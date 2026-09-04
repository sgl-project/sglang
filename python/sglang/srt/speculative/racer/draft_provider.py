from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from sglang.srt.speculative.racer.refill_automaton import BinaryRefillRacerAutomaton
from sglang.srt.speculative.racer.stats import InstrumentedRacerAutomaton


class RacerDraftProvider:
    """Per-request RACER proposal state with the NgramCorpus batch_get contract."""

    def __init__(
        self,
        *,
        draft_token_num: int,
        ngram: int = 10,
        topk: int = 9,
        max_nodes: int = 10_000,
        stats_enabled: bool = False,
    ):
        self.draft_token_num = int(draft_token_num)
        self.ngram = int(ngram)
        self.topk = int(topk)
        self.max_nodes = int(max_nodes)
        self.stats_enabled = bool(stats_enabled)
        self._states: dict[object, BinaryRefillRacerAutomaton] = {}
        self._last_batch_stats: list[dict[str, int | float]] = []

    def _state(self, req_id: object) -> BinaryRefillRacerAutomaton:
        state = self._states.get(req_id)
        if state is None:
            cls = (
                InstrumentedRacerAutomaton
                if self.stats_enabled
                else BinaryRefillRacerAutomaton
            )
            state = cls(
                ngram=self.ngram,
                topk=self.topk,
                max_nodes=self.max_nodes,
            )
            self._states[req_id] = state
        return state

    def synchronize(self) -> None:
        return None

    def reset(self) -> None:
        self._states.clear()
        self._last_batch_stats.clear()

    def erase_match_state(self, req_ids: Sequence[object]) -> None:
        for rid in req_ids:
            self._states.pop(rid, None)

    def batch_put(self, batch_tokens) -> None:
        return None

    def batch_get(
        self,
        req_ids: Sequence[object],
        batch_tokens: Sequence[Sequence[int]],
        total_lens: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        del total_lens
        all_tokens: list[int] = []
        all_masks: list[bool] = []
        batch_stats: list[dict[str, int | float]] = []

        for rid, context in zip(req_ids, batch_tokens):
            context = [int(x) for x in context]
            if not context:
                raise RuntimeError("RACER requires a non-empty decode context.")
            root_token = context[-1]
            state = self._state(rid)

            start = time.perf_counter() if self.stats_enabled else 0.0
            state.sync_history(context[:-1])
            draft_tokens, tree_mask = state.retrieve(
                root_token, self.draft_token_num
            )
            if self.stats_enabled:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                stats = state.last_proposal_stats.as_dict()
                stats["proposal_ms"] = elapsed_ms
                batch_stats.append(stats)

            if len(draft_tokens) != self.draft_token_num:
                raise RuntimeError(
                    f"RACER returned {len(draft_tokens)} draft tokens, "
                    f"expected {self.draft_token_num}."
                )
            if len(tree_mask) != self.draft_token_num:
                raise RuntimeError(
                    f"RACER returned {len(tree_mask)} mask rows, "
                    f"expected {self.draft_token_num}."
                )
            all_tokens.extend(draft_tokens)
            for row in tree_mask:
                if len(row) != self.draft_token_num:
                    raise RuntimeError(
                        f"RACER returned a mask row of width {len(row)}, "
                        f"expected {self.draft_token_num}."
                    )
                all_masks.extend(bool(x) for x in row)

        if self.stats_enabled:
            self._last_batch_stats = batch_stats
        return np.asarray(all_tokens, dtype=np.int64), np.asarray(
            all_masks, dtype=np.bool_
        )

    def consume_last_batch_stats(self) -> list[dict[str, int | float]]:
        if not self.stats_enabled:
            return []
        stats = self._last_batch_stats
        self._last_batch_stats = []
        return stats

    def seed_prompt_logits(
        self,
        req_ids: Sequence[object],
        prompt_tokens: np.ndarray,
        topk_ids: np.ndarray,
        prompt_lens: Sequence[int],
    ) -> None:
        """Warm each request's TokenBin from flattened prompt-position logits.

        ``prompt_tokens`` and ``topk_ids`` follow SGLang's flattened extend
        layout.  For a prompt token x_i, row i contains the target model's
        top-k next-token predictions p(x_{i+1} | x_{<=i}), matching RACER's
        original prompt-time ``Automaton::update(input_ids, adj_topk)`` use.
        Repeated token ids intentionally keep the latest row, as in TokenBin.
        """

        prompt_tokens = np.asarray(prompt_tokens).reshape(-1)
        topk_ids = np.asarray(topk_ids)
        if topk_ids.ndim != 2:
            raise ValueError(f"RACER prompt top-k must be rank-2, got {topk_ids.shape=}")

        lengths = [int(x) for x in prompt_lens]
        if len(lengths) != len(req_ids):
            raise ValueError(
                f"RACER prompt warm-up length mismatch: {len(req_ids)=}, "
                f"{len(lengths)=}"
            )
        total = sum(lengths)
        if total != len(prompt_tokens) or total != topk_ids.shape[0]:
            raise ValueError(
                "RACER prompt warm-up flattened shape mismatch: "
                f"{total=}, tokens={len(prompt_tokens)}, rows={topk_ids.shape[0]}"
            )

        offset = 0
        for rid, length in zip(req_ids, lengths):
            end = offset + length
            if length > 0:
                self._state(rid).update_logits(
                    prompt_tokens[offset:end].tolist(),
                    topk_ids[offset:end].tolist(),
                )
            offset = end

    def update_logits(
        self,
        req_ids: Sequence[object],
        draft_tokens: np.ndarray,
        topk_ids: np.ndarray,
    ) -> None:
        bs = len(req_ids)
        k = self.draft_token_num
        draft_tokens = np.asarray(draft_tokens).reshape(bs, k)
        topk_ids = np.asarray(topk_ids).reshape(bs, k, -1)
        for b, rid in enumerate(req_ids):
            self._state(rid).update_logits(
                draft_tokens[b].tolist(), topk_ids[b].tolist()
            )

    def load_external_corpus_named(self, *args, **kwargs):
        raise NotImplementedError("RACER does not support external NGRAM corpora.")

    def commit_external_corpus_load(self, *args, **kwargs) -> None:
        raise NotImplementedError("RACER does not support external NGRAM corpora.")

    def remove_external_corpus(self, *args, **kwargs) -> None:
        raise NotImplementedError("RACER does not support external NGRAM corpora.")

    def list_external_corpora(self) -> dict[str, int]:
        return {}
