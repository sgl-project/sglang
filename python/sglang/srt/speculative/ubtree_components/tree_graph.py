# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See LICENSE in this directory for the original license.

"""Fixed-shape CUDA Graph replay for UBTree proposal-tree construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from sglang.srt.speculative.ubtree_components.tree import (
    UBTreeBatch,
    build_ubtree_batch,
)
from sglang.srt.speculative.ubtree_components.vocab import vocab_parallel_topk


class _SingleRankGroup:
    world_size = 1


@dataclass
class _CapturedProposal:
    graph: torch.cuda.CUDAGraph
    hidden: torch.Tensor
    bonus_tokens: torch.Tensor
    output: UBTreeBatch


class UBTreeProposalGraphRunner:
    """Capture selector/top-k and proposal-tree construction as one graph.

    The previous two-runner path copied ``candidate_ids`` and
    ``edge_log_probs`` into a second graph's private buffers.  Capturing the
    producer and consumer together keeps those intermediates at graph-owned
    addresses and removes one replay plus both D2D copies.
    """

    def __init__(
        self,
        *,
        batch_sizes: Iterable[int],
        depth_count: int,
        hidden_size: int,
        hidden_dtype: torch.dtype,
        candidate_k: int,
        vocab_size: int,
        lm_head,
        selector,
        budget: int,
        supertree_width: int,
        depth_bonus: float,
        device: torch.device | str,
        tp_group=None,
    ) -> None:
        self.device = torch.device(device)
        self.depth_count = int(depth_count)
        self.hidden_size = int(hidden_size)
        self.hidden_dtype = hidden_dtype
        self.candidate_k = int(candidate_k)
        self.vocab_size = int(vocab_size)
        self.lm_head = lm_head
        self.selector = selector
        self.budget = int(budget)
        self.supertree_width = int(supertree_width)
        self.depth_bonus = float(depth_bonus)
        self._tp_group = tp_group if tp_group is not None else _SingleRankGroup()
        self._captured: dict[int, _CapturedProposal] = {}
        requested = sorted({int(bs) for bs in batch_sizes if int(bs) > 0})
        if self.device.type == "cuda":
            for batch_size in requested:
                self._captured[batch_size] = self._capture(batch_size)

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        return tuple(self._captured)

    def build(self, hidden: torch.Tensor, bonus_tokens: torch.Tensor) -> UBTreeBatch:
        """Build a proposal without replaying this runner's standalone graph.

        This entry point is used while capturing the proposal as the tail of the
        full draft graph.
        """

        return self._build(hidden, bonus_tokens)

    def _build(self, hidden: torch.Tensor, bonus_tokens: torch.Tensor) -> UBTreeBatch:
        batch_size = int(hidden.shape[0])
        flat_hidden = hidden.reshape(-1, self.hidden_size)
        unary_values, candidate_ids = vocab_parallel_topk(
            flat_hidden,
            self.lm_head,
            k=self.candidate_k,
            vocab_size=self.vocab_size,
            tp_group=self._tp_group,
        )
        unary_values = unary_values.view(batch_size, self.depth_count, self.candidate_k)
        candidate_ids = candidate_ids.view(
            batch_size, self.depth_count, self.candidate_k
        )
        edge_log_probs = self.selector.build_edge_log_probs(
            hidden,
            bonus_tokens,
            candidate_ids,
            unary_candidate_logits=unary_values,
        )
        return build_ubtree_batch(
            anchor_tokens=bonus_tokens,
            candidate_ids=candidate_ids,
            edge_log_probs=edge_log_probs,
            budget=self.budget,
            supertree_width=self.supertree_width,
            depth_bonus=self.depth_bonus,
        )

    def _capture(self, batch_size: int) -> _CapturedProposal:
        hidden = torch.zeros(
            (batch_size, self.depth_count, self.hidden_size),
            dtype=self.hidden_dtype,
            device=self.device,
        )
        bonus_tokens = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        current_stream = torch.cuda.current_stream(self.device)
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(2):
                self._build(hidden, bonus_tokens)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._build(hidden, bonus_tokens)
        current_stream.wait_stream(capture_stream)
        return _CapturedProposal(
            graph=graph, hidden=hidden, bonus_tokens=bonus_tokens, output=output
        )

    def __call__(
        self, *, hidden: torch.Tensor, bonus_tokens: torch.Tensor
    ) -> UBTreeBatch:
        batch_size = int(hidden.shape[0])
        expected_hidden_shape = (batch_size, self.depth_count, self.hidden_size)
        if tuple(hidden.shape) != expected_hidden_shape:
            raise ValueError(f"hidden must have shape {expected_hidden_shape}")
        if tuple(bonus_tokens.shape) != (batch_size,):
            raise ValueError("bonus_tokens must have shape [batch]")

        captured = self._captured.get(batch_size)
        if captured is None:
            return self._build(hidden, bonus_tokens)
        captured.hidden.copy_(hidden)
        captured.bonus_tokens.copy_(bonus_tokens)
        captured.graph.replay()
        return captured.output
