# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See LICENSE in this directory for the original license.

"""Full CUDA Graph replay of the UBTree draft and proposal tail."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Iterable

import torch

from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_capture_stream,
)
from sglang.srt.models.dflash import DFlashDraftModel


@dataclass
class _CapturedFullDraftBucket:
    input_ids: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    seq_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    forward_batch: object
    graph: torch.cuda.CUDAGraph
    output: LogitsProcessorOutput | None
    tail_output: object | None


class UBTreeDraftSegmentGraphRunner:
    """Capture the DFlash backbone for exact decode batch sizes."""

    def __init__(
        self,
        *,
        model,
        input_embedding,
        attn_backend,
        batch_sizes: Iterable[int],
        block_size: int,
        hidden_size: int,
        device: torch.device | str,
    ) -> None:
        self.model = model
        self.input_embedding = input_embedding
        self.attn_backend = attn_backend
        self.block_size = int(block_size)
        self.hidden_size = int(hidden_size)
        self.device = torch.device(device)
        self._full_captured: dict[int, _CapturedFullDraftBucket] = {}
        self._requested_batch_sizes = tuple(
            sorted({int(bs) for bs in batch_sizes if int(bs) > 0})
        )
        self._capture_tail = None

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        return self._requested_batch_sizes

    def attach_capture_tail(self, capture_tail) -> None:
        """Attach fixed-shape work to the full draft CUDA Graph tail.

        The callback receives the draft model output, graph-owned input ids, and
        captured forward batch. Its tensor output remains owned by the graph and
        is refreshed on every replay. Attach it before the first full capture.
        """

        if self._full_captured:
            raise RuntimeError("draft capture tail must be attached before capture")
        self._capture_tail = capture_tail

    def tail_output(self, batch_size: int):
        captured = self._full_captured.get(int(batch_size))
        return None if captured is None else captured.tail_output

    def can_run(self, batch_size: int) -> bool:
        return int(batch_size) in self._requested_batch_sizes

    def _run_full_model(self, captured, forward_batch):
        self.attn_backend.init_forward_metadata_in_graph(forward_batch)
        input_hidden = self.input_embedding(captured.input_ids).reshape(
            captured.input_ids.numel(), self.hidden_size
        )
        output = DFlashDraftModel.forward(
            self.model,
            input_ids=captured.input_ids,
            positions=captured.positions,
            forward_batch=forward_batch,
            input_embeds=input_hidden,
        )
        if self._capture_tail is not None:
            captured.tail_output = self._capture_tail(
                output, captured.input_ids, forward_batch
            )
        return output

    def _capture_full(
        self, *, input_ids, positions, forward_batch
    ) -> _CapturedFullDraftBucket:
        batch_size = int(forward_batch.batch_size)
        rows = batch_size * self.block_size
        captured_input_ids = torch.empty(rows, dtype=torch.int64, device=self.device)
        captured_positions = torch.empty(rows, dtype=torch.int64, device=self.device)
        out_cache_loc = torch.empty(rows, dtype=torch.int64, device=self.device)
        captured_input_ids.copy_(input_ids)
        captured_positions.copy_(positions)
        out_cache_loc.copy_(forward_batch.out_cache_loc)
        seq_lens = forward_batch.seq_lens.clone()
        req_pool_indices = forward_batch.req_pool_indices.clone()

        captured_forward_batch = copy(forward_batch)
        captured_forward_batch.input_ids = captured_input_ids
        captured_forward_batch.positions = captured_positions
        captured_forward_batch.out_cache_loc = out_cache_loc
        captured_forward_batch.seq_lens = seq_lens
        captured_forward_batch.req_pool_indices = req_pool_indices
        captured_forward_batch.input_embeds = None

        placeholder = _CapturedFullDraftBucket(
            input_ids=captured_input_ids,
            positions=captured_positions,
            out_cache_loc=out_cache_loc,
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            forward_batch=captured_forward_batch,
            graph=torch.cuda.CUDAGraph(),
            output=None,
            tail_output=None,
        )
        current_stream = torch.cuda.current_stream(self.device)
        # Use SGLang's distributed graph-capture context instead of a raw CUDA
        # stream.  Besides synchronizing the shared capture stream, this enables
        # graph-safe TP communicators for the vocab-parallel embedding and the
        # proposal Top-K all-gathers.  All TP ranks enter this path in the same
        # order for an exact decode batch size.
        with (
            graph_capture(
                stream=get_or_create_global_graph_capture_stream()
            ) as capture_context,
            forward_context(ForwardContext(attn_backend=self.attn_backend)),
        ):
            capture_stream = capture_context.stream
            self.attn_backend.init_forward_metadata_out_graph(
                captured_forward_batch, in_capture=True
            )
            captured_forward_batch.mark_forward_metadata_ready()
            for _ in range(2):
                self._run_full_model(placeholder, captured_forward_batch)
            with torch.cuda.graph(placeholder.graph, stream=capture_stream):
                placeholder.output = self._run_full_model(
                    placeholder, captured_forward_batch
                )
        current_stream.wait_stream(capture_stream)
        return placeholder

    def _replay_full(self, *, input_ids, positions, forward_batch):
        batch_size = int(forward_batch.batch_size)
        captured = self._full_captured.get(batch_size)
        if captured is None:
            captured = self._capture_full(
                input_ids=input_ids, positions=positions, forward_batch=forward_batch
            )
            self._full_captured[batch_size] = captured
            # Capture establishes graph-owned output buffers, but tail builders
            # may initialize those buffers before their recorded kernels have
            # produced the first usable values. Replay once so the first caller
            # observes the same completed graph state as subsequent callers.
            captured.graph.replay()
            assert captured.output is not None
            return captured.output
        captured.input_ids.copy_(input_ids)
        captured.positions.copy_(positions)
        captured.out_cache_loc.copy_(forward_batch.out_cache_loc)
        # The scheduler may replace these tensors between decode rounds or
        # requests. Both the model metadata and proposal tail must read stable
        # graph-owned addresses containing this replay's values.
        captured.seq_lens.copy_(forward_batch.seq_lens)
        captured.req_pool_indices.copy_(forward_batch.req_pool_indices)
        self.attn_backend.init_forward_metadata_out_graph(forward_batch)
        captured.graph.replay()
        assert captured.output is not None
        return captured.output

    def __call__(self, *, input_ids, positions, forward_batch):
        batch_size = int(forward_batch.batch_size)
        if not self.can_run(batch_size):
            return None
        expected_rows = batch_size * self.block_size
        expected_shape = (expected_rows,)
        if tuple(input_ids.shape) != expected_shape:
            raise ValueError(f"input_ids must have shape {expected_shape}")
        if tuple(positions.shape) != expected_shape:
            raise ValueError(f"positions must have shape {expected_shape}")
        if tuple(forward_batch.out_cache_loc.shape) != expected_shape:
            raise ValueError(f"out_cache_loc must have shape {expected_shape}")
        return self._replay_full(
            input_ids=input_ids, positions=positions, forward_batch=forward_batch
        )


__all__ = ["UBTreeDraftSegmentGraphRunner"]
