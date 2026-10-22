# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See LICENSE in this directory for the original license.

"""CUDA Graph replay for the UBTree DFlash draft backbone.

The production path captures embedding, transformer layers, and FlashInfer
attention together.  A segmented non-attention path remains for unit-level
coverage and backends that do not provide graph metadata.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Iterable

import torch

from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner_utils import (
    get_or_create_global_graph_capture_stream,
)
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.models.utils import apply_qk_norm


@dataclass
class _CapturedDraftBucket:
    input_ids: torch.Tensor
    positions: torch.Tensor
    pre_graph: torch.cuda.CUDAGraph
    pre_output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    bridge_inputs: list[torch.Tensor]
    bridge_graphs: list[torch.cuda.CUDAGraph]
    bridge_outputs: list[tuple[torch.Tensor, ...]]


@dataclass
class _CapturedFullDraftBucket:
    input_ids: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
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
        attn_backend=None,
        batch_sizes: Iterable[int],
        block_size: int,
        hidden_size: int,
        hidden_dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.model = model
        self.input_embedding = input_embedding
        self.attn_backend = attn_backend
        self.block_size = int(block_size)
        self.hidden_size = int(hidden_size)
        self.hidden_dtype = hidden_dtype
        self.device = torch.device(device)
        self._captured: dict[int, _CapturedDraftBucket] = {}
        self._full_captured: dict[int, _CapturedFullDraftBucket] = {}
        self._requested_batch_sizes = tuple(
            sorted({int(bs) for bs in batch_sizes if int(bs) > 0})
        )
        self._capture_tail = None
        if self.device.type == "cuda" and self.attn_backend is None:
            for batch_size in self._requested_batch_sizes:
                self._captured[batch_size] = self._capture(batch_size)

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

    @staticmethod
    def _prepare_attention(layer, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(hidden_states, residual)
        attn = layer.self_attn
        qkv, _ = attn.qkv_proj(hidden_states)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q, k = apply_qk_norm(q, k, attn.q_norm, attn.k_norm, attn.head_dim)
        q, k = attn.rotary_emb(positions, q, k)
        return q, k, v, residual

    def _finish_attention(self, layer_idx, positions, attn_output, residual):
        layer = self.model.layers[layer_idx]
        attn_output = layer.self_attn.apply_attention_output(attn_output, residual)
        hidden_states, _ = layer.self_attn.o_proj(attn_output)
        hidden_states, residual = layer.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = layer.mlp(hidden_states)
        if layer_idx + 1 == len(self.model.layers):
            hidden_states, _ = self.model.norm(hidden_states, residual)
            return (hidden_states,)
        return self._prepare_attention(
            self.model.layers[layer_idx + 1], positions, hidden_states, residual
        )

    def _capture(self, batch_size: int) -> _CapturedDraftBucket:
        rows = batch_size * self.block_size
        input_ids = torch.zeros(rows, dtype=torch.int64, device=self.device)
        positions = torch.zeros(rows, dtype=torch.int64, device=self.device)
        current_stream = torch.cuda.current_stream(self.device)
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(current_stream)
        bridge_inputs: list[torch.Tensor] = []
        bridge_graphs: list[torch.cuda.CUDAGraph] = []
        bridge_outputs: list[tuple[torch.Tensor, ...]] = []
        with torch.cuda.stream(capture_stream):
            for _ in range(2):
                input_hidden = self.input_embedding(input_ids).reshape(
                    rows, self.hidden_size
                )
                self._prepare_attention(
                    self.model.layers[0], positions, input_hidden, None
                )
            pre_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(pre_graph):
                input_hidden = self.input_embedding(input_ids).reshape(
                    rows, self.hidden_size
                )
                pre_output = self._prepare_attention(
                    self.model.layers[0], positions, input_hidden, None
                )

            residual = pre_output[3]
            for layer_idx, layer in enumerate(self.model.layers):
                attn_input = torch.zeros(
                    (rows, layer.self_attn.q_size),
                    dtype=self.hidden_dtype,
                    device=self.device,
                )
                for _ in range(2):
                    self._finish_attention(layer_idx, positions, attn_input, residual)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self._finish_attention(
                        layer_idx, positions, attn_input, residual
                    )
                bridge_inputs.append(attn_input)
                bridge_graphs.append(graph)
                bridge_outputs.append(output)
                if layer_idx + 1 < len(self.model.layers):
                    residual = output[3]
        current_stream.wait_stream(capture_stream)
        return _CapturedDraftBucket(
            input_ids=input_ids,
            positions=positions,
            pre_graph=pre_graph,
            pre_output=pre_output,
            bridge_inputs=bridge_inputs,
            bridge_graphs=bridge_graphs,
            bridge_outputs=bridge_outputs,
        )

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

        captured_forward_batch = copy(forward_batch)
        captured_forward_batch.input_ids = captured_input_ids
        captured_forward_batch.positions = captured_positions
        captured_forward_batch.out_cache_loc = out_cache_loc
        captured_forward_batch.input_embeds = None

        placeholder = _CapturedFullDraftBucket(
            input_ids=captured_input_ids,
            positions=captured_positions,
            out_cache_loc=out_cache_loc,
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
        if self.attn_backend is not None:
            if tuple(forward_batch.out_cache_loc.shape) != expected_shape:
                raise ValueError(f"out_cache_loc must have shape {expected_shape}")
            return self._replay_full(
                input_ids=input_ids, positions=positions, forward_batch=forward_batch
            )

        captured = self._captured.get(batch_size)
        if captured is None:
            return None
        captured.input_ids.copy_(input_ids)
        captured.positions.copy_(positions)
        captured.pre_graph.replay()
        q, k, v, residual = captured.pre_output
        for layer_idx, layer in enumerate(self.model.layers):
            attn_output = layer.self_attn.attn(q, k, v, forward_batch)
            captured.bridge_inputs[layer_idx].copy_(attn_output)
            captured.bridge_graphs[layer_idx].replay()
            output = captured.bridge_outputs[layer_idx]
            if layer_idx + 1 == len(self.model.layers):
                hidden_states = output[0]
            else:
                q, k, v, residual = output
        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )


__all__ = ["UBTreeDraftSegmentGraphRunner"]
