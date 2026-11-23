# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Decode CUDA graphs for all-beam batches.

An all-beam decode batch runs the beam fast paths its server has on: the compact
trie head, shared-context attention, or both. Either one keeps the batch off the
ordinary decode graphs. The runner here captures a second graph per size that
holds those paths, and replays it with the batch's trie levels and shared-context
metadata staged into the buffers the graph reads.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.beam_search import shared_context_attention
from sglang.srt.beam_search.forward_inputs import BeamForwardInputs
from sglang.srt.beam_search.shared_context_attention import (
    BeamAttentionBucket,
    BeamAttentionMetadata,
)
from sglang.srt.beam_search.trie_config import get_beam_trie_config
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Graph key label of the beam graphs. It sits where the LoRA label does, and a
# server with LoRA captures no beam graph.
BEAM_GRAPH_VARIANT = "beam"


class SharedContextCapacity(msgspec.Struct, frozen=True):
    """What the shared-context beam graphs hold per beam group."""

    # The one beam width the graphs serve.
    beam_width: int
    # Prompt tokens: the model's context length, so every prompt fits.
    max_context: int
    # Generated tokens a decode step attends. The kernel unrolls its decode loop
    # this many times at compile time, so every replay pays for all of them.
    max_decode: int


def _graph_max_new_tokens(model_config: ModelConfig) -> Optional[int]:
    """Tokens a beam request generates at most inside a shared-context graph;
    None when nothing bounds them."""
    trie_config = get_beam_trie_config(model_config)
    if trie_config is not None:
        # Admission pins a trie request's max_new_tokens to the codebook count.
        return trie_config.num_codebooks
    max_new_tokens = envs.SGLANG_BEAM_DECODE_GRAPH_MAX_NEW_TOKENS.get()
    if max_new_tokens is not None and max_new_tokens < 2:
        raise ValueError(
            "SGLANG_BEAM_DECODE_GRAPH_MAX_NEW_TOKENS must be at least 2 (the first "
            f"token comes from the prefill), got {max_new_tokens}."
        )
    return max_new_tokens


def _shared_context_capacity(
    model_runner: ModelRunner,
) -> Optional[SharedContextCapacity]:
    """None when all-beam batches run ordinary attention, or when their
    shared-context attention gets no graph; the reason is logged."""
    if not (
        shared_context_attention.enabled()
        and shared_context_attention.supports_shared_context(model_runner)
    ):
        return None
    model_config = model_runner.model_config
    max_new_tokens = _graph_max_new_tokens(model_config)
    reason = None
    if model_runner.lora_manager is not None:
        reason = "LoRA is enabled"
    elif max_new_tokens is None:
        reason = (
            "the model bundles no trie and "
            "SGLANG_BEAM_DECODE_GRAPH_MAX_NEW_TOKENS is not set"
        )
    if reason is not None:
        logger.warning(
            "SGLANG_ENABLE_BEAM_DECODE_GRAPH is set, but shared-context beam "
            "batches get no decode graph (%s); they run eagerly.",
            reason,
        )
        return None
    return SharedContextCapacity(
        beam_width=envs.SGLANG_BEAM_DECODE_GRAPH_WIDTH.get(),
        max_context=model_config.context_len,
        # The step that generates token n attends the n - 1 tokens before it.
        max_decode=max_new_tokens - 1,
    )


class BeamGraphInputs:
    """The buffers every beam graph reads, sized for the largest one; a graph of
    fewer rows reads their leading part."""

    def __init__(
        self,
        *,
        max_rows: int,
        device,
        trie_head: bool,
        shared_context: Optional[SharedContextCapacity],
    ):
        # [max_rows] int64 SID codebook depth per row; None under the dense LM head.
        self.levels = (
            torch.zeros(max_rows, dtype=torch.int64, device=device)
            if trie_head
            else None
        )
        self.shared_context = shared_context
        # Rows of the graphs captured so far.
        self.captured_rows: set[int] = set()
        self._bucket: Optional[BeamAttentionBucket] = None
        if shared_context is not None:
            width = shared_context.beam_width
            groups = max_rows // width
            int32 = {"dtype": torch.int32, "device": device}
            # Captured with one prompt token and max_decode steps per group, on KV
            # slot 0; every replay stages the batch's values first.
            self._bucket = BeamAttentionBucket(
                row_indices=torch.arange(groups * width, **int32).view(groups, width),
                context_slots=torch.zeros(groups * shared_context.max_context, **int32),
                context_cu_seqlens=torch.arange(groups + 1, **int32),
                decode_slots=torch.zeros(
                    (groups, width * shared_context.max_decode), **int32
                ),
                beam_width=width,
                decode_lens=torch.full((groups,), shared_context.max_decode, **int32),
            )

    def supports_rows(self, rows: int) -> bool:
        """With shared-context attention a graph holds whole groups of one width."""
        return self.shared_context is None or rows % self.shared_context.beam_width == 0

    def for_capture(self, rows: int) -> BeamForwardInputs:
        self.captured_rows.add(rows)
        return BeamForwardInputs(
            trie_levels=None if self.levels is None else self.levels[:rows],
            attention_metadata=self._metadata_for_capture(rows),
        )

    def _metadata_for_capture(self, rows: int) -> Optional[BeamAttentionMetadata]:
        if self._bucket is None:
            return None
        groups = rows // self._bucket.beam_width
        bucket = BeamAttentionBucket(
            row_indices=self._bucket.row_indices[:groups],
            context_slots=self._bucket.context_slots,
            context_cu_seqlens=self._bucket.context_cu_seqlens[: groups + 1],
            decode_slots=self._bucket.decode_slots[:groups],
            beam_width=self._bucket.beam_width,
            decode_lens=self._bucket.decode_lens[:groups],
        )
        return BeamAttentionMetadata(buckets=[bucket], num_rows=rows)

    def can_replay(self, beam: BeamForwardInputs, *, captured_rows: int) -> bool:
        if captured_rows not in self.captured_rows:
            return False
        metadata, capacity = beam.attention_metadata, self.shared_context
        if metadata is None or capacity is None:
            # Ordinary attention pads like any decode batch.
            return metadata is None and capacity is None
        return (
            metadata.num_rows == captured_rows
            and len(metadata.buckets) == 1
            and metadata.buckets[0].beam_width == capacity.beam_width
            and metadata.buckets[0].decode_slots.shape[1]
            <= capacity.beam_width * capacity.max_decode
        )

    def stage(self, beam: BeamForwardInputs) -> None:
        """Copy a batch that can_replay accepted. Slots past the staged lengths
        and padding rows keep an earlier batch's values: the kernel reads no slot
        past a length, and a stale level is a valid depth."""
        # A server runs the compact head for every all-beam batch or for none, so
        # a batch carries levels exactly when the graphs read them.
        if self.levels is not None:
            self.levels[: beam.trie_levels.shape[0]].copy_(beam.trie_levels)
        if beam.attention_metadata is None:
            return
        source, static = beam.attention_metadata.buckets[0], self._bucket
        groups = source.row_indices.shape[0]
        static.row_indices[:groups].copy_(source.row_indices)
        static.context_slots[: source.context_slots.numel()].copy_(source.context_slots)
        static.context_cu_seqlens[: groups + 1].copy_(source.context_cu_seqlens)
        static.decode_slots[:groups, : source.decode_slots.shape[1]].copy_(
            source.decode_slots
        )
        static.decode_lens[:groups].copy_(source.decode_lens)


class BeamDecodeCudaGraphRunner(DecodeCudaGraphRunner):
    """Decode graph runner that also holds the beam fast paths: every size gets
    an ordinary graph and, where the size holds whole beam groups, a beam graph.
    With no fast path on it captures and replays what its parent does."""

    def __init__(self, model_runner: ModelRunner, **kwargs):
        self._trie_head = model_runner.beam_trie_compact_enabled
        self._shared_context = _shared_context_capacity(model_runner)
        self._beam_inputs: Optional[BeamGraphInputs] = None
        self._capturing_beam_graph = False
        super().__init__(model_runner, **kwargs)

    def capture(self) -> None:
        if self._trie_head or self._shared_context is not None:
            self._beam_inputs = BeamGraphInputs(
                max_rows=self.max_num_token,
                device=self.device,
                trie_head=self._trie_head,
                shared_context=self._shared_context,
            )
        super().capture()
        if self._beam_inputs is None:
            logger.info(
                "No beam decode graphs captured; all-beam batches that run neither "
                "the compact trie head nor shared-context attention replay the "
                "ordinary decode graphs."
            )
        elif self._beam_inputs.captured_rows:
            logger.info(
                "Beam decode graphs captured for %s rows (compact trie head %s, "
                "shared-context attention %s).",
                sorted(self._beam_inputs.captured_rows),
                "on" if self._trie_head else "off",
                "on" if self._shared_context is not None else "off",
            )
        else:
            logger.warning(
                "SGLANG_ENABLE_BEAM_DECODE_GRAPH is set, but no decode graph size "
                "in %s is a multiple of SGLANG_BEAM_DECODE_GRAPH_WIDTH=%d; "
                "shared-context beam batches run eagerly.",
                self.capture_bs,
                self._shared_context.beam_width,
            )

    def capture_one_shape(
        self, size, forward, variant_label=None, attention_variant=None
    ):
        super().capture_one_shape(size, forward, variant_label, attention_variant)
        if self._beam_inputs is None or not self._beam_inputs.supports_rows(size):
            return
        self._capturing_beam_graph = True
        try:
            super().capture_one_shape(
                size, forward, BEAM_GRAPH_VARIANT, attention_variant
            )
        finally:
            self._capturing_beam_graph = False

    def capture_prepare(self, size, num_tokens=None):
        forward_batch, attn_backend, pp_proxy_tensors = super().capture_prepare(
            size, num_tokens=num_tokens
        )
        if self._capturing_beam_graph:
            forward_batch.beam = self._beam_inputs.for_capture(size)
        return forward_batch, attn_backend, pp_proxy_tensors

    def _resolve_lora_variant(self, forward_batch: ForwardBatch):
        if forward_batch.beam is not None:
            return BEAM_GRAPH_VARIANT
        return super()._resolve_lora_variant(forward_batch)

    def _can_replay_beam(self, forward_batch: ForwardBatch) -> bool:
        rows = forward_batch.batch_size
        if self._beam_inputs is None or rows > self.max_bs:
            return False
        captured_rows = (
            rows if self.disable_padding else self._pad_to_bucket(rows, self.capture_bs)
        )
        return self._beam_inputs.can_replay(
            forward_batch.beam, captured_rows=captured_rows
        )

    def load_batch(self, forward_batch: ForwardBatch, pp_proxy_tensors=None):
        if forward_batch.beam is not None:
            self._beam_inputs.stage(forward_batch.beam)
        super().load_batch(forward_batch, pp_proxy_tensors)

    def _init_replay_attn_metadata(self, forward_batch: ForwardBatch, **kwargs):
        # A shared-context beam graph reads the staged beam metadata and no
        # FlashInfer plan.
        if (
            forward_batch.beam is not None
            and forward_batch.beam.attention_metadata is not None
        ):
            return
        super()._init_replay_attn_metadata(forward_batch, **kwargs)

    def _process_output_after_replay(self, output, forward_batch: ForwardBatch):
        if output.beam_normalizer is None:
            return output
        return dataclasses.replace(
            output, beam_normalizer=output.beam_normalizer[: self.raw_num_token]
        )
