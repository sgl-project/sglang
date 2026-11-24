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
metadata staged into the buffers the graph reads. BeamDecodeCudaGraphRunner lists
the batches a beam graph replays.
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


def _static_bucket(
    *, max_rows: int, device, capacity: SharedContextCapacity
) -> BeamAttentionBucket:
    width = capacity.beam_width
    groups = max_rows // width
    int32 = {"dtype": torch.int32, "device": device}
    # Captured with one prompt token and max_decode steps per group, on KV slot 0;
    # every replay stages the batch's values first.
    return BeamAttentionBucket(
        row_indices=torch.arange(groups * width, **int32).view(groups, width),
        context_slots=torch.zeros(groups * capacity.max_context, **int32),
        context_cu_seqlens=torch.arange(groups + 1, **int32),
        decode_slots=torch.zeros((groups, width * capacity.max_decode), **int32),
        beam_width=width,
        decode_lens=torch.full((groups,), capacity.max_decode, **int32),
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
        self.shared_context = shared_context
        # Sizes, in rows, of the beam graphs captured so far.
        self.captured_sizes: set[int] = set()
        # [max_rows] int64 SID codebook depth per row; None under the dense LM head.
        self.levels: Optional[torch.Tensor] = (
            torch.zeros(max_rows, dtype=torch.int64, device=device)
            if trie_head
            else None
        )
        # The one bucket every shared-context graph reads; None under ordinary
        # attention.
        self._bucket: Optional[BeamAttentionBucket] = (
            _static_bucket(max_rows=max_rows, device=device, capacity=shared_context)
            if shared_context is not None
            else None
        )

    def supports_rows(self, rows: int) -> bool:
        """With shared-context attention a graph holds whole groups of one width."""
        return self.shared_context is None or rows % self.shared_context.beam_width == 0

    def for_capture(self, rows: int) -> BeamForwardInputs:
        self.captured_sizes.add(rows)
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

    def can_replay(self, beam: BeamForwardInputs, *, graph_size: int) -> bool:
        """Whether the beam graph of graph_size rows holds this batch; graph_size
        is the batch's row count padded up to a decode graph size."""
        if graph_size not in self.captured_sizes:
            return False
        metadata = beam.attention_metadata
        capacity = self.shared_context
        if metadata is None or capacity is None:
            # Ordinary attention pads like any decode batch.
            return metadata is None and capacity is None
        # Shared-context attention takes whole groups of the one width and no
        # padding rows, and attends at most max_decode generated tokens per beam.
        return (
            metadata.num_rows == graph_size
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


def _beam_graph_inputs(model_runner: ModelRunner) -> Optional[BeamGraphInputs]:
    """None when the server runs neither beam fast path; its all-beam batches
    carry no beam inputs and replay the ordinary decode graphs."""
    trie_head = model_runner.beam_trie_compact_enabled
    shared_context = _shared_context_capacity(model_runner)
    if not trie_head and shared_context is None:
        return None
    return BeamGraphInputs(
        # The shared logits buffer holds the largest decode graph's rows.
        max_rows=model_runner.max_decode_logits_rows(),
        device=model_runner.device,
        trie_head=trie_head,
        shared_context=shared_context,
    )


class BeamDecodeCudaGraphRunner(DecodeCudaGraphRunner):
    """Decode graph runner that also captures the beam fast paths.

    Besides the ordinary graph, a decode graph size gets a beam graph that holds
    the fast paths the server has on: the compact trie head, shared-context
    attention, or both. A beam graph replays only a decode batch made of beam
    requests alone:

    - With shared-context attention, only sizes that are a multiple of
      SGLANG_BEAM_DECODE_GRAPH_WIDTH get a beam graph. It replays a batch whose
      requests all have that beam width, whose rows (requests x width) are
      exactly a captured size, and whose requests are within the graph's token
      budget: the trie's codebook count, or
      SGLANG_BEAM_DECODE_GRAPH_MAX_NEW_TOKENS on a model without a trie. Any
      other all-beam batch runs eagerly.
    - With the compact trie head on ordinary attention, every size gets a beam
      graph. Requests may differ in beam width, and the batch pads to the next
      captured size like any decode batch.

    A batch that mixes beam and regular requests carries no beam inputs and
    decodes on the ordinary path. So does every batch when neither fast path is
    on; this runner then captures and replays what its parent does.
    """

    def __init__(self, model_runner: ModelRunner, **kwargs):
        # Both are set before the parent's __init__, which captures the graphs.
        self._beam_inputs = _beam_graph_inputs(model_runner)
        self._capturing_beam_graph = False
        super().__init__(model_runner, **kwargs)
        self._log_captured_beam_graphs()

    def _log_captured_beam_graphs(self) -> None:
        inputs = self._beam_inputs
        if inputs is None:
            logger.info(
                "No beam decode graphs captured; all-beam batches that run neither "
                "the compact trie head nor shared-context attention replay the "
                "ordinary decode graphs."
            )
        elif inputs.captured_sizes:
            logger.info(
                "Beam decode graphs captured for %s rows (compact trie head %s, "
                "shared-context attention %s).",
                sorted(inputs.captured_sizes),
                "on" if inputs.levels is not None else "off",
                "on" if inputs.shared_context is not None else "off",
            )
        else:
            logger.warning(
                "SGLANG_ENABLE_BEAM_DECODE_GRAPH is set, but no decode graph size "
                "in %s is a multiple of SGLANG_BEAM_DECODE_GRAPH_WIDTH=%d; "
                "shared-context beam batches run eagerly.",
                self.capture_bs,
                inputs.shared_context.beam_width,
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
        graph_size = (
            rows if self.disable_padding else self._pad_to_bucket(rows, self.capture_bs)
        )
        return self._beam_inputs.can_replay(forward_batch.beam, graph_size=graph_size)

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
