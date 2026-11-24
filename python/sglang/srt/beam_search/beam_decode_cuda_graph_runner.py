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
holds those paths, and replays it with the batch's beam inputs staged into the
buffers the graph reads. BeamDecodeCudaGraphRunner lists the batches a beam graph
replays.
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


class TrieLevelsInput:
    """The trie levels of the beam graphs, read by the compact trie head. Any row
    count fits, and a batch pads like any decode batch."""

    def __init__(self, *, max_rows: int, device):
        # [max_rows] int64 SID codebook depth per row.
        self._levels = torch.zeros(max_rows, dtype=torch.int64, device=device)

    def supports_rows(self, rows: int) -> bool:
        return True

    def for_capture(self, rows: int) -> torch.Tensor:
        return self._levels[:rows]

    def can_replay(
        self, trie_levels: Optional[torch.Tensor], *, graph_size: int
    ) -> bool:
        return trie_levels is not None

    def stage(self, trie_levels: torch.Tensor) -> None:
        # Padding rows keep an earlier batch's levels; a stale level is a valid depth.
        self._levels[: trie_levels.shape[0]].copy_(trie_levels)


class FixedWidthCapacity(msgspec.Struct, frozen=True):
    """What a fixed-width shared-context graph holds per beam group."""

    # The one beam width the graphs serve.
    beam_width: int
    # Prompt tokens: the model's context length, so every prompt fits.
    max_context: int
    # Generated tokens a decode step attends. The kernel unrolls its decode loop
    # this many times at compile time, so every replay pays for all of them.
    max_decode: int


class FixedWidthAttentionInput:
    """The shared-context attention metadata of the beam graphs, for one beam width.

    The kernel takes whole beam groups of that width and no padding rows. So only
    graph sizes that are a multiple of the width get a beam graph, and a batch
    replays the graph of exactly its own row count. One bucket sized for the
    largest graph serves every size; a graph of fewer groups reads its leading part.
    """

    def __init__(self, *, max_rows: int, device, capacity: FixedWidthCapacity):
        self.capacity = capacity
        width = capacity.beam_width
        groups = max_rows // width
        int32 = {"dtype": torch.int32, "device": device}
        # Captured with one prompt token and max_decode steps per group, on KV slot 0;
        # every replay stages the batch's values first.
        self._bucket = BeamAttentionBucket(
            row_indices=torch.arange(groups * width, **int32).view(groups, width),
            context_slots=torch.zeros(groups * capacity.max_context, **int32),
            context_cu_seqlens=torch.arange(groups + 1, **int32),
            decode_slots=torch.zeros((groups, width * capacity.max_decode), **int32),
            beam_width=width,
            decode_lens=torch.full((groups,), capacity.max_decode, **int32),
        )

    @classmethod
    def create(
        cls, model_runner: ModelRunner, *, max_rows: int
    ) -> Optional[FixedWidthAttentionInput]:
        """None when shared-context attention gets no graph on this server; the
        reason is logged."""
        beam_width = cls._beam_width()
        max_new_tokens = cls._max_new_tokens(model_runner.model_config)
        reason = None
        if model_runner.lora_manager is not None:
            reason = "LoRA is enabled"
        elif beam_width is None:
            reason = (
                "SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH is not set; set it to the "
                "requests' beam width"
            )
        elif max_new_tokens is None:
            reason = (
                "the model bundles no trie and "
                "SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH_MAX_NEW_TOKENS is not set"
            )
        if reason is not None:
            logger.warning(
                "SGLANG_ENABLE_BEAM_DECODE_GRAPH is set, but shared-context beam "
                "batches get no decode graph (%s); they run eagerly.",
                reason,
            )
            return None
        capacity = FixedWidthCapacity(
            beam_width=beam_width,
            max_context=model_runner.model_config.context_len,
            # The step that generates token n attends the n - 1 tokens before it.
            max_decode=max_new_tokens - 1,
        )
        return cls(max_rows=max_rows, device=model_runner.device, capacity=capacity)

    @staticmethod
    def _beam_width() -> Optional[int]:
        beam_width = envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH.get()
        if beam_width is not None and beam_width < 2:
            raise ValueError(
                "SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH must be at least 2 (a beam "
                f"request has a beam width above 1), got {beam_width}."
            )
        return beam_width

    @staticmethod
    def _max_new_tokens(model_config: ModelConfig) -> Optional[int]:
        """Tokens a beam request generates at most inside a graph; None when
        nothing bounds them."""
        trie_config = get_beam_trie_config(model_config)
        if trie_config is not None:
            # Admission pins a trie request's max_new_tokens to the codebook count.
            return trie_config.num_codebooks
        max_new_tokens = envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH_MAX_NEW_TOKENS.get()
        if max_new_tokens is not None and max_new_tokens < 2:
            raise ValueError(
                "SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH_MAX_NEW_TOKENS must be at "
                "least 2 (the first token comes from the prefill), got "
                f"{max_new_tokens}."
            )
        return max_new_tokens

    def size_requirement(self) -> str:
        """What to change when no decode graph size holds a beam graph."""
        width = self.capacity.beam_width
        return (
            f"With SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH={width} a beam graph needs "
            f"a size that is a multiple of {width} rows: add {width}, {2 * width}, "
            "... to --cuda-graph-bs-decode, one size per number of requests that "
            "decode together."
        )

    def supports_rows(self, rows: int) -> bool:
        return rows % self.capacity.beam_width == 0

    def for_capture(self, rows: int) -> BeamAttentionMetadata:
        groups = rows // self.capacity.beam_width
        bucket = BeamAttentionBucket(
            row_indices=self._bucket.row_indices[:groups],
            context_slots=self._bucket.context_slots,
            context_cu_seqlens=self._bucket.context_cu_seqlens[: groups + 1],
            decode_slots=self._bucket.decode_slots[:groups],
            beam_width=self._bucket.beam_width,
            decode_lens=self._bucket.decode_lens[:groups],
        )
        return BeamAttentionMetadata(buckets=[bucket], num_rows=rows)

    def can_replay(
        self, metadata: Optional[BeamAttentionMetadata], *, graph_size: int
    ) -> bool:
        # No padding rows, whole groups of the one width, and at most max_decode
        # generated tokens attended per beam.
        return (
            metadata is not None
            and metadata.num_rows == graph_size
            and len(metadata.buckets) == 1
            and metadata.buckets[0].beam_width == self.capacity.beam_width
            and metadata.buckets[0].decode_slots.shape[1]
            <= self.capacity.beam_width * self.capacity.max_decode
        )

    def stage(self, metadata: BeamAttentionMetadata) -> None:
        # Slots past the staged lengths keep an earlier batch's values; the kernel
        # reads no slot past a length.
        source, static = metadata.buckets[0], self._bucket
        groups = source.row_indices.shape[0]
        static.row_indices[:groups].copy_(source.row_indices)
        static.context_slots[: source.context_slots.numel()].copy_(source.context_slots)
        static.context_cu_seqlens[: groups + 1].copy_(source.context_cu_seqlens)
        static.decode_slots[:groups, : source.decode_slots.shape[1]].copy_(
            source.decode_slots
        )
        static.decode_lens[:groups].copy_(source.decode_lens)


class BeamGraphInputs:
    """What the beam graphs read: one part per BeamForwardInputs field, None where
    no graph reads the field. Every part answers supports_rows, for_capture,
    can_replay and stage for its own field, and owns buffers sized for the largest
    graph."""

    def __init__(
        self,
        *,
        trie_levels: Optional[TrieLevelsInput],
        attention: Optional[FixedWidthAttentionInput],
    ):
        self.trie_levels = trie_levels
        self.attention = attention
        # Sizes, in rows, of the beam graphs captured so far.
        self.captured_sizes: set[int] = set()

    def _parts(self, beam: BeamForwardInputs):
        """Each part with the field of the batch it reads."""
        return (
            (self.trie_levels, beam.trie_levels),
            (self.attention, beam.attention_metadata),
        )

    def supports_rows(self, rows: int) -> bool:
        parts = (self.trie_levels, self.attention)
        return all(part.supports_rows(rows) for part in parts if part is not None)

    def for_capture(self, rows: int) -> BeamForwardInputs:
        self.captured_sizes.add(rows)
        return BeamForwardInputs(
            trie_levels=(
                None if self.trie_levels is None else self.trie_levels.for_capture(rows)
            ),
            attention_metadata=(
                None if self.attention is None else self.attention.for_capture(rows)
            ),
        )

    def can_replay(self, beam: BeamForwardInputs, *, graph_size: int) -> bool:
        """Whether the beam graph of graph_size rows holds this batch; graph_size
        is the batch's row count padded up to a decode graph size."""
        if graph_size not in self.captured_sizes:
            return False
        for part, value in self._parts(beam):
            if part is None:
                # The batch runs a path the graphs do not hold.
                if value is not None:
                    return False
            elif not part.can_replay(value, graph_size=graph_size):
                return False
        return True

    def stage(self, beam: BeamForwardInputs) -> None:
        """Copy a batch that can_replay accepted into the buffers the graphs read."""
        for part, value in self._parts(beam):
            if part is not None:
                part.stage(value)


def _beam_graph_inputs(model_runner: ModelRunner) -> Optional[BeamGraphInputs]:
    """None when no all-beam batch could replay a beam graph; the reason is logged."""
    # The shared logits buffer holds the largest decode graph's rows.
    max_rows = model_runner.max_decode_logits_rows()
    attention = None
    if shared_context_attention.enabled() and (
        shared_context_attention.supports_shared_context(model_runner)
    ):
        attention = FixedWidthAttentionInput.create(model_runner, max_rows=max_rows)
        if attention is None:
            # Every all-beam batch carries shared-context metadata here, so a
            # graph without it would never replay.
            return None
    trie_levels = (
        TrieLevelsInput(max_rows=max_rows, device=model_runner.device)
        if model_runner.beam_trie_compact_enabled
        else None
    )
    if trie_levels is None and attention is None:
        logger.info(
            "No beam decode graphs captured; all-beam batches run neither the "
            "compact trie head nor shared-context attention and replay the "
            "ordinary decode graphs."
        )
        return None
    return BeamGraphInputs(trie_levels=trie_levels, attention=attention)


class BeamDecodeCudaGraphRunner(DecodeCudaGraphRunner):
    """Decode graph runner that also captures the beam fast paths.

    Besides the ordinary graph, a decode graph size gets a beam graph that holds
    the fast paths the server has on: the compact trie head, shared-context
    attention, or both. A beam graph replays only a decode batch made of beam
    requests alone:

    - With shared-context attention, only sizes that are a multiple of
      SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH get a beam graph. It replays a batch
      whose requests all have that beam width, whose rows (requests x width) are
      exactly a captured size, and whose requests are within the graph's token
      budget: the trie's codebook count, or
      SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH_MAX_NEW_TOKENS on a model without a
      trie. Any other all-beam batch runs eagerly.
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
            return
        if inputs.captured_sizes:
            logger.info(
                "Beam decode graphs captured for %s rows (compact trie head %s, "
                "shared-context attention %s).",
                sorted(inputs.captured_sizes),
                "on" if inputs.trie_levels is not None else "off",
                "on" if inputs.attention is not None else "off",
            )
        else:
            # Only the attention part restricts the sizes.
            logger.warning(
                "SGLANG_ENABLE_BEAM_DECODE_GRAPH is set, but no decode graph size "
                "in %s holds a beam graph, so all-beam batches run eagerly. %s",
                self.capture_bs,
                inputs.attention.size_requirement(),
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
