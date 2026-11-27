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
"""CUDA graphs for decode batches made only of beam requests.

Such a batch runs the compact trie head, shared-context attention, or both. The
ordinary decode graphs hold neither, so the parent runner runs the batch eagerly.
BeamDecodeCudaGraphRunner adds beam graphs:

1. Capture: next to the ordinary graph of a size, it captures a beam graph that
   reads its beam inputs from fixed buffers.
2. Replay: it copies the batch's beam inputs into those buffers, then replays
   the beam graph.
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

# Graph key label of the beam graphs. It takes the LoRA label's place in the key;
# a server with LoRA captures no beam graph.
BEAM_GRAPH_VARIANT = "beam"


class TrieLevelsInput:
    """Graph buffer for the trie levels, which the compact trie head reads.

    Every graph size works; a batch pads like any decode batch.
    """

    def __init__(self, *, max_rows: int, device):
        # [max_rows] int64 SID codebook depth per row.
        self._graph_levels = torch.zeros(max_rows, dtype=torch.int64, device=device)

    def can_capture(self, graph_size: int) -> bool:
        return True

    def prepare_capture_inputs(self, graph_size: int) -> torch.Tensor:
        return self._graph_levels[:graph_size]

    def can_replay(
        self, trie_levels: Optional[torch.Tensor], *, graph_size: int
    ) -> bool:
        return trie_levels is not None

    def load_replay_inputs(self, trie_levels: torch.Tensor) -> None:
        # Padding rows keep an earlier batch's levels; a stale level is a valid depth.
        self._graph_levels[: trie_levels.shape[0]].copy_(trie_levels)


class FixedWidthCapacity(msgspec.Struct, frozen=True):
    """Limits of the fixed-width shared-context graphs."""

    # The one beam width the graphs serve.
    beam_width: int
    # Prompt tokens per request: the model's context length, so every prompt fits.
    max_context_len: int
    # Generated tokens per beam that a decode step attends. The kernel unrolls its
    # decode loop this many times at compile time, so every replay pays for all.
    max_decode_len: int


class FixedWidthSharedContextInput:
    """Graph buffers for shared-context attention with one beam width.

    A graph serves several requests at once when each has that beam width: a
    graph of N rows holds N / beam_width requests. The beams of one request are
    a group. The kernel takes whole groups and no padding rows:

    - Only a graph size that is a multiple of the beam width gets a beam graph.
    - A batch replays only the graph of exactly its own row count.

    All graph sizes share two buffers, sized for the largest graph:

    - _packed_buffer holds row_indices, context_cu_seqlens, decode_lens and
      context_slots back to back, so that one torch.cat loads a batch's four
      tensors. Where each starts depends on the graph's number of groups.
    - _decode_slots_buffer holds one row per group; a smaller graph reads its
      leading rows.
    """

    def __init__(self, *, max_rows: int, device, capacity: FixedWidthCapacity):
        self.capacity = capacity
        max_groups = max_rows // capacity.beam_width
        int32 = {"dtype": torch.int32, "device": device}
        self._packed_buffer = torch.zeros(sum(self._packed_sizes(max_groups)), **int32)
        self._decode_slots_buffer = torch.zeros(
            (max_groups, capacity.beam_width * capacity.max_decode_len), **int32
        )

    @classmethod
    def create(
        cls, model_runner: ModelRunner, *, max_rows: int
    ) -> Optional[FixedWidthSharedContextInput]:
        """Returns None, and logs why, when shared-context batches get no graph."""
        beam_width = cls._read_beam_width()
        max_new_tokens = cls._read_max_new_tokens(model_runner.model_config)
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
            max_context_len=model_runner.model_config.context_len,
            # The step that generates token n attends the n - 1 tokens before it.
            max_decode_len=max_new_tokens - 1,
        )
        return cls(max_rows=max_rows, device=model_runner.device, capacity=capacity)

    @staticmethod
    def _read_beam_width() -> Optional[int]:
        beam_width = envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH.get()
        if beam_width is not None and beam_width < 2:
            raise ValueError(
                "SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH must be at least 2 (a beam "
                f"request has a beam width above 1), got {beam_width}."
            )
        return beam_width

    @staticmethod
    def _read_max_new_tokens(model_config: ModelConfig) -> Optional[int]:
        """Tokens a request may generate on the graphs; None when nothing sets it."""
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

    def describe_size_requirement(self) -> str:
        """Which decode graph sizes to add; logged when no size got a beam graph."""
        width = self.capacity.beam_width
        return (
            f"With SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH={width} a beam graph needs "
            f"a size that is a multiple of {width} rows: add {width}, {2 * width}, "
            "... to --cuda-graph-bs-decode, one size per number of requests that "
            "decode together."
        )

    def _packed_sizes(self, num_groups: int) -> tuple[int, int, int, int]:
        """Element counts of row_indices, context_cu_seqlens, decode_lens and
        context_slots, in buffer order, for a graph of num_groups groups."""
        return (
            num_groups * self.capacity.beam_width,
            num_groups + 1,
            num_groups,
            num_groups * self.capacity.max_context_len,
        )

    def _graph_bucket(self, num_groups: int) -> BeamAttentionBucket:
        """Views of the buffers that a graph of num_groups groups reads."""
        sizes = self._packed_sizes(num_groups)
        packed = self._packed_buffer[: sum(sizes)]
        row_indices, context_cu_seqlens, decode_lens, context_slots = packed.split(
            sizes
        )
        return BeamAttentionBucket(
            row_indices=row_indices.view(num_groups, self.capacity.beam_width),
            context_slots=context_slots,
            context_cu_seqlens=context_cu_seqlens,
            decode_slots=self._decode_slots_buffer[:num_groups],
            beam_width=self.capacity.beam_width,
            decode_lens=decode_lens,
        )

    def can_capture(self, graph_size: int) -> bool:
        return graph_size % self.capacity.beam_width == 0

    def prepare_capture_inputs(self, graph_size: int) -> BeamAttentionMetadata:
        num_groups = graph_size // self.capacity.beam_width
        graph_bucket = self._graph_bucket(num_groups)
        # Capture-time values: per group, one prompt token and max_decode_len
        # generated tokens on KV slot 0. Each size writes its own, because a
        # smaller graph reads the packed buffer at other offsets.
        rows = torch.arange(graph_size, device=self._packed_buffer.device)
        graph_bucket.row_indices.copy_(rows.view_as(graph_bucket.row_indices))
        graph_bucket.context_cu_seqlens.copy_(rows[: num_groups + 1])
        graph_bucket.decode_lens.fill_(self.capacity.max_decode_len)
        graph_bucket.context_slots.zero_()
        graph_bucket.decode_slots.zero_()
        return BeamAttentionMetadata(buckets=[graph_bucket], num_rows=graph_size)

    def can_replay(
        self, metadata: Optional[BeamAttentionMetadata], *, graph_size: int
    ) -> bool:
        # A batch has one bucket per beam width among its requests, so more than
        # one bucket means requests of different widths.
        if metadata is None or len(metadata.buckets) != 1:
            return False
        batch_bucket = metadata.buckets[0]
        # [num_groups, beam_width * generated tokens per beam]
        decode_width = batch_bucket.decode_slots.shape[1]
        return (
            # Every request has the graphs' beam width.
            batch_bucket.beam_width == self.capacity.beam_width
            # The requests fill the graph: the kernel takes no padding rows.
            and metadata.num_rows == graph_size
            and decode_width <= self.capacity.beam_width * self.capacity.max_decode_len
        )

    def load_replay_inputs(self, metadata: BeamAttentionMetadata) -> None:
        # Slots past the batch's lengths keep an earlier batch's values; the kernel
        # reads no slot past a length.
        batch_bucket = metadata.buckets[0]
        # The order of _packed_sizes; context_slots is last because its length varies.
        packed = (
            batch_bucket.row_indices.reshape(-1),
            batch_bucket.context_cu_seqlens,
            batch_bucket.decode_lens,
            batch_bucket.context_slots,
        )
        packed_len = sum(tensor.numel() for tensor in packed)
        torch.cat(packed, out=self._packed_buffer[:packed_len])
        num_groups, decode_width = batch_bucket.decode_slots.shape
        self._decode_slots_buffer[:num_groups, :decode_width].copy_(
            batch_bucket.decode_slots
        )


class BeamGraphInputs:
    """The buffers the beam graphs read: one input per BeamForwardInputs field.

    An input is None when no graph reads its field. Every input has four methods:

    - can_capture(graph_size): whether a graph of this size can read the input.
    - prepare_capture_inputs(graph_size): the buffer views that graph captures.
    - can_replay(batch_value, graph_size=...): whether that graph fits the batch.
    - load_replay_inputs(batch_value): copy the batch's value into the buffers.
    """

    def __init__(
        self,
        *,
        trie_levels_input: Optional[TrieLevelsInput],
        shared_context_input: Optional[FixedWidthSharedContextInput],
    ):
        self.trie_levels_input = trie_levels_input
        self.shared_context_input = shared_context_input
        # Sizes, in rows, of the beam graphs captured so far.
        self.captured_sizes: set[int] = set()

    def _inputs_with_batch_values(self, beam: BeamForwardInputs):
        return (
            (self.trie_levels_input, beam.trie_levels),
            (self.shared_context_input, beam.attention_metadata),
        )

    def can_capture(self, graph_size: int) -> bool:
        graph_inputs = (self.trie_levels_input, self.shared_context_input)
        return all(
            graph_input.can_capture(graph_size)
            for graph_input in graph_inputs
            if graph_input is not None
        )

    def prepare_capture_inputs(self, graph_size: int) -> BeamForwardInputs:
        self.captured_sizes.add(graph_size)
        return BeamForwardInputs(
            trie_levels=(
                None
                if self.trie_levels_input is None
                else self.trie_levels_input.prepare_capture_inputs(graph_size)
            ),
            attention_metadata=(
                None
                if self.shared_context_input is None
                else self.shared_context_input.prepare_capture_inputs(graph_size)
            ),
        )

    def can_replay(self, beam: BeamForwardInputs, *, graph_size: int) -> bool:
        """graph_size is the batch's row count padded up to a decode graph size."""
        if graph_size not in self.captured_sizes:
            return False
        for graph_input, batch_value in self._inputs_with_batch_values(beam):
            if graph_input is None:
                # The batch runs a fast path the graphs do not hold.
                if batch_value is not None:
                    return False
            elif not graph_input.can_replay(batch_value, graph_size=graph_size):
                return False
        return True

    def load_replay_inputs(self, beam: BeamForwardInputs) -> None:
        """Only for a batch that can_replay accepted."""
        for graph_input, batch_value in self._inputs_with_batch_values(beam):
            if graph_input is not None:
                graph_input.load_replay_inputs(batch_value)


def _create_beam_graph_inputs(model_runner: ModelRunner) -> Optional[BeamGraphInputs]:
    """Returns None, and logs why, when no all-beam batch could replay a beam graph."""
    # Rows of the largest decode graph; the shared logits buffer holds as many.
    max_rows = model_runner.max_decode_logits_rows()
    shared_context_input = None
    if shared_context_attention.enabled() and (
        shared_context_attention.supports_shared_context(model_runner)
    ):
        shared_context_input = FixedWidthSharedContextInput.create(
            model_runner, max_rows=max_rows
        )
        if shared_context_input is None:
            # Every all-beam batch carries shared-context metadata on this server,
            # so a beam graph without that input would never replay.
            return None
    trie_levels_input = (
        TrieLevelsInput(max_rows=max_rows, device=model_runner.device)
        if model_runner.beam_trie_compact_enabled
        else None
    )
    if trie_levels_input is None and shared_context_input is None:
        logger.info(
            "No beam decode graphs captured; all-beam batches run neither the "
            "compact trie head nor shared-context attention and replay the "
            "ordinary decode graphs."
        )
        return None
    return BeamGraphInputs(
        trie_levels_input=trie_levels_input, shared_context_input=shared_context_input
    )


class BeamDecodeCudaGraphRunner(DecodeCudaGraphRunner):
    """Decode graph runner that also replays decode batches of beam requests.

    Next to the ordinary graph, a decode graph size gets a beam graph that holds
    the beam fast paths the server has on: the compact trie head, shared-context
    attention, or both. A beam graph replays a decode batch made only of beam
    requests:

    - With shared-context attention:
        - Only sizes that are a multiple of SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH
          get a beam graph.
        - Every request has that beam width.
        - The batch's rows (requests x width) equal a captured size.
        - Every request is within the graphs' token budget: the trie's codebook
          count, or SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH_MAX_NEW_TOKENS on a
          model without a trie.
    - With the compact trie head on ordinary attention:
        - Every size gets a beam graph.
        - Requests may differ in beam width.
        - The batch pads to the next captured size like any decode batch.

    Every other batch runs as it does on the parent:

    - An all-beam batch that fits no beam graph runs eagerly.
    - A batch that mixes beam and regular requests carries no beam inputs and
      decodes on the ordinary path.
    - With neither fast path on, no beam graph is captured.
    """

    def __init__(self, model_runner: ModelRunner, **kwargs):
        # Set both before the parent's __init__, which captures the graphs.
        self._beam_graph_inputs = _create_beam_graph_inputs(model_runner)
        self._capturing_beam_graph = False
        super().__init__(model_runner, **kwargs)
        self._log_captured_beam_graphs()

    def _log_captured_beam_graphs(self) -> None:
        graph_inputs = self._beam_graph_inputs
        if graph_inputs is None:
            return
        if graph_inputs.captured_sizes:
            logger.info(
                "Beam decode graphs captured for %s rows (compact trie head %s, "
                "shared-context attention %s).",
                sorted(graph_inputs.captured_sizes),
                "on" if graph_inputs.trie_levels_input is not None else "off",
                "on" if graph_inputs.shared_context_input is not None else "off",
            )
        else:
            # Only the shared-context input refuses sizes, so it is set here.
            logger.warning(
                "SGLANG_ENABLE_BEAM_DECODE_GRAPH is set, but no decode graph size "
                "in %s holds a beam graph, so all-beam batches run eagerly. %s",
                self.capture_bs,
                graph_inputs.shared_context_input.describe_size_requirement(),
            )

    def capture_one_shape(
        self, size, forward, variant_label=None, attention_variant=None
    ):
        super().capture_one_shape(size, forward, variant_label, attention_variant)
        graph_inputs = self._beam_graph_inputs
        if graph_inputs is None or not graph_inputs.can_capture(size):
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
            forward_batch.beam = self._beam_graph_inputs.prepare_capture_inputs(size)
        return forward_batch, attn_backend, pp_proxy_tensors

    def _resolve_lora_variant(self, forward_batch: ForwardBatch):
        if forward_batch.beam is not None:
            return BEAM_GRAPH_VARIANT
        return super()._resolve_lora_variant(forward_batch)

    def _can_replay_beam(self, forward_batch: ForwardBatch) -> bool:
        rows = forward_batch.batch_size
        if self._beam_graph_inputs is None or rows > self.max_bs:
            return False
        graph_size = (
            rows if self.disable_padding else self._pad_to_bucket(rows, self.capture_bs)
        )
        return self._beam_graph_inputs.can_replay(
            forward_batch.beam, graph_size=graph_size
        )

    def load_batch(self, forward_batch: ForwardBatch, pp_proxy_tensors=None):
        if forward_batch.beam is not None:
            self._beam_graph_inputs.load_replay_inputs(forward_batch.beam)
        super().load_batch(forward_batch, pp_proxy_tensors)

    def _init_replay_attn_metadata(self, forward_batch: ForwardBatch, **kwargs):
        # A shared-context beam graph reads the loaded beam metadata and needs no
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
