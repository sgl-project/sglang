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

Next to each ordinary decode graph, a beam graph reads the compact trie head and
shared-context attention inputs from fixed buffers; replay copies the batch's
inputs into them.
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
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner_utils.capture_mode import (
    get_capture_logits_variant,
)
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)

# Takes the LoRA label's place in the graph key; a server with LoRA captures no
# beam graph.
BEAM_GRAPH_VARIANT = "beam"


class TrieLevelsInput:
    """Graph buffer for the trie levels, which the compact trie head reads."""

    def __init__(self, *, max_rows: int, device):
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


class FixedWidthGraphConfig(msgspec.Struct, frozen=True):
    """What the fixed-width shared-context graphs are captured for."""

    beam_width: int
    max_context_len: int
    # Decode-slot capacity per beam. The kernel loops over the runtime decode
    # length, so a replay pays only for the steps its groups have taken.
    max_decode_len: int


class FixedWidthSharedContextInput:
    """Graph buffers for shared-context attention with one beam width.

    The beams of one request are a group. The kernel takes whole groups and no
    padding rows, so only sizes that are a multiple of the beam width get a graph,
    and a batch replays only the graph of exactly its own row count.

    All sizes share two buffers, sized for the largest graph. _packed_buffer holds
    four tensors back to back so that one torch.cat loads them; their offsets
    depend on the graph's number of groups.
    """

    def __init__(self, *, max_rows: int, device, config: FixedWidthGraphConfig):
        self.config = config
        max_groups = max_rows // config.beam_width
        int32 = {"dtype": torch.int32, "device": device}
        self._packed_buffer = torch.zeros(sum(self._packed_sizes(max_groups)), **int32)
        self._decode_slots_buffer = torch.zeros(
            (max_groups, config.beam_width * config.max_decode_len), **int32
        )

    @classmethod
    def create(
        cls, model_runner: ModelRunner, *, max_rows: int
    ) -> Optional[FixedWidthSharedContextInput]:
        beam_width = cls._read_beam_width()
        new_token_capacity = cls._read_new_token_capacity(model_runner.model_config)
        reason = None
        if model_runner.lora_manager is not None:
            reason = "LoRA is enabled"
        elif beam_width is None:
            reason = (
                "SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH is not set; set it to the "
                "requests' beam width"
            )
        elif new_token_capacity is None:
            reason = (
                "the model bundles no trie and "
                "SGLANG_BEAM_DECODE_GRAPH_NEW_TOKEN_CAPACITY is not set"
            )
        if reason is not None:
            logger.warning(
                "SGLANG_ENABLE_BEAM_DECODE_GRAPH is set, but shared-context beam "
                "batches get no decode graph (%s); they run eagerly.",
                reason,
            )
            return None
        config = FixedWidthGraphConfig(
            beam_width=beam_width,
            max_context_len=model_runner.model_config.context_len,
            # The step that generates token n attends the n - 1 earlier tokens.
            max_decode_len=new_token_capacity - 1,
        )
        return cls(max_rows=max_rows, device=model_runner.device, config=config)

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
    def _read_new_token_capacity(model_config: ModelConfig) -> Optional[int]:
        trie_config = get_beam_trie_config(model_config)
        if trie_config is not None:
            # Admission pins a trie request's max_new_tokens to the codebook count.
            return trie_config.num_codebooks
        new_token_capacity = envs.SGLANG_BEAM_DECODE_GRAPH_NEW_TOKEN_CAPACITY.get()
        if new_token_capacity is not None and new_token_capacity < 2:
            raise ValueError(
                "SGLANG_BEAM_DECODE_GRAPH_NEW_TOKEN_CAPACITY must be at least 2 "
                "(the first token comes from the prefill), got "
                f"{new_token_capacity}."
            )
        return new_token_capacity

    def _packed_sizes(self, num_groups: int) -> tuple[int, int, int, int]:
        """Element counts of row_indices, context_cu_seqlens, decode_lens and
        context_slots, in buffer order."""
        return (
            num_groups * self.config.beam_width,
            num_groups + 1,
            num_groups,
            num_groups * self.config.max_context_len,
        )

    def _graph_bucket(self, num_groups: int) -> BeamAttentionBucket:
        sizes = self._packed_sizes(num_groups)
        packed = self._packed_buffer[: sum(sizes)]
        row_indices, context_cu_seqlens, decode_lens, context_slots = packed.split(
            sizes
        )
        return BeamAttentionBucket(
            row_indices=row_indices.view(num_groups, self.config.beam_width),
            context_slots=context_slots,
            context_cu_seqlens=context_cu_seqlens,
            decode_slots=self._decode_slots_buffer[:num_groups],
            beam_width=self.config.beam_width,
            decode_lens=decode_lens,
        )

    def can_capture(self, graph_size: int) -> bool:
        return graph_size % self.config.beam_width == 0

    def prepare_capture_inputs(self, graph_size: int) -> BeamAttentionMetadata:
        num_groups = graph_size // self.config.beam_width
        graph_bucket = self._graph_bucket(num_groups)
        # Each size writes its own values: a smaller graph reads the packed buffer
        # at other offsets.
        rows = torch.arange(graph_size, device=self._packed_buffer.device)
        graph_bucket.row_indices.copy_(rows.view_as(graph_bucket.row_indices))
        graph_bucket.context_cu_seqlens.copy_(rows[: num_groups + 1])
        graph_bucket.decode_lens.fill_(self.config.max_decode_len)
        graph_bucket.context_slots.zero_()
        graph_bucket.decode_slots.zero_()
        return BeamAttentionMetadata(buckets=[graph_bucket], num_rows=graph_size)

    def can_replay(
        self, metadata: Optional[BeamAttentionMetadata], *, graph_size: int
    ) -> bool:
        # One bucket per beam width among the requests.
        if metadata is None or len(metadata.buckets) != 1:
            return False
        batch_bucket = metadata.buckets[0]
        decode_width = batch_bucket.decode_slots.shape[1]
        return (
            batch_bucket.beam_width == self.config.beam_width
            and metadata.num_rows == graph_size
            and decode_width <= self.config.beam_width * self.config.max_decode_len
        )

    def load_replay_inputs(self, metadata: BeamAttentionMetadata) -> None:
        # Slots past the batch's lengths keep an earlier batch's values; the kernel
        # reads no slot past a length.
        batch_bucket = metadata.buckets[0]
        # The order of _packed_sizes.
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
    """The buffers the beam graphs read: one input per BeamForwardInputs field,
    None when no graph reads its field."""

    def __init__(
        self,
        *,
        trie_levels_input: Optional[TrieLevelsInput],
        shared_context_input: Optional[FixedWidthSharedContextInput],
    ):
        self.trie_levels_input = trie_levels_input
        self.shared_context_input = shared_context_input

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
        for graph_input, batch_value in self._inputs_with_batch_values(beam):
            if graph_input is None:
                # The batch runs a fast path the graphs do not hold.
                if batch_value is not None:
                    return False
            elif not graph_input.can_replay(batch_value, graph_size=graph_size):
                return False
        return True

    def load_replay_inputs(self, beam: BeamForwardInputs) -> None:
        for graph_input, batch_value in self._inputs_with_batch_values(beam):
            if graph_input is not None:
                graph_input.load_replay_inputs(batch_value)

    @classmethod
    def create(cls, model_runner: ModelRunner) -> Optional[BeamGraphInputs]:
        max_rows = model_runner.max_decode_logits_rows()
        shared_context_input = None
        if shared_context_attention.enabled() and (
            shared_context_attention.supports_shared_context(model_runner)
        ):
            shared_context_input = FixedWidthSharedContextInput.create(
                model_runner, max_rows=max_rows
            )
            if shared_context_input is None:
                # Every all-beam batch carries shared-context metadata here, so a
                # graph without that input would never replay.
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
        return cls(
            trie_levels_input=trie_levels_input,
            shared_context_input=shared_context_input,
        )


class BeamDecodeCudaGraphRunner(DecodeCudaGraphRunner):
    """Decode graph runner that also replays decode batches of beam requests.

    With shared-context attention, a beam graph replays only a batch whose
    requests all have SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH and whose rows equal a
    captured size. With the compact trie head alone, requests may differ in width
    and the batch pads like any decode batch. Other batches run as on the parent.
    """

    @staticmethod
    def enabled(spec_algorithm: SpeculativeAlgorithm) -> bool:
        # BeamCoordinator.validate_and_init refuses beam requests under
        # speculative decoding, PP and DP attention. PP also breaks replay:
        # non-last ranks return no beam_normalizer.
        return (
            envs.SGLANG_ENABLE_BEAM_DECODE_GRAPH.get()
            and spec_algorithm.is_none()
            and get_parallel().pp_size == 1
            and not is_dp_attention_enabled()
        )

    def __init__(self, model_runner: ModelRunner, **kwargs):
        # Set before the parent's __init__, which captures the graphs.
        self._beam_graph_inputs = BeamGraphInputs.create(model_runner)
        self._capturing_beam_graph = False
        super().__init__(model_runner, **kwargs)
        self._log_captured_beam_graphs()

    def _log_captured_beam_graphs(self) -> None:
        graph_inputs = self._beam_graph_inputs
        if graph_inputs is None:
            return
        captured_sizes = [
            size for size in self.capture_bs if graph_inputs.can_capture(size)
        ]
        if captured_sizes:
            logger.info(
                "Beam decode graphs captured for %s rows (compact trie head %s, "
                "shared-context attention %s).",
                captured_sizes,
                "on" if graph_inputs.trie_levels_input is not None else "off",
                "on" if graph_inputs.shared_context_input is not None else "off",
            )
        else:
            width = graph_inputs.shared_context_input.config.beam_width
            logger.warning(
                "SGLANG_ENABLE_BEAM_DECODE_GRAPH is set, but no decode graph size "
                "in %s holds a beam graph, so all-beam batches run eagerly. With "
                "SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH=%d a beam graph needs a size "
                "that is a multiple of %d rows: add %d, %d, ... to "
                "--cuda-graph-bs-decode, one size per number of requests that "
                "decode together.",
                self.capture_bs,
                width,
                width,
                width,
                2 * width,
            )

    def capture_one_shape(
        self, size, forward, variant_label=None, attention_variant=None
    ):
        super().capture_one_shape(size, forward, variant_label, attention_variant)
        graph_inputs = self._beam_graph_inputs
        if graph_inputs is None or not graph_inputs.can_capture(size):
            return
        self._set_profile_trace_tag(
            size, BEAM_GRAPH_VARIANT, attention_variant, get_capture_logits_variant()
        )
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

    def _can_run_beam_graph(self, forward_batch: ForwardBatch) -> bool:
        graph_inputs = self._beam_graph_inputs
        rows = forward_batch.batch_size
        if graph_inputs is None or rows > self.max_bs:
            return False
        graph_size = (
            rows if self.disable_padding else self._pad_to_bucket(rows, self.capture_bs)
        )
        return graph_inputs.can_replay(forward_batch.beam, graph_size=graph_size)

    def load_batch(self, forward_batch: ForwardBatch, pp_proxy_tensors=None):
        if forward_batch.beam is not None:
            self._beam_graph_inputs.load_replay_inputs(forward_batch.beam)
        super().load_batch(forward_batch, pp_proxy_tensors)

    def _init_replay_attn_metadata(self, forward_batch: ForwardBatch, **kwargs):
        # A shared-context beam graph reads the loaded beam metadata and needs no
        # FlashInfer plan.
        if shared_context_attention.has_shared_context(forward_batch):
            return
        super()._init_replay_attn_metadata(forward_batch, **kwargs)

    def _process_output_after_replay(self, output, forward_batch: ForwardBatch):
        if output.beam_normalizer is None:
            return output
        return dataclasses.replace(
            output, beam_normalizer=output.beam_normalizer[: self.raw_num_token]
        )
