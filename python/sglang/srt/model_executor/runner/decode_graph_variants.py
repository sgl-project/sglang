# Copyright 2023-2026 SGLang Team
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
"""Speculative graph variants of the decode / target-verify CUDA-graph runner.

A speculative algorithm whose verify width changes from step to step (for
example an adaptive draft length) cannot replay one graph per batch size at
the static width. It returns a ``DecodeGraphVariants`` from
``create_decode_graph_variants(info)`` (``info`` is the runner's
``DecodeGraphRunnerInfo``) and the runner then:

* captures, per batch size, one graph per ``capture_variants(bs)`` entry, each
  at ``bs * variant.num_tokens_per_req`` tokens, keyed by
  ``ShapeKey.spec_variant = variant.label``;
* at replay asks ``replay_variant(forward_batch)`` which graph the step runs
  (``None``, or a variant not captured for the padded batch size, falls back to
  eager);
* lets the provider pick the attention backend of each variant
  (``begin_capture`` / ``begin_replay``) and clean up after the backend built
  its out-of-graph metadata (``end_metadata``).

Every width must fit the runner's static buffers
(``num_tokens_per_req <= info.max_tokens_per_req``). The provider owns the
graph axis: when one is set, the runner captures no default-width graph.
Without a provider (the default for every builtin algorithm) the runner is
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence

if TYPE_CHECKING:
    import torch

    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


class DecodeGraphRunnerInfo(NamedTuple):
    """What a variants provider may know about the runner that asks for it."""

    is_draft_worker: bool
    # DECODE, TARGET_VERIFY or DLLM_EXTEND: the mode the runner captures.
    capture_forward_mode: ForwardMode
    # The widest verify a variant may use (the runner's static buffers).
    max_tokens_per_req: int
    device: torch.device


class DecodeGraphVariant(NamedTuple):
    """One captured graph per batch size: ``label`` keys it, and every request
    of the forward carries ``num_tokens_per_req`` tokens."""

    label: str
    num_tokens_per_req: int


class DecodeGraphVariants:
    """Base provider: no variants. Subclass and override what you need."""

    def capture_variants(self, bs: int) -> Sequence[DecodeGraphVariant]:
        """Variants to capture for batch size ``bs``, in capture order (put
        the widest first so narrower graphs reuse its memory pool)."""
        return ()

    def replay_variant(
        self, forward_batch: ForwardBatch
    ) -> Optional[DecodeGraphVariant]:
        """The variant this batch replays, from host-side state only (called
        by ``can_run_graph`` and ``load_batch``). ``None`` runs the batch
        eagerly."""
        return None

    def begin_capture(
        self,
        variant: DecodeGraphVariant,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
        stream_idx: Optional[int],
    ) -> AttentionBackend:
        """Called after the capture batch is built and before the backend
        plans its capture metadata. May edit ``forward_batch.spec_info`` (for
        example pin its width to the variant). Returns the attention backend
        to capture this variant with (default: the runner's)."""
        return attn_backend

    def begin_replay(
        self,
        variant: DecodeGraphVariant,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
        stream_idx: Optional[int],
    ) -> AttentionBackend:
        """Called before the backend builds the replay metadata. Must return
        the backend this variant was captured with."""
        return attn_backend

    def end_metadata(
        self, variant: DecodeGraphVariant, forward_batch: ForwardBatch
    ) -> None:
        """Called once the backend built its out-of-graph metadata, at capture
        and at replay."""
