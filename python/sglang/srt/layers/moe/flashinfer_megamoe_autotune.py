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
"""Startup tuning and immutable capacity profiles for FlashInfer MegaMoE."""

from __future__ import annotations

import json
import logging
import os
from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from functools import cached_property
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class _AutotuneContext:
    decode_num_tokens: int
    prefill_num_tokens: int
    records: dict
    dirty: bool = False


_active_context: _AutotuneContext | None = None
# Keep direct knob records separate from the legacy native cache-file payloads.
_CACHE_NAMESPACE = "sglang_flashinfer_megamoe_knobs_v1"


@contextmanager
def megamoe_autotune_context(
    decode_num_tokens: int,
    prefill_num_tokens: int,
):
    """Prepare profiles inside the engine's pre-capture FlashInfer autotune pass."""
    from flashinfer.autotuner import AutoTuner

    global _active_context
    tuner = AutoTuner.get()
    previous = _active_context
    context = _AutotuneContext(
        decode_num_tokens,
        prefill_num_tokens,
        tuner.get_namespaced_records(_CACHE_NAMESPACE),
    )
    _active_context = context
    try:
        yield
    finally:
        _active_context = previous
        if context.dirty:
            tuner.publish_namespaced_records(_CACHE_NAMESPACE, context.records)


def _capacities(limit: int) -> list[int]:
    limit = max(1, limit)
    return sorted({1 << i for i in range(limit.bit_length())} | {limit})


def _profile_inputs(tensors, capacity: int, num_experts: int):
    count = tensors.num_tokens
    if count:

        def resize(tensor):
            if capacity <= count:
                return tensor[:capacity]
            return tensor.repeat(
                ((capacity + count - 1) // count,) + (1,) * (tensor.ndim - 1)
            )[:capacity]

        hidden = resize(tensors.hidden_states)
        ids = resize(tensors.topk_ids)
        scores = resize(tensors.topk_weights)
        scales = resize(tensors.scales) if tensors.scales is not None else None
    else:
        # Idle ranks still enter every collective tuning iteration.
        hidden = tensors.hidden_states.new_zeros(
            (capacity, tensors.hidden_states.shape[1])
        )
        top_k = tensors.topk_ids.shape[1]
        ids = (
            torch.arange(
                capacity * top_k, dtype=tensors.topk_ids.dtype, device=hidden.device
            ).reshape(capacity, top_k)
            % num_experts
        )
        scores = tensors.topk_weights.new_full((capacity, top_k), 1.0 / top_k)
        scales = (
            tensors.scales.new_ones((capacity, *tensors.scales.shape[1:]))
            if tensors.scales is not None
            else None
        )
    return replace(
        tensors, hidden_states=hidden, topk_ids=ids, topk_weights=scores, scales=scales
    )


def _valid_knobs(knobs) -> bool:
    return isinstance(knobs, dict) and bool(knobs)


class MegaMoeTunedForward:
    """Own public workspace handles; FlashInfer pools their storage across layers.

    The caller installs this for backends with prepared weights and default
    ``knobs=None``. Preparation must precede ordinary forwarding and capture.
    """

    def __init__(self, bootstrap, fleet_params, backend):
        if backend.megakernel.knobs is not None:
            raise ValueError("MegaMoE startup tuning requires knobs=None")
        if backend.preprocess_weights or backend.transformed_weights is None:
            raise ValueError("MegaMoE startup tuning requires prepared weights")
        self.bootstrap = bootstrap
        self.fleet_params = fleet_params
        self.backend = backend
        self.workspaces: dict[int, Any] = {}
        self._fallback_used = False
        self._profile_capacities: tuple[int, ...] = ()

    def _common_tokens(self, tensors) -> int:
        if self.bootstrap.world_size == 1:
            return tensors.num_tokens
        from sglang.srt.layers.dp_attention import get_dp_global_num_tokens

        counts = get_dp_global_num_tokens()
        if counts is None:
            raise RuntimeError(
                "MegaMoE workspace selection requires synchronized DP token counts"
            )
        return max(counts, default=0)

    @cached_property
    def _geometry_key(self) -> str:
        config = {}
        for item in fields(self.backend.megakernel):
            # Per-layer scale values do not change the pooled kernel geometry.
            if item.name in (
                "knobs",
                "input_norm_const",
                "fc1_norm_const",
                "fc1_alpha",
                "fc2_alpha",
            ):
                continue
            value = getattr(self.backend.megakernel, item.name)
            if not isinstance(value, torch.Tensor):
                config[item.name] = value
        # The enclosing flashinfer.autotune context validates runtime/compiler
        # metadata. This key only adds the MegaMoE configuration and geometry.
        return json.dumps(
            {
                "torch": torch.__version__,
                "mega_use_ncu": os.environ.get("MEGA_USE_NCU", "0"),
                "config": config,
                "world_size": self.bootstrap.world_size,
                "hidden": self.fleet_params.token_hidden_size,
                "num_experts": self.fleet_params.num_experts,
            },
            sort_keys=True,
        )

    def _broadcast(self, value):
        if self.bootstrap.world_size == 1:
            return value
        import torch.distributed as dist

        group = self.bootstrap.process_group
        values = [value]
        dist.broadcast_object_list(values, group=group, group_src=0)
        return values[0]

    def _tune(self, tensors, capacity: int):
        from flashinfer.moe_ep import MoEEpMegaLayer

        backend = replace(
            self.backend, megakernel=replace(self.backend.megakernel, knobs="auto")
        )
        temporary = MoEEpMegaLayer(
            bootstrap=self.bootstrap,
            fleet_params=replace(self.fleet_params, max_tokens_per_rank=capacity),
            weights=None,
            backend=backend,
        )
        try:
            temporary.warmup(tensors)
            knobs = temporary.get_tuned_knobs()
        finally:
            temporary.destroy()
        knobs = self._broadcast(knobs)
        if not _valid_knobs(knobs):
            raise RuntimeError("MegaMoE autotune did not return its selected knobs")
        return knobs

    def _prepare_profile(self, mega, tensors, capacity: int) -> None:
        context = _active_context
        assert context is not None
        profile_key = f"{self._geometry_key}:{capacity}"
        profile_inputs = _profile_inputs(
            tensors, capacity, self.fleet_params.num_experts
        )
        # Rank zero decides whether every EP rank reuses or tunes this profile.
        # The context records also deduplicate layers with the same geometry.
        knobs = self._broadcast(
            context.records.get(profile_key) if self.bootstrap.rank == 0 else None
        )
        if not _valid_knobs(knobs):
            knobs = self._tune(profile_inputs, capacity)
            if self.bootstrap.rank == 0:
                logger.info("MegaMoE tuned: capacity=%s, knobs=%s", capacity, knobs)
        if context.records.get(profile_key) != knobs:
            context.records[profile_key] = knobs
            context.dirty = True
        # Restore tuple-valued knobs after AutoTuner's JSON round trip, as
        # FlashInfer's native cache loader does.
        knobs = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in knobs.items()
        }
        workspace = mega.create_workspace(capacity, knobs=knobs)
        mega.warmup(profile_inputs, workspace=workspace)
        self.workspaces[capacity] = workspace

    @staticmethod
    def _forward(mega, tensors, workspace=None):
        if mega.supports_output_view:
            return mega.forward(
                tensors, workspace=workspace, return_workspace_view=True
            )
        return mega.forward(tensors, workspace=workspace)

    def __call__(self, mega, tensors):
        context = _active_context
        if context is None and not self.workspaces:
            self._fallback_used = True
            return self._forward(mega, tensors)
        if context is not None and not self.workspaces:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MegaMoE autotune must complete before CUDA graph capture"
                )
            if self._fallback_used:
                raise RuntimeError(
                    "MegaMoE autotune must precede its first ordinary forward"
                )
            import torch.distributed as dist

            if (
                dist.is_initialized()
                and self.bootstrap.world_size != dist.get_world_size()
            ):
                raise RuntimeError(
                    "FlashInfer MegaMoE startup tuning currently requires the full WORLD process group"
                )
            # The startup dummy can be much larger than captured decode batches.
            # Keep small-N profiles bounded by decode, plus one prefill profile.
            capacities = set(_capacities(context.decode_num_tokens))
            capacities.add(context.prefill_num_tokens)
            for capacity in sorted(capacities):
                self._prepare_profile(mega, tensors, capacity)
            self._profile_capacities = tuple(sorted(self.workspaces))

        count = self._common_tokens(tensors)
        index = bisect_left(self._profile_capacities, count)
        if index < len(self._profile_capacities):
            return self._forward(
                mega, tensors, self.workspaces[self._profile_capacities[index]]
            )
        raise ValueError(
            f"MegaMoE received {count} tokens per rank, exceeding its prepared "
            f"decode/prefill capacity {max(self.workspaces)}"
        )
