# Copyright 2023-2024 SGLang Team
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

"""Ascend EPLB adapters used by the shared recorder and weight updater.

Callers select these helpers after NPU distributed initialization.
Import torch_npu only when an NPU operation actually needs it.
"""

import logging
import time
from functools import wraps
from typing import Callable

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
_warmed_device_group = None


def get_eplb_device_group() -> dist.ProcessGroup:
    """Reuse TP's device group and its existing communication configuration."""
    from sglang.srt.runtime_context import get_parallel

    tp_group = get_parallel().tp_group
    # Shared statistics and migration use WORLD ranks, including root rank 0.
    if tp_group.ranks != list(range(dist.get_world_size())):
        raise ValueError(
            "NPU EPLB requires a TP group covering WORLD in rank order; "
            "TP subgroups are not supported."
        )
    return tp_group.device_group


def warmup_eplb_communication() -> None:
    """Initialize statistics and all-peer migration channels before serving."""
    global _warmed_device_group
    group = get_eplb_device_group()
    if group is _warmed_device_group:
        return

    start = time.perf_counter()
    world_size = dist.get_world_size(group)
    if world_size > 1:
        rank = dist.get_rank()
        send = torch.zeros(1, dtype=torch.int32, device="npu")
        recv = torch.empty((world_size, 1), dtype=torch.int32, device="npu")
        dist.all_reduce(send, group=group)
        ops = []
        for peer in range(world_size):
            if peer != rank:
                ops.append(dist.P2POp(dist.isend, send, peer, group=group))
                ops.append(dist.P2POp(dist.irecv, recv[peer], peer, group=group))
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        # Retain the tiny buffers until communication has actually completed.
        torch.npu.synchronize()
    _warmed_device_group = group
    logger.info(
        "[NPU EPLB] communication warmup finished: ranks=%d time=%.3fs",
        world_size,
        time.perf_counter() - start,
    )


def prepare_expert_weight_update(
    *,
    weights_by_layer: dict[int, list[torch.Tensor]],
    layer_ids: list[int],
    old_map: torch.Tensor,
    new_map: torch.Tensor,
) -> tuple[list[int], list[torch.Tensor] | None, Callable[..., None] | None]:
    """Return selected layers, receive buffers, and a single-layer update callable.

    Validate the entire chunk before allocating workspaces or moving weights.
    The callable loads internal-format weights into ND storage, invokes the
    shared migration algorithm, and writes changed local slots back in place.
    Receive buffers live only for this chunk and are reused across compatible layers.
    ND staging keeps only the current layer's conversion results.
    """
    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
    from sglang.srt.eplb.expert_location_updater import (
        update_expert_weights_single_layer,
    )

    elastic_ep_enabled = ElasticEPStateManager.instance() is not None
    if not elastic_ep_enabled:
        # A rank with unchanged local slots may still need to send to another rank.
        layer_ids = [
            layer
            for layer in layer_ids
            if not torch.equal(old_map[layer], new_map[layer])
        ]
    if not layer_ids:
        return layer_ids, None, None

    rank = dist.get_rank()
    needs_staging = _validate_migration_weights(weights_by_layer, layer_ids, rank)
    staging = ExpertWeightStaging(needs_staging)
    temp_buffers: list[torch.Tensor] = []
    group = get_eplb_device_group()

    def update_layer(*, routed_experts_weights: list[torch.Tensor], **kwargs) -> None:
        start = kwargs["rank"] * kwargs["num_local_physical_experts"]
        end = start + kwargs["num_local_physical_experts"]
        old_placement = kwargs["old_physical_to_logical_map"]
        new_placement = kwargs["new_physical_to_logical_map"]
        local_changed = old_placement[start:end] != new_placement[start:end]
        if not local_changed and not elastic_ep_enabled:
            # Any outgoing expert must appear in a changed destination slot.
            # Keep possible senders; only skip ranks with provably no work.
            local_experts = set(old_placement[start:end])
            if not any(
                old != new and new in local_experts
                for old, new in zip(old_placement, new_placement, strict=True)
            ):
                return
        # Populate the shared receive list only when this rank participates.
        if not temp_buffers:
            temp_buffers.extend(staging.create_temp_buffers(routed_experts_weights))
        migration_weights = staging.load(routed_experts_weights)
        update_expert_weights_single_layer(
            routed_experts_weights=migration_weights,
            group=group,
            **kwargs,
        )
        # Unchanged local slots may still send; only their writeback is redundant.
        if local_changed:
            staging.store(routed_experts_weights, migration_weights)

    return layer_ids, temp_buffers, update_layer


def _validate_migration_weights(
    weights_by_layer: dict[int, list[torch.Tensor]], layer_ids: list[int], rank: int
) -> list[bool]:
    """Validate chunk compatibility and return its ND staging requirements."""
    import torch_npu

    def specification(weights):
        return [
            (
                tuple(t.shape),
                t.dtype,
                t.device,
                torch_npu.get_npu_format(t) not in (0, 2) or not t.is_contiguous(),
            )
            for t in weights
        ]

    sample_layer = layer_ids[0]
    expected = specification(weights_by_layer[sample_layer])
    if not expected:
        raise ValueError(
            f"[NPU EPLB] rank={rank} layer={sample_layer}: no expert weights"
        )
    for layer in layer_ids[1:]:
        if specification(weights_by_layer[layer]) != expected:
            raise ValueError(
                f"[NPU EPLB] rank={rank} layer={layer}: migration weights must match "
                f"layer {sample_layer} in tensor count, shape, dtype, device, "
                "and ND staging requirements"
            )
    return [needs_staging for _, _, _, needs_staging in expected]


class ExpertWeightStaging:
    """Prepare ND migration weights for a validated chunk's tensor formats.

    load() prepares the tensors that the updater mutates; store() copies the
    result into model storage without changing addresses held by captured graphs.
    """

    def __init__(self, needs_staging: list[bool]):
        # Convert internal formats and pack strided ND before migrating slices.
        self._needs_staging = needs_staging

    def create_temp_buffers(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)
            for weight in weights
        ]

    def load(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        # Keep the conversion result; copy_ would add another ND-to-ND copy.
        return [
            torch.ops.npu.npu_format_cast(weight, 2).contiguous() if needed else weight
            for weight, needed in zip(weights, self._needs_staging)
        ]

    def store(
        self, weights: list[torch.Tensor], migration_weights: list[torch.Tensor]
    ) -> None:
        for weight, buffer, needed in zip(
            weights, migration_weights, self._needs_staging
        ):
            if not needed:
                continue
            restored = torch.ops.npu.npu_format_cast(
                buffer, torch.ops.npu.get_npu_format(weight)
            )
            # Keep the storage addresses referenced by captured graphs.
            weight.copy_(restored)


def isolate_npu_draft_recording(forward):
    """Keep target EPLB statistics untouched during NPU draft execution."""

    @wraps(forward)
    def wrapped(runner, *args, **kwargs):
        if runner.is_draft_worker:
            from sglang.srt.eplb.expert_distribution import (
                get_global_expert_distribution_recorder,
            )

            with get_global_expert_distribution_recorder().disable_this_region():
                return forward(runner, *args, **kwargs)
        return forward(runner, *args, **kwargs)

    return wrapped
