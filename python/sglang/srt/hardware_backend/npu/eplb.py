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

Non-NPU calls retain default communication and tensor allocation behavior.
Import torch_npu only when an NPU operation actually needs it.
"""

import logging
import time
from functools import wraps
from typing import Callable

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import get_device
from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)
_group_world = None
_device_group = None
_warmed_device_group = None


def get_eplb_device_group() -> dist.ProcessGroup | None:
    """Return the NPU EPLB group, or None to use the default group elsewhere.

    Creation is collective: all WORLD ranks must reach the first call in the
    same order. Cache against WORLD identity to handle distributed reinit.
    """
    global _group_world, _device_group
    if get_device().device != "npu" or not dist.is_initialized():
        return None

    world = dist.group.WORLD
    if _group_world is not world:
        import torch_npu

        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        # Keep migration off the forward graph's device-expanded HCCL group.
        # Use a smaller buffer to limit this separate group's memory footprint.
        options.hccl_config = {
            "hccl_buffer_size": 32,
            "hccl_op_expansion_mode": 1,
        }
        _device_group = dist.new_group(backend="hccl", pg_options=options)
        _group_world = world
        logger.info(
            "[NPU EPLB host group] world=%d buffer=32MiB", dist.get_world_size()
        )
    return _device_group


def warmup_eplb_communication() -> None:
    """Initialize statistics and all-peer migration channels before serving."""
    global _warmed_device_group
    group = get_eplb_device_group()
    if group is None or group is _warmed_device_group:
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
    update_single_layer: Callable[..., object],
) -> tuple[list[int], list[torch.Tensor] | None, Callable[..., object] | None]:
    """Return selected layers, receive buffers, and a single-layer update callable.

    Validate the entire chunk before allocating workspaces or moving weights.
    The callable loads internal-format weights into ND storage, invokes the
    shared migration algorithm, and writes changed local slots back in place.
    Receive buffers live only for this chunk and are reused across compatible layers.
    ND staging keeps only the current layer's conversion results.
    """
    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

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
    _validate_migration_weights(weights_by_layer, layer_ids, rank)
    sample = weights_by_layer[layer_ids[0]]
    staging = ExpertWeightStaging(sample, coalesce_aux=not elastic_ep_enabled)
    temp_buffers = staging.create_temp_buffers(sample)
    group = get_eplb_device_group()

    def update_layer(*, routed_experts_weights: list[torch.Tensor], **kwargs) -> object:
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
        result = update_single_layer(
            routed_experts_weights=staging.load(routed_experts_weights),
            group=group,
            **kwargs,
        )
        # Unchanged local slots may still send; only their writeback is redundant.
        if local_changed:
            staging.store(routed_experts_weights)
        return result

    return layer_ids, temp_buffers, update_layer


def _validate_migration_weights(
    weights_by_layer: dict[int, list[torch.Tensor]], layer_ids: list[int], rank: int
) -> None:
    def specification(tensor):
        return (
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            needs_nd_staging(tensor),
        )

    sample_layer = layer_ids[0]
    expected = [specification(t) for t in weights_by_layer[sample_layer]]
    if not expected:
        raise ValueError(
            f"[NPU EPLB] rank={rank} layer={sample_layer}: no expert weights"
        )
    for layer in layer_ids[1:]:
        weights = weights_by_layer[layer]
        if len(weights) != len(expected):
            raise ValueError(
                f"[NPU EPLB] rank={rank} layer={layer}: expected {len(expected)} "
                f"weight tensors from layer {sample_layer}, got {len(weights)}"
            )
        for index, (tensor, spec) in enumerate(zip(weights, expected, strict=True)):
            actual = specification(tensor)
            if actual != spec:
                raise ValueError(
                    f"[NPU EPLB] rank={rank} layer={layer} tensor={index}: incompatible "
                    f"migration workspace; (shape, dtype, device, needs_nd_staging) "
                    f"expected {spec} from layer {sample_layer}, got {actual}"
                )


def needs_nd_staging(tensor: torch.Tensor) -> bool:
    """Internal NPU formats need ordinary buffers for expert-slot copies."""
    if tensor.device.type != "npu":
        return False
    import torch_npu

    return torch_npu.get_npu_format(tensor) not in (0, 2)


class ExpertWeightStaging:
    """Keep one layer's ND conversion results in a validated migration chunk.

    load() prepares the tensors that the updater mutates; store() copies the
    result into model storage without changing addresses held by captured graphs.
    Small auxiliary matrices share per-layer buffers by dtype, reducing
    P2P operations without packing the large expert weights.
    """

    def __init__(self, sample_tensors: list[torch.Tensor], *, coalesce_aux=True):
        # Per-expert NZ conversion can copy the whole backing tensor.
        self._needs_staging = [needs_nd_staging(tensor) for tensor in sample_tensors]
        groups = {}
        for index, tensor in enumerate(sample_tensors):
            # Shape/dtype alone determine groups so ranks may use different formats.
            small_aux = (
                coalesce_aux
                and tensor.ndim == 2
                and tensor.dtype in (torch.bfloat16, torch.float32)
                and 0 < tensor.shape[1] * tensor.element_size() <= 65536
            )
            groups.setdefault(tensor.dtype if small_aux else index, []).append(index)
        self._groups = list(groups.values())
        self._packed_buffers: dict[int, torch.Tensor] = {}
        self._packed_views: dict[int, tuple[torch.Tensor, ...]] = {}
        self.buffers: list[torch.Tensor] = []

    def create_temp_buffers(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        buffers = []
        for indices in self._groups:
            weight = weights[indices[0]]
            shape = weight.shape
            if len(indices) > 1:
                shape = (weight.shape[0], sum(weights[i].shape[1] for i in indices))
            buffers.append(
                torch.empty(shape, dtype=weight.dtype, device=weight.device)
                if len(indices) > 1 or self._needs_staging[indices[0]]
                else torch.empty_like(weight)
            )
        return buffers

    def load(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        # Keep the conversion result; copy_ would add another ND-to-ND copy.
        self.buffers.clear()
        converted = [
            torch.ops.npu.npu_format_cast(weight, 2) if needed else weight
            for weight, needed in zip(weights, self._needs_staging, strict=True)
        ]
        for group_index, indices in enumerate(self._groups):
            if len(indices) > 1:
                # The updater waits for P2P before store/load on this stream.
                # Retain only small packed auxiliaries across layers in the chunk.
                buffer = torch.cat(
                    [converted[i] for i in indices],
                    dim=1,
                    out=self._packed_buffers.get(group_index),
                )
                if group_index not in self._packed_buffers:
                    self._packed_buffers[group_index] = buffer
                    # Shapes are validated for the chunk; these views remain
                    # valid while load() reuses the same packed storage.
                    self._packed_views[group_index] = buffer.split(
                        [weights[i].shape[1] for i in indices], dim=1
                    )
            else:
                buffer = converted[indices[0]]
            self.buffers.append(buffer)
        return self.buffers

    def store(self, weights: list[torch.Tensor]) -> None:
        for group_index, (indices, buffer) in enumerate(
            zip(self._groups, self.buffers, strict=True)
        ):
            if len(indices) == 1 and not self._needs_staging[indices[0]]:
                continue
            pieces = self._packed_views[group_index] if len(indices) > 1 else [buffer]
            for index, piece in zip(indices, pieces, strict=True):
                weight = weights[index]
                if self._needs_staging[index]:
                    piece = torch.ops.npu.npu_format_cast(
                        piece, torch.ops.npu.get_npu_format(weight)
                    )
                # Keep the storage addresses referenced by captured graphs.
                weight.copy_(piece)


def isolate_npu_draft_recording(forward):
    """Keep target EPLB statistics untouched throughout an NPU draft operation."""
    # Preserve the original method on other devices, with no forward-time wrapper.
    if not is_npu():
        return forward

    @wraps(forward)
    def wrapped(runner, *args, **kwargs):
        if runner.is_draft_worker and get_device().device == "npu":
            from sglang.srt.eplb.expert_distribution import (
                get_global_expert_distribution_recorder,
            )

            with get_global_expert_distribution_recorder().disable_this_region():
                return forward(runner, *args, **kwargs)
        return forward(runner, *args, **kwargs)

    return wrapped
