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
from functools import wraps
from typing import Callable

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import get_device
from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)
_group_world = None
_device_group = None


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
    Workspaces live only for this chunk and are reused across compatible layers.
    """
    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

    if ElasticEPStateManager.instance() is None:
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
    temp_buffers = create_temp_buffers(sample)
    staging = ExpertWeightStaging(sample)
    group = get_eplb_device_group()
    if logger.isEnabledFor(logging.DEBUG):
        # Logical tensor bytes exclude allocator padding and cached memory.
        workspace_bytes = sum(t.numel() * t.element_size() for t in temp_buffers) + sum(
            t.numel() * t.element_size() for t in staging.buffers if t is not None
        )
        logger.debug(
            "[NPU EPLB] rank=%d layers=%d workspace_logical_bytes=%d",
            rank,
            len(layer_ids),
            workspace_bytes,
        )

    def update_layer(*, routed_experts_weights: list[torch.Tensor], **kwargs) -> object:
        start = kwargs["rank"] * kwargs["num_local_physical_experts"]
        end = start + kwargs["num_local_physical_experts"]
        local_changed = (
            kwargs["old_physical_to_logical_map"][start:end]
            != kwargs["new_physical_to_logical_map"][start:end]
        )
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
    for layer in layer_ids:
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


def create_temp_buffers(sample_tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Allocate slot-copy buffers, using ordinary storage for internal NPU formats."""
    return [
        torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        if needs_nd_staging(tensor)
        else torch.empty_like(tensor)
        for tensor in sample_tensors
    ]


class ExpertWeightStaging:
    """Reuse ND buffers for internal-format tensors in a validated layer chunk.

    load() prepares the tensors that the updater mutates; store() copies the
    result into model storage without changing addresses held by captured graphs.
    Ordinary tensors pass through directly, even in mixed-format weight lists.
    """

    def __init__(self, sample_tensors: list[torch.Tensor]):
        # Per-expert NZ conversion can copy the whole backing tensor.
        self.buffers: list[torch.Tensor | None] = [
            torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            if needs_nd_staging(tensor)
            else None
            for tensor in sample_tensors
        ]

    def load(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            buffer.copy_(weight) if buffer is not None else weight
            for buffer, weight in zip(self.buffers, weights, strict=True)
        ]

    def store(self, weights: list[torch.Tensor]) -> None:
        for weight, buffer in zip(weights, self.buffers, strict=True):
            if buffer is not None:
                weight.copy_(buffer)


def isolate_npu_draft_recording(forward):
    """Keep target EPLB statistics untouched throughout an NPU draft operation."""
    # Preserve the original method on other devices, with no forward-time wrapper.
    if not is_npu():
        return forward

    @wraps(forward)
    def wrapped(runner, *args, **kwargs):
        if get_device().device == "npu" and runner.is_draft_worker:
            from sglang.srt.eplb.expert_distribution import (
                get_global_expert_distribution_recorder,
            )

            with get_global_expert_distribution_recorder().disable_this_region():
                return forward(runner, *args, **kwargs)
        return forward(runner, *args, **kwargs)

    return wrapped
