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

import torch
import torch.distributed as dist
from torch.distributed import P2POp

from sglang.srt.runtime_context import get_device

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
        # EPLB needs a smaller buffer than the MoE dispatch group.
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


def execute_p2p_ops(ops: list[P2POp]) -> None:
    """Wait for one P2P batch, then copy staged receives back to their targets.

    The main updater already stages whole NZ layers. Keep this fallback for
    callers of update_expert_weights_single_layer that pass internal views.
    """
    staged_ops, receive_copies = [], []
    for op in ops:
        tensor = op.tensor
        if needs_nd_staging(tensor):
            buffer = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            if op.op == torch.distributed.isend:
                buffer.copy_(tensor)
            else:
                receive_copies.append((tensor, buffer))
            op = P2POp(op.op, buffer, op.peer, op.group, op.tag)
        staged_ops.append(op)
    for request in dist.batch_isend_irecv(staged_ops):
        request.wait()
    for target, buffer in receive_copies:
        target.copy_(buffer)


def needs_nd_staging(tensor: torch.Tensor) -> bool:
    """Internal NPU formats need ordinary buffers for expert-slot copies."""
    if tensor.device.type != "npu":
        return False
    import torch_npu

    return torch_npu.get_npu_format(tensor) not in (0, 2)


def filter_unchanged_layers(layer_ids, old_map, new_map, *, elastic_enabled):
    if get_device().device != "npu" or elastic_enabled:
        return layer_ids
    # Check global maps: an unchanged local rank may still be a sender.
    return [
        layer for layer in layer_ids if not torch.equal(old_map[layer], new_map[layer])
    ]


def create_temp_buffers(sample_tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Allocate slot-copy buffers, using ordinary storage for internal NPU formats."""
    return [
        torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        if needs_nd_staging(tensor)
        else torch.empty_like(tensor)
        for tensor in sample_tensors
    ]


class ExpertWeightStaging:
    """Reuse one ND workspace across layers with matching weight shapes.

    load() prepares the tensors that the updater mutates; store() copies the
    result into model storage without changing addresses held by captured graphs.
    Ordinary weights pass through unchanged and need no extra workspace.
    """

    def __init__(self, sample_tensors: list[torch.Tensor]):
        # Per-expert NZ conversion can copy the whole backing tensor.
        self.buffers = (
            create_temp_buffers(sample_tensors)
            if any(needs_nd_staging(tensor) for tensor in sample_tensors)
            else None
        )

    def load(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.buffers is None:
            return weights
        for buffer, weight in zip(self.buffers, weights, strict=True):
            buffer.copy_(weight)
        return self.buffers

    def store(self, weights: list[torch.Tensor]) -> None:
        if self.buffers is not None:
            for weight, buffer in zip(weights, self.buffers, strict=True):
                weight.copy_(buffer)


def isolate_npu_draft_recording(forward):
    """Keep target EPLB statistics untouched throughout an NPU draft operation."""

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
