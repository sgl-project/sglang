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

"""Ascend EPLB communication, weight staging and initial scheduling helpers."""

import logging

import torch
import torch.distributed as dist
from torch.distributed import P2POp

from sglang.srt.runtime_context import get_device, get_exec, get_parallel

logger = logging.getLogger(__name__)
_group_world = None
_device_group = None


def get_eplb_device_group():
    """Return a cached HCCL group with WORLD membership for NPU EPLB."""
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


def stage_p2p_ops(ops):
    """Stage internal-format views through ND buffers for HCCL.

    Received values are copied back into the original views after completion.
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
    return staged_ops, receive_copies


def needs_nd_staging(tensor):
    """Internal NPU formats need ordinary buffers for expert-slot copies."""
    if tensor.device.type != "npu":
        return False
    import torch_npu

    return torch_npu.get_npu_format(tensor) not in (0, 2)


def get_initial_rebalance_iterations(steady_iterations):
    """Use a short first window only for single-node trivial NPU placement."""
    moe = get_exec().moe
    if (
        get_device().device == "npu"
        and get_parallel().nnodes == 1
        and moe.ep_dispatch_algorithm == "static"
        and moe.init_expert_location == "trivial"
        and moe.elastic_ep_backend is None
        and steady_iterations > 64
    ):
        logger.info(
            "[NPU EPLB initial learning] first=64 steady=%d window=%d",
            steady_iterations,
            moe.expert_distribution_recorder_buffer_size,
        )
        return 64
    return None


def filter_unchanged_layers(layer_ids, old_map, new_map, *, elastic_enabled):
    if get_device().device != "npu" or elastic_enabled:
        return layer_ids
    # Check global maps: an unchanged local rank may still be a sender.
    return [
        layer for layer in layer_ids if not torch.equal(old_map[layer], new_map[layer])
    ]


def create_temp_buffers(sample_tensors):
    return [
        torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        if needs_nd_staging(tensor)
        else torch.empty_like(tensor)
        for tensor in sample_tensors
    ]


class ExpertWeightStaging:
    """Reuse one ND layer workspace and write back into original model storage."""

    def __init__(self, sample_tensors):
        # Per-expert NZ conversion can copy the whole backing tensor.
        self.buffers = (
            create_temp_buffers(sample_tensors)
            if any(needs_nd_staging(tensor) for tensor in sample_tensors)
            else None
        )

    def load(self, weights):
        if self.buffers is None:
            return weights
        for buffer, weight in zip(self.buffers, weights, strict=True):
            buffer.copy_(weight)
        return self.buffers

    def store(self, weights):
        if self.buffers is not None:
            for weight, buffer in zip(weights, self.buffers, strict=True):
                weight.copy_(buffer)
