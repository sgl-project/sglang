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

"""Ascend EPLB statistics and communication helpers. Safe to import on CPU."""

import logging
import os

import torch
import torch.distributed as dist
from torch.distributed import P2POp

from sglang.srt.runtime_context import get_device, get_exec, get_parallel

logger = logging.getLogger(__name__)
_group_world = None
_device_group = None


class BatchedTopKStatisticsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        capacity = int(os.environ.get("SGLANG_NPU_EPLB_STATS_MAX_ROUTES", "96"))
        if not 1 <= capacity <= 65536:
            raise ValueError("Invalid batched statistics route capacity")
        # All captured graphs write to this shared buffer.
        self._batched_routes = torch.full(
            (self._data.shape[0], capacity),
            -1,
            dtype=torch.int32,
            device=self._data.device,
        )
        self._route_offsets = [0] * self._data.shape[0]
        self._batch_collected = False
        self._has_batched_routes = False
        logger.info("[NPU EPLB batch statistics] capacity=%d routes/layer", capacity)

    def on_graph_capture_forward_start(self):
        # Capture warmups bypass the normal reset. Reset offsets without
        # recording a buffer clear into the graph.
        self._route_offsets = [0] * len(self._route_offsets)

    def get_graph_capture_state(self):
        return any(self._route_offsets)

    def on_graph_replay_forward_start(self, has_batched_routes):
        # Graph replay does not call the Python on_select_experts hook.
        self._has_batched_routes |= has_batched_routes

    def reset(self):
        super().reset()
        if self._has_batched_routes:
            self._batched_routes.fill_(-1)
        self._route_offsets = [0] * self._data.shape[0]
        self._batch_collected = False
        self._has_batched_routes = False

    def on_select_experts(self, layer_idx, topk_ids):
        count = topk_ids.numel()
        start = self._route_offsets[layer_idx]
        end = start + count
        if end > self._batched_routes.shape[1]:
            # Count overflow directly; collect() adds any earlier buffered calls.
            return super().on_select_experts(layer_idx, topk_ids)
        self._batched_routes[layer_idx, start:end].copy_(topk_ids.flatten())
        self._route_offsets[layer_idx] = end
        if count:
            self._has_batched_routes = True

    def collect(self):
        if self._has_batched_routes and not self._batch_collected:
            ids = self._batched_routes
            mask = (ids >= 0) & (ids < self._data.shape[1])
            self._data.scatter_add_(1, ids.masked_fill(~mask, 0).long(), mask.int())
            self._batch_collected = True
        return super().collect()


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
        return 64
    return None
