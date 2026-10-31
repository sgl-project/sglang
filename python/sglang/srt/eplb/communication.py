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
"""Communication shared by EPLB statistics and expert-weight migration."""

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import get_device, get_exec, get_parallel

_warmed_group = None


def get_eplb_group():
    """Reuse a WORLD-equivalent TP device group, otherwise keep default WORLD."""
    if not dist.is_initialized():
        return None
    tp_group = get_parallel().tp_group
    # EPLB roots, peer IDs, and expert-slot maps currently use WORLD ranks.
    # Elastic membership and TP subgroups retain the existing communication path.
    if (
        get_exec().moe.elastic_ep_backend
        or tp_group is None
        or tp_group.ranks != list(range(dist.get_world_size()))
        or dist.get_backend(tp_group.device_group) != dist.get_backend()
    ):
        return None
    return tp_group.device_group


def warmup_eplb_communication(group):
    """Initialize collective and all-peer P2P channels once before serving."""
    global _warmed_group
    if group is _warmed_group:
        return

    world_size = dist.get_world_size(group)
    if world_size > 1:
        device = get_device().device
        rank = dist.get_rank()
        send = torch.zeros(1, dtype=torch.int32, device=device)
        recv = torch.empty((world_size, 1), dtype=torch.int32, device=device)
        dist.all_reduce(send, group=group)
        # A rank with no transfers may skip later migrations. Join all peers
        # here so the group's first batch_isend_irecv includes every rank.
        ops = []
        for peer in range(world_size):
            if peer != rank:
                ops.append(dist.P2POp(dist.isend, send, peer, group=group))
                ops.append(dist.P2POp(dist.irecv, recv[peer], peer, group=group))
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        # Retain the startup buffers until the device has finished using them.
        torch.get_device_module(device).synchronize()
    _warmed_group = group
