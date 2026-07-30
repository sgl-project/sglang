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

"""Consumer-direct DCP Top-K merge over owner-local CUDA VMM storage."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.profiler import record_function

from sglang.srt.distributed.device_communicators.peer_memory import (
    RankMajorPeerBuffer,
    create_rank_major_peer_buffer,
    make_rank_major_tensor_view,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)

_HEADER_BYTES = 256
_MAX_FENCE_SPINS = 100_000_000
DCP_TOPK_VMM_MAX_ROWS = 512
_logged_rows: set[int] = set()


@triton.jit
def _trap_if_nonzero(value):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred failed;
            setp.ne.u32 failed, $1, 0;
            @failed trap;
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,r",
        args=[value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_writable_kernel(
    peer_flags,
    peer_stride,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    block_size: tl.constexpr,
    max_spins: tl.constexpr,
):
    write_seq = tl.atomic_add(
        peer_flags + my_rank * peer_stride,
        0,
        sem="acquire",
        scope="sys",
    )
    peer = tl.arange(0, block_size)
    mask = peer < world_size
    observed = tl.atomic_add(
        peer_flags + peer * peer_stride + 1,
        0,
        mask=mask,
        sem="acquire",
        scope="sys",
    )
    pending = tl.max(tl.where(mask & (observed < write_seq), 1, 0))
    spins = 0
    while (pending != 0) & (spins < max_spins):
        observed = tl.atomic_add(
            peer_flags + peer * peer_stride + 1,
            0,
            mask=mask,
            sem="acquire",
            scope="sys",
        )
        pending = tl.max(tl.where(mask & (observed < write_seq), 1, 0))
        spins += 1
    _trap_if_nonzero(pending)


@triton.jit
def _publish_and_wait_kernel(
    peer_flags,
    peer_stride,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    block_size: tl.constexpr,
    max_spins: tl.constexpr,
):
    epoch = (
        tl.atomic_add(
            peer_flags + my_rank * peer_stride,
            1,
            sem="release",
            scope="sys",
        )
        + 1
    )
    peer = tl.arange(0, block_size)
    mask = peer < world_size
    observed = tl.atomic_add(
        peer_flags + peer * peer_stride,
        0,
        mask=mask,
        sem="acquire",
        scope="sys",
    )
    pending = tl.max(tl.where(mask & (observed < epoch), 1, 0))
    spins = 0
    while (pending != 0) & (spins < max_spins):
        observed = tl.atomic_add(
            peer_flags + peer * peer_stride,
            0,
            mask=mask,
            sem="acquire",
            scope="sys",
        )
        pending = tl.max(tl.where(mask & (observed < epoch), 1, 0))
        spins += 1
    _trap_if_nonzero(pending)


@triton.jit
def _ack_kernel(peer_flags, peer_stride, my_rank: tl.constexpr):
    tl.atomic_add(
        peer_flags + my_rank * peer_stride + 1,
        1,
        sem="release",
        scope="sys",
    )


@dataclass
class DcpTopKVmmWorkspace:
    rank: int
    world_size: int
    max_rows: int
    local_candidates_count: int
    group: GroupCoordinator
    device: torch.device
    allocation: RankMajorPeerBuffer
    local_candidates: torch.Tensor
    peer_candidates: torch.Tensor
    peer_flags: torch.Tensor

    @property
    def physical_bytes_per_rank(self) -> int:
        return self.allocation.bytes_per_rank

    def _validate_live(self) -> None:
        if self.allocation.closed:
            raise RuntimeError("DCP Top-K VMM workspace is closed")
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError(
                "DCP Top-K VMM current device changed after initialization: "
                f"workspace={self.device}, "
                f"current=cuda:{torch.cuda.current_device()}"
            )

    def merge(
        self,
        logits: torch.Tensor,
        local_indices: torch.Tensor,
        topk: int,
        *,
        dcp_rank: int,
        dcp_size: int,
        pipelined: bool = False,
    ) -> torch.Tensor:
        from sglang.kernels.ops.attention.dsa.dcp_indexer_cutedsl import (
            pack_dcp_topk_candidates_cutedsl,
            stable_topk_from_rank_major_candidates_cutedsl,
        )

        self._validate_live()
        if dcp_rank != self.rank or dcp_size != self.world_size:
            raise RuntimeError(
                "DCP Top-K VMM geometry changed after initialization: "
                f"workspace=({self.rank}, {self.world_size}), "
                f"request=({dcp_rank}, {dcp_size})"
            )
        rows = local_indices.shape[0]
        if rows <= 0 or rows > self.max_rows:
            raise RuntimeError(
                "DCP Top-K VMM row bound violated: "
                f"max_rows={self.max_rows}, requested={rows}"
            )
        if local_indices.shape[1] != self.local_candidates_count:
            raise RuntimeError(
                "DCP Top-K VMM candidate width changed: "
                f"workspace={self.local_candidates_count}, "
                f"request={local_indices.shape[1]}"
            )
        if topk != self.local_candidates_count:
            raise RuntimeError(
                "DCP Top-K VMM requires local candidate width == global Top-K: "
                f"{self.local_candidates_count} != {topk}"
            )
        if rows not in _logged_rows:
            _logged_rows.add(rows)
            logger.debug(
                "Executing consumer-direct CUDA VMM DCP Top-K for rows=%d",
                rows,
            )

        output = torch.empty(
            (rows, topk), dtype=torch.int32, device=local_indices.device
        )
        if not pipelined:
            with record_function("dcp.topk_vmm.wait_reuse"):
                _wait_writable_kernel[(1,)](
                    self.peer_flags,
                    self.peer_flags.stride(0),
                    my_rank=self.rank,
                    world_size=self.world_size,
                    block_size=triton.next_power_of_2(self.world_size),
                    max_spins=_MAX_FENCE_SPINS,
                )
        with record_function("dcp.topk_vmm.pack"):
            pack_dcp_topk_candidates_cutedsl(
                logits,
                local_indices,
                self.local_candidates[:rows],
                dcp_rank,
                dcp_size,
                None,
            )
        with record_function("dcp.topk_vmm.publish"):
            _publish_and_wait_kernel[(1,)](
                self.peer_flags,
                self.peer_flags.stride(0),
                my_rank=self.rank,
                world_size=self.world_size,
                block_size=triton.next_power_of_2(self.world_size),
                max_spins=_MAX_FENCE_SPINS,
            )
        with record_function("dcp.topk_vmm.direct_select"):
            stable_topk_from_rank_major_candidates_cutedsl(
                self.peer_candidates[:, :rows],
                topk,
                output,
            )
        if not pipelined:
            with record_function("dcp.topk_vmm.ack"):
                _ack_kernel[(1,)](
                    self.peer_flags,
                    self.peer_flags.stride(0),
                    my_rank=self.rank,
                )
        return output

    def close(self) -> None:
        if self.allocation.closed:
            return
        torch.cuda.synchronize()
        dist.barrier(group=self.group.cpu_group)
        del self.peer_flags
        del self.peer_candidates
        del self.local_candidates
        self.allocation.close()


def create_dcp_topk_vmm_workspace(
    max_rows: int,
    local_candidates: int,
    group: GroupCoordinator,
) -> DcpTopKVmmWorkspace:
    if group.world_size <= 1:
        raise RuntimeError("DCP Top-K VMM requires dcp_size > 1")
    if max_rows <= 0 or local_candidates <= 0:
        raise ValueError(
            "DCP Top-K VMM dimensions must be positive: "
            f"max_rows={max_rows}, local_candidates={local_candidates}"
        )
    if (group.world_size * local_candidates) % 512:
        raise RuntimeError(
            "DCP Top-K VMM exact selector requires total candidates to be "
            "a multiple of 512, got "
            f"{group.world_size} * {local_candidates}"
        )

    payload_bytes = max_rows * local_candidates * 2 * torch.float32.itemsize
    requested_bytes = _HEADER_BYTES + payload_bytes
    allocation = create_rank_major_peer_buffer(
        requested_bytes,
        group=group.cpu_group,
        device=group.device,
        require_native_atomics=True,
    )
    allocation.local_view.zero_()
    torch.cuda.synchronize()
    dist.barrier(group=group.cpu_group)

    local_flags = allocation.local_view[: 2 * torch.int32.itemsize].view(torch.int32)
    local_candidates_tensor = (
        allocation.local_view[_HEADER_BYTES : _HEADER_BYTES + payload_bytes]
        .view(torch.float32)
        .view(max_rows, local_candidates, 2)
    )
    return DcpTopKVmmWorkspace(
        rank=group.rank_in_group,
        world_size=group.world_size,
        max_rows=max_rows,
        local_candidates_count=local_candidates,
        group=group,
        device=group.device,
        allocation=allocation,
        local_candidates=local_candidates_tensor,
        peer_candidates=make_rank_major_tensor_view(
            allocation, local_candidates_tensor
        ),
        peer_flags=make_rank_major_tensor_view(allocation, local_flags),
    )


_workspaces: dict[int, DcpTopKVmmWorkspace] = {}
_workspace_failed = False


def get_dcp_topk_vmm_workspace(
    max_rows: int,
    local_candidates: int,
    group: GroupCoordinator,
    *,
    workspace_slot: int = 0,
) -> DcpTopKVmmWorkspace:
    global _workspace_failed
    if workspace_slot < 0:
        raise ValueError(f"workspace_slot must be non-negative, got {workspace_slot}")
    if _workspace_failed:
        raise RuntimeError("DCP Top-K VMM workspace is unavailable")
    workspace = _workspaces.get(workspace_slot)
    if workspace is not None:
        actual = (
            workspace.max_rows,
            workspace.local_candidates_count,
            workspace.world_size,
            workspace.rank,
            workspace.device,
        )
        requested = (
            max_rows,
            local_candidates,
            group.world_size,
            group.rank_in_group,
            group.device,
        )
        if actual != requested or workspace.group is not group:
            raise RuntimeError(
                "DCP Top-K VMM workspace identity changed: "
                f"actual={actual}, requested={requested}, "
                f"same_group={workspace.group is group}, "
                f"workspace_slot={workspace_slot}"
            )
        return workspace
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "DCP Top-K VMM workspace was not initialized before CUDA graph " "capture"
        )
    try:
        workspace = create_dcp_topk_vmm_workspace(
            max_rows,
            local_candidates,
            group,
        )
        _workspaces[workspace_slot] = workspace
    except Exception as error:
        _workspace_failed = True
        raise RuntimeError(
            "DCP Top-K VMM initialization failed; refusing to fall back after "
            "selecting the VMM route"
        ) from error
    logger.info(
        "Initialized DCP Top-K VMM workspace: slot=%d, max_rows=%d, "
        "local_candidates=%d, physical_bytes_per_rank=%d",
        workspace_slot,
        max_rows,
        local_candidates,
        workspace.physical_bytes_per_rank,
    )
    return workspace


def init_dcp_topk_vmm_workspace(
    group: GroupCoordinator,
    local_candidates: int,
    *,
    max_rows: int = DCP_TOPK_VMM_MAX_ROWS,
    workspace_slots: int = 2,
) -> None:
    if workspace_slots <= 0:
        raise ValueError(f"workspace_slots must be positive, got {workspace_slots}")
    for workspace_slot in range(workspace_slots):
        get_dcp_topk_vmm_workspace(
            max_rows,
            local_candidates,
            group,
            workspace_slot=workspace_slot,
        )


def merge_owner_topk_vmm(
    logits: torch.Tensor,
    local_indices: torch.Tensor,
    topk: int,
    *,
    dcp_rank: int,
    dcp_size: int,
    workspace_slot: int = 0,
    pipelined: bool = False,
) -> torch.Tensor:
    from sglang.srt.runtime_context import get_parallel

    workspace = get_dcp_topk_vmm_workspace(
        DCP_TOPK_VMM_MAX_ROWS,
        local_indices.shape[1],
        get_parallel().dcp_group,
        workspace_slot=workspace_slot,
    )
    return workspace.merge(
        logits,
        local_indices,
        topk,
        dcp_rank=dcp_rank,
        dcp_size=dcp_size,
        pipelined=pipelined,
    )


def close_dcp_topk_vmm_workspace() -> None:
    global _workspace_failed
    for workspace in _workspaces.values():
        workspace.close()
    _workspaces.clear()
    _workspace_failed = False
