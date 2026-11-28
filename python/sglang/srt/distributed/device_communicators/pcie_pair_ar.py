# SPDX-License-Identifier: Apache-2.0
"""All-reduce for 4 PCIe GPUs on two switches, for decode-sized messages.

Opt-in with ``SGLANG_ENABLE_PCIE_PAIR_ALLREDUCE=1``. For hosts like a 4x A100 PCIe
box where GPUs 0-1 and 2-3 each sit behind one PCIe switch and the two switches
meet at the CPU, without NVLink. CustomAllReduceV2 needs full NVLink there, and
the legacy custom all-reduce is limited to 2 PCIe GPUs, so every per-layer
reduction goes to NCCL, whose ring crosses the switches twice and pays per-hop
latency on every message.

The kernel (``csrc/distributed/pcie_pair_all_reduce.cuh``) works in three steps:
reduce-scatter within each switch pair, exchange the pair sums across the
switches, all-gather within the pair. Each step is a P2P store into the
receiver's IPC buffer plus one release flag; per-block 64-bit epochs in device
memory make it CUDA-graph safe.

Numerics: every element is ``bf16(fp32(r0 + r1) + fp32(r2 + r3))`` (default;
``SGLANG_PCIE_PAIR_AR_FP32=0`` also rounds the pair sums to bf16). The order is
fixed and does not depend on the message size or the launch shape, so the result
is identical on all ranks and a row's bits do not depend on which other rows were
reduced with it. That makes it usable under deterministic inference, unlike NCCL
(the ring's split follows the message size) and the shape-tuned PCIe-IPC kernels.

Measured on 4x A100 80GB PCIe (Gen3 switches), hidden 5120 bf16, per call inside
a CUDA graph (microseconds):

====  ======  ======================
rows  NCCL    pair (fp32 pair sums)
====  ======  ======================
1     35.1    18.1
6     51.0    36.1
24    126.6   123.4
64    214.7   314.7 (bf16 sums: 203.6)
4096  10112   about 12000
====  ======  ======================

Large messages stay on NCCL: each GPU pushes 1.5x the message here, and SM peer
stores reach about 6.7 GB/s on a Gen3 pair link. ``SGLANG_PCIE_PAIR_AR_MAX_BYTES``
sets the bound (default 24 rows of hidden 5120, where fp32 pair sums still win).
"""

from __future__ import annotations

import ctypes
import functools
import logging
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

WORLD_SIZE = 4
KSTAGES = 3
SLOTS = 2
_ELIGIBLE_GROUP_NAMES = frozenset({"tp"})

# Launch shape by message size, measured on 4x A100 PCIe (hidden 5120 rows; the
# bits are the same for every choice, so this only affects speed).
_ROW_BYTES = 5120 * 2
_PLAN: Tuple[Tuple[int, int, bool], ...] = (
    (2 * _ROW_BYTES, 1, True),
    (4 * _ROW_BYTES, 1, False),
    (12 * _ROW_BYTES, 2, False),
)
_PLAN_LARGE = (4, False)


def plan(nbytes: int) -> Tuple[int, bool]:
    """(blocks, one_shot) for a message of ``nbytes``."""
    for limit, blocks, one_shot in _PLAN:
        if nbytes <= limit:
            return blocks, one_shot
    return _PLAN_LARGE


def eligible_group(*, group_name: Optional[str], world_size: int) -> bool:
    """Whether ``GroupCoordinator`` should try to build this backend for a group."""
    return (
        envs.SGLANG_ENABLE_PCIE_PAIR_ALLREDUCE.get()
        and world_size == WORLD_SIZE
        and group_name in _ELIGIBLE_GROUP_NAMES
    )


def _bus_id(device: torch.device) -> str:
    props = torch.cuda.get_device_properties(device)
    return (
        f"{props.pci_domain_id:08x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    )


def is_two_switch_pairs(bus_ids: List[str]) -> Optional[bool]:
    """True if ranks 0-1 and 2-3 each share a PCIe switch and the pairs don't.

    None when NVML is unavailable.
    """
    try:
        import pynvml
    except ImportError:
        return None
    pynvml.nvmlInit()
    try:
        h = [pynvml.nvmlDeviceGetHandleByPciBusId(b) for b in bus_ids]
        level = pynvml.nvmlDeviceGetTopologyCommonAncestor
        near = pynvml.NVML_TOPOLOGY_MULTIPLE  # PIX or PXB: the same switch tree
        return all(level(h[a], h[a ^ 1]) <= near for a in (0, 2)) and all(
            level(h[a], h[a ^ 2]) > near for a in (0, 1)
        )
    finally:
        pynvml.nvmlShutdown()


@functools.lru_cache(maxsize=None)
def _warn_topology(reason: str) -> None:
    logger.warning(
        "PCIe pair all-reduce disabled: %s; reductions stay on NCCL.", reason
    )


class PciePairCommunicator:
    """Adapter between ``GroupCoordinator`` and the PCIe pair all-reduce kernel."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device | int,
        *,
        max_blocks: int = 16,
        half_bytes: int = 32768,
        threads: int = 512,
    ):
        self.disabled = True
        self.max_bytes = 0
        self._pointers: Optional[List[int]] = None
        self._group = group

        world_size = dist.get_world_size(group=group)
        if world_size != WORLD_SIZE:
            return
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        if device.type != "cuda" or torch.version.cuda is None:
            return

        rank = dist.get_rank(group=group)
        buses = [None] * world_size
        dist.all_gather_object(buses, _bus_id(device), group=group)
        topo = is_two_switch_pairs(buses)
        if not topo:
            _warn_topology(
                "NVML unavailable"
                if topo is None
                else "ranks 0-1 and 2-3 are not two PCIe switch pairs"
            )
            return

        from sglang.srt.distributed.device_communicators.cuda_wrapper import (
            CudaRTLibrary,
        )
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.wide = envs.SGLANG_PCIE_PAIR_AR_FP32.get()
        self.max_bytes = envs.SGLANG_PCIE_PAIR_AR_MAX_BYTES.get()
        self.device = device
        self.pos = rank & 1
        self.max_blocks = max_blocks
        self.threads = threads
        # Wide: the cross-switch stage carries fp32, so the segments double to keep
        # the same elements per iteration (and the same one-shot limit) as bf16.
        half_bytes = half_bytes * 2 if self.wide else half_bytes
        self.half_cap = half_bytes // 16
        flag_bytes = KSTAGES * max_blocks * 8
        self.data_off = -(-flag_bytes // 4096) * 4096
        size = self.data_off + SLOTS * KSTAGES * max_blocks * half_bytes

        with torch.cuda.device(device):
            # cudaMalloc'd, zeroed (flags start at epoch 0), opened by every peer.
            self._pointers = CustomAllreduce.create_shared_buffer(size, group=group)
            lib = CudaRTLibrary()
            lib.cudaMemset(ctypes.c_void_p(self._pointers[rank]), 0, size)
            lib.cudaDeviceSynchronize()
            self.counters = torch.zeros(max_blocks, dtype=torch.int64, device=device)
        dist.barrier(group=group)
        self.self_buf = self._pointers[rank]
        self.partner_buf = self._pointers[rank ^ 1]
        self.cross_buf = self._pointers[rank ^ 2]
        self.disabled = False
        logger.info(
            "PCIe pair all-reduce enabled (up to %d bytes, %s pair sums)",
            self.max_bytes,
            "fp32" if self.wide else "bf16",
        )

    def should_pcie_pair_ar(self, inp: torch.Tensor) -> bool:
        return (
            not self.disabled
            and inp.is_cuda
            and inp.dtype is torch.bfloat16
            and inp.is_contiguous()
            and inp.numel() % 8 == 0
            and inp.data_ptr() % 16 == 0
            and inp.numel() * 2 <= self.max_bytes
        )

    def pcie_pair_all_reduce(
        self,
        inp: torch.Tensor,
        *,
        blocks: Optional[int] = None,
        one_shot: Optional[bool] = None,
    ) -> torch.Tensor:
        """Out-of-place all-reduce. ``blocks``/``one_shot`` override the plan (tests)."""
        from sglang.kernels.ops.communication.pcie_pair_all_reduce import (
            pcie_pair_all_reduce,
        )

        auto_blocks, auto_one = plan(inp.numel() * 2)
        blocks = blocks or auto_blocks
        one_shot = auto_one if one_shot is None else one_shot
        cap = self.half_cap // 2 if self.wide else self.half_cap
        one_shot = one_shot and -(-inp.numel() // (8 * blocks)) <= cap
        out = torch.empty_like(inp)
        pcie_pair_all_reduce(
            inp,
            out,
            self.counters,
            self_buf=self.self_buf,
            partner_buf=self.partner_buf,
            cross_buf=self.cross_buf,
            pos=self.pos,
            max_blocks=self.max_blocks,
            half_cap=self.half_cap,
            data_off=self.data_off,
            blocks=blocks,
            threads=self.threads,
            one_shot=one_shot,
            wide=self.wide,
        )
        return out

    def destroy(self) -> None:
        if self._pointers is not None:
            from sglang.srt.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )

            CustomAllreduce.free_shared_buffer(self._pointers, group=self._group)
            self._pointers = None
        self.disabled = True
